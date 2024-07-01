import os
import pandas as pd
import numpy as np
from preprocessing import PreProcessing, Statistics, ExtractionFeatures
from class_reequilibrage_and_select_feature import DataLoader, ResamplingPipeline, ModelEvaluator, DataSaver, FileProcessor, DirectoryProcessor, FeatureRankingProcessor
from machine_learning import ModelTraining

def batch_process_and_store_c3d_files(root_directory, statistics_directory):
    concat_stats_df = pd.DataFrame()
    for dirpath, _, filenames in os.walk(root_directory):
        c3d_files = [f for f in filenames if f.endswith('.c3d')]
        if not c3d_files:
            continue
        for file_index, file_name in enumerate(c3d_files):
            file_path = os.path.join(dirpath, file_name)
            print(f"Processing file {file_index + 1}/{len(c3d_files)}: {file_name}")
            subdirectory = '2_ON' if '_ON_' in file_name else '1_OFF'
            extraction_dir = os.path.join(root_directory, subdirectory, 'extraction_features_by_file')
            os.makedirs(extraction_dir, exist_ok=True)
            print(f"Created/verified subdirectory: {extraction_dir}")
            try:
                detector = PreProcessing(file_path)
                detector.creation_json_grace_c3d()
                detector.extract_data_interval()
                detector.normalize_data()
                detector.decoupage_en_fenetres()
                detector.label_fenetre()
                detector.association_label_fenetre_data()
                data = detector.concat_label_fenetre_data()
                stats_processor = Statistics(file_path, data)
                statistiques_resultat = stats_processor.stats()
                concat_stats_df = pd.concat([concat_stats_df, statistiques_resultat], ignore_index=True)
                extraction_features = ExtractionFeatures(data)
                extraction_features.enlever_derniere_ligne_et_colonne_label()
                data_features = extraction_features.dataframe_caracteristiques_final()
                features_output_file = os.path.join(extraction_dir, f"{os.path.splitext(file_name)[0]}_extraction_features.csv")
                data_features.to_csv(features_output_file, index=False)
                print(f"Features CSV saved to {features_output_file}")
            except Exception as e:
                print(f"An error occurred while processing file {file_name}: {e}")
    os.makedirs(statistics_directory, exist_ok=True)
    final_stats_file = os.path.join(statistics_directory, 'final_statistics_dernier_fichier.csv')
    concat_stats_df.to_csv(final_stats_file, index=False)
    print(f"Final statistics have been saved to {final_stats_file}")
    print("All files have been processed.\n")
    return data_features, concat_stats_df

def combine_csv_files(source_directory, output_directory, category, patient_id):
    combined_df = pd.DataFrame()
    extraction_features_dir = os.path.join(source_directory, 'extraction_features_by_file')
    print(f"Combining {category} CSV files from: {extraction_features_dir}")
    for file_name in os.listdir(extraction_features_dir):
        if file_name.endswith('_extraction_features.csv'):
            file_path = os.path.join(extraction_features_dir, file_name)
            print(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    print(f"All {category} CSV files have been combined. Saving to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    combined_file_path = os.path.join(output_directory, f'{patient_id}_all_extraction_{category}.csv')
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Combined {category} CSV saved to {combined_file_path}")

def combine_on_off_files(on_output_directory, off_output_directory, combined_directory, patient_id):
    on_file_path = os.path.join(on_output_directory, f'{patient_id}_all_extraction_ON.csv')
    off_file_path = os.path.join(off_output_directory, f'{patient_id}_all_extraction_OFF.csv')
    print(f"Starting to combine ON and OFF CSV files:\nON: {on_file_path}\nOFF: {off_file_path}")
    on_df = pd.read_csv(on_file_path)
    off_df = pd.read_csv(off_file_path)
    combined_df = pd.concat([on_df, off_df], ignore_index=True)
    os.makedirs(combined_directory, exist_ok=True)
    combined_file_path = os.path.join(combined_directory, f'{patient_id}_all_extraction_ON_OFF.csv')
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Finished combining ON and OFF CSV saved to {combined_file_path}\n")

def process_patient(root_directory, statistics_directory, top_n_values, methods):
    patient_id = os.path.basename(root_directory)
    print(f"Processing patient with ID: {patient_id}")
    print("Starting batch processing and storing C3D files...")
    data_features, concat_stats_df = batch_process_and_store_c3d_files(root_directory, statistics_directory)
    print("Finished batch processing and storing C3D files.")
    on_source_directory = os.path.join(root_directory, '2_ON')
    on_output_directory = os.path.join(root_directory, '2_ON', 'all_extraction_ON')
    print("Starting to combine ON CSV files...")
    combine_csv_files(on_source_directory, on_output_directory, 'ON', patient_id)
    print("Finished combining ON CSV files.\n")
    off_source_directory = os.path.join(root_directory, '1_OFF')
    off_output_directory = os.path.join(root_directory, '1_OFF', 'all_extraction_OFF')
    print("Starting to combine OFF CSV files...")
    combine_csv_files(off_source_directory, off_output_directory, 'OFF', patient_id)
    print("Finished combining OFF CSV files.\n")
    combined_directory = os.path.join(root_directory, '3_ON_OFF')
    combine_on_off_files(on_output_directory, off_output_directory, combined_directory, patient_id)
    combined_file_path = os.path.join(combined_directory, f'{patient_id}_all_extraction_ON_OFF.csv')
    processor_over = FileProcessor(combined_file_path)
    processor_over.process_file_over(root_directory, os.path.join(root_directory, 'best_combinations_over100.csv'))
    processor_optimise = FileProcessor(combined_file_path)
    processor_optimise.process_file_optimise(root_directory, os.path.join(root_directory, 'best_combinations_optimize.csv'))
    processor_raw = FileProcessor(combined_file_path)
    processor_raw.save_raw_data_splits(root_directory)
    train_folder = os.path.join(root_directory, '4_train_ON_OFF')
    output_folder = os.path.join(root_directory, '6_classement_features_ON_OFF')
    feature_ranking_processor = FeatureRankingProcessor(train_folder, output_folder)
    for method in methods:
        feature_ranking_processor.apply_relief(data_type=method)
    test_folder = os.path.join(root_directory, '5_test_ON_OFF')
    output_folder = os.path.join(root_directory, '7_resultats_machine_learning')
    model_trainer = ModelTraining(train_folder, test_folder, output_folder, output_folder)
    for method in methods:
        train_data = model_trainer.load_train(method)
        test_data = model_trainer.load_test(method)
        grouped_data = model_trainer.group_data([train_data, test_data])
        feature_importances = model_trainer.load_feature_importances(method)
        results = model_trainer.train_models(grouped_data, feature_importances, top_n_values, method)

def main():
    patients_root_directories = [
        'C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01'
    ]
    statistics_directory = 'C:/Users/antho/Documents/MEMOIRE_M2/statistiques_audeline/'
    top_n_values = [10, 20, 30]  # Vous pouvez ajuster ces valeurs selon vos besoins
    methods = ['brut', 'optimise', 'over100']  # Les méthodes que vous souhaitez utiliser
    for root_directory in patients_root_directories:
        process_patient(root_directory, statistics_directory, top_n_values, methods)

if __name__ == "__main__":
    main()
