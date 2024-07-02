import os
import pandas as pd
from preprocessing import PreProcessing, Statistics, ExtractionFeatures
from rebalancing_and_select_features import FileProcessor, FeatureRankingProcessor
from machine_learning import ModelTraining

def batch_process_and_store_c3d_files(root_directory, statistics_directory):
    """
    Batch processes and stores C3D files ON and OFF in a given directory
    
    Args:
        root_directory (str): Root directory containing the C3D files to process with ON and OFF.
        statistics_directory (str): Directory where statistics will be stored.

    Returns:
        tuple: DataFrame of extracted features and concatenated statistics DataFrame.
    """
    concat_stats_df = pd.DataFrame()  # Initialize an empty DataFrame for concatenated statistics
    identifiant = os.path.basename(root_directory)  # Extract the identifier from the root directory name

    # Walk through the directory tree rooted at root_directory
    for dirpath, _, filenames in os.walk(root_directory):
        c3d_files = [f for f in filenames if f.endswith('.c3d')]
        if not c3d_files:  # Skip directories without C3D files
            continue

        # Process each C3D file found
        for file_index, file_name in enumerate(c3d_files):
            file_path = os.path.join(dirpath, file_name)
            print(f"Processing file {file_index + 1}/{len(c3d_files)}: {file_name}")

            # Determine the subdirectory based on the filename
            subdirectory = '2_ON' if '_ON_' in file_name else '1_OFF'
            extraction_dir = os.path.join(root_directory, subdirectory, 'extraction_features_by_file')
            os.makedirs(extraction_dir, exist_ok=True)  # Create the subdirectory if it doesn't exist
            print(f"Created/verified subdirectory: {extraction_dir}")

            try:
                # Initialize and apply the various preprocessing steps
                detector = PreProcessing(file_path)
                detector.creation_json_grace_c3d()
                detector.extract_data_interval()
                detector.normalize_data()
                detector.decoupage_en_fenetres()
                detector.label_fenetre()
                detector.association_label_fenetre_data()
                data = detector.concat_label_fenetre_data()

                # Calculate statistics
                stats_processor = Statistics(file_name, data)
                statistiques_resultat = stats_processor.stats()
                concat_stats_df = pd.concat([concat_stats_df, statistiques_resultat], ignore_index=True)

                # Extract features
                extraction_features = ExtractionFeatures(data)
                extraction_features.enlever_derniere_ligne_et_colonne_label()
                data_features = extraction_features.dataframe_caracteristiques_final()

                # Save the extracted features to a CSV file
                features_output_file = os.path.join(extraction_dir, f"{os.path.splitext(file_name)[0]}_extraction_features.csv")
                data_features.to_csv(features_output_file, index=False)
                print(f"Features CSV saved to {features_output_file}")
            except Exception as e:
                print(f"An error occurred while processing file {file_name}: {e}")

    # Save the final statistics to a CSV file
    os.makedirs(statistics_directory, exist_ok=True)
    final_stats_file = os.path.join(statistics_directory, f'{identifiant}_statistics.csv')
    concat_stats_df.to_csv(final_stats_file, index=False)
    print(f"Final statistics have been saved to {final_stats_file}")
    print("All files have been processed.")

    return data_features, concat_stats_df

def combine_csv_files(source_directory, output_directory, category, patient_id):
    """
    Combines multiple CSV files from a source directory into a single output CSV file.
    All OFF files are combined, and all ON files are combined.

    Args:
        source_directory (str): Directory containing the CSV files to combine.
        output_directory (str): Directory where the combined CSV file will be saved.
        category (str): Category of files (e.g., 'ON' or 'OFF').
        patient_id (str): Patient identifier.
    """
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame for combined data
    extraction_features_dir = os.path.join(source_directory, 'extraction_features_by_file')
    print(f"Combining {category} CSV files from: {extraction_features_dir}")

    # Read and combine all CSV files ending with '_extraction_features.csv' in the source directory
    for file_name in os.listdir(extraction_features_dir):
        if file_name.endswith('_extraction_features.csv'):
            file_path = os.path.join(extraction_features_dir, file_name)
            print(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined DataFrame to a CSV file
    print(f"All {category} CSV files have been combined. Saving to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    combined_file_path = os.path.join(output_directory, f'{patient_id}_all_extraction_{category}.csv')
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Combined {category} CSV saved to {combined_file_path}")

def combine_on_off_files(on_output_directory, off_output_directory, combined_directory, patient_id):
    """
    Combines 'ON' and 'OFF' CSV files into a single combined CSV file.

    Args:
        on_output_directory (str): Directory containing 'ON' CSV files.
        off_output_directory (str): Directory containing 'OFF' CSV files.
        combined_directory (str): Directory where the combined CSV file will be saved.
        patient_id (str): Patient identifier.
    """
    on_file_path = os.path.join(on_output_directory, f'{patient_id}_all_extraction_ON.csv')
    off_file_path = os.path.join(off_output_directory, f'{patient_id}_all_extraction_OFF.csv')
    print(f"Starting to combine ON and OFF CSV files:\nON: {on_file_path}\nOFF: {off_file_path}")

    # Read 'ON' and 'OFF' CSV files
    on_df = pd.read_csv(on_file_path)
    off_df = pd.read_csv(off_file_path)

    # Combine the 'ON' and 'OFF' DataFrames
    combined_df = pd.concat([on_df, off_df], ignore_index=True)
    os.makedirs(combined_directory, exist_ok=True)
    combined_file_path = os.path.join(combined_directory, f'{patient_id}_all_extraction_ON_OFF.csv')
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Finished combining ON and OFF CSV saved to {combined_file_path}\n")

def process_patient(root_directory, statistics_directory, top_n_values, methods):
    """
    Processes a given patient, performing all preprocessing, feature extraction, file combination, and model evaluation steps.

    Args:
        root_directory (str): Root directory containing the patient's data.
        statistics_directory (str): Directory where statistics will be stored.
        top_n_values (list): List of top N values for main features to consider.
        methods (list): List of feature extraction methods to use.
    """
    patient_id = os.path.basename(root_directory)
    print(f"Processing patient with ID: {patient_id}")

    # Step 1: Batch processing and storing C3D files
    print("Starting batch processing and storing C3D files...")
    data_features, concat_stats_df = batch_process_and_store_c3d_files(root_directory, statistics_directory)
    print("Finished batch processing and storing C3D files.\n")

    # Step 2: Combining 'ON' CSV files
    on_source_directory = os.path.join(root_directory, '2_ON')
    on_output_directory = os.path.join(root_directory, '2_ON', 'all_extraction_ON')
    print("Starting to combine ON CSV files...")
    combine_csv_files(on_source_directory, on_output_directory, 'ON', patient_id)
    print("Finished combining ON CSV files.\n")

    # Step 3: Combining 'OFF' CSV files
    off_source_directory = os.path.join(root_directory, '1_OFF')
    off_output_directory = os.path.join(root_directory, '1_OFF', 'all_extraction_OFF')
    print("Starting to combine OFF CSV files...")
    combine_csv_files(off_source_directory, off_output_directory, 'OFF', patient_id)
    print("Finished combining OFF CSV files.\n")

    # Step 4 : Combining 'ON' and 'OFF' CSV files
    combined_directory = os.path.join(root_directory, '3_ON_OFF')
    combine_on_off_files(on_output_directory, off_output_directory, combined_directory, patient_id)

    # Step 5: Application of different rebalancing methods
    combined_file_path = os.path.join(combined_directory, f'{patient_id}_all_extraction_ON_OFF.csv')
    for method in methods:
        processor = FileProcessor(combined_file_path)
        if method == 'over':
            processor.process_file_over(root_directory, os.path.join(root_directory, 'best_combinations_over.csv'))
        elif method == 'optimise':
            processor.process_file_optimise(root_directory, os.path.join(root_directory, 'best_combinations_optimise.csv'))
        elif method == 'raw':
            processor.save_raw_data_splits(root_directory)

    # Step 6 : Ranking features
    train_folder = os.path.join(root_directory, '4_train_ON_OFF')
    output_folder_top_features = os.path.join(root_directory, '6_feature_ranking_ON_OFF')
    feature_ranking_processor = FeatureRankingProcessor(train_folder, output_folder_top_features)
    for method in methods:
        feature_ranking_processor.apply_relief(data_type=method)

    # Step 7: Training and evaluating machine learning models
    test_folder = os.path.join(root_directory, '5_test_ON_OFF')
    output_folder = os.path.join(root_directory, '7_results_machine_learning')
    os.makedirs(output_folder, exist_ok=True)
    model_trainer = ModelTraining(train_folder, test_folder, output_folder_top_features, output_folder)
    for method in methods:
        train_data = model_trainer.load_train(method)
        test_data = model_trainer.load_test(method)
        grouped_data = model_trainer.group_data([train_data, test_data])
        feature_importances = model_trainer.load_feature_importances(method)
        results = model_trainer.train_models(grouped_data, feature_importances, top_n_values, method)

def process_patients(patients_directories, statistics_directory, top_n_values, methods):
    """
    Processes a list of patients using the specified patient directories, feature extraction methods, and top_n values.

    Args:
        patients_directories (list): List of directories for the patients to process.
        statistics_directory (str): Directory where statistics will be stored.
        top_n_values (list): List of top N values for main features to consider.
        methods (list): List of feature extraction methods to use.
    """
    for patient_directory in patients_directories:
        process_patient(patient_directory, statistics_directory, top_n_values, methods)

if __name__ == "__main__":
    # Define directories for new patients to process
    new_patients_directories = ['C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01',
                                'C:/Users/antho/Documents/MEMOIRE_M2/A_P_1956_02_21']
    
    # Define the resample methods to use, include the select_features
    methods = ['over','optimise']

    # methods = ['raw', 'over', 'optimise']

    # Define top_n values from Relief F to include in Machine Learning models.
    top_n_values = [10, 20]

    # Define directory where statistics will be stored
    statistics_directory = 'C:/Users/antho/Documents/MEMOIRE_M2/statistiques_audeline/'

    # Process specified patients
    process_patients(new_patients_directories, statistics_directory, top_n_values, methods)
