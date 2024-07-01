import os
import pandas as pd
from preprocessing import PreProcessing, Statistics, ExtractionFeatures
from class_reequilibrage_and_select_feature import DataLoader, ResamplingPipeline, ModelEvaluator, DataSaver, FileProcessor, DirectoryProcessor, FeatureRankingProcessor
from machine_learning import ModelTraining

def batch_process_and_store_c3d_files(root_directory, statistics_directory):
    """
    Traite par lot et stocke les fichiers C3D dans un répertoire donné.

    Args:
        root_directory (str): Répertoire racine contenant les fichiers C3D à traiter.
        statistics_directory (str): Répertoire où les statistiques seront stockées.

    Returns:
        tuple: DataFrame des caractéristiques extraites et DataFrame des statistiques concaténées.
    """
    concat_stats_df = pd.DataFrame()

    # Parcourt tous les fichiers et répertoires sous root_directory
    for dirpath, _, filenames in os.walk(root_directory):
        c3d_files = [f for f in filenames if f.endswith('.c3d')]
        if not c3d_files:
            continue

        # Traite chaque fichier C3D trouvé
        for file_index, file_name in enumerate(c3d_files):
            file_path = os.path.join(dirpath, file_name)
            print(f"Processing file {file_index + 1}/{len(c3d_files)}: {file_name}")

            # Détermine le sous-répertoire de destination en fonction du nom du fichier
            subdirectory = '2_ON' if '_ON_' in file_name else '1_OFF'
            extraction_dir = os.path.join(root_directory, subdirectory, 'extraction_features_by_file')
            os.makedirs(extraction_dir, exist_ok=True)
            print(f"Created/verified subdirectory: {extraction_dir}")

            try:
                # Initialise et applique les différentes étapes de prétraitement
                detector = PreProcessing(file_path)
                detector.creation_json_grace_c3d()
                detector.extract_data_interval()
                detector.normalize_data()
                detector.decoupage_en_fenetres()
                detector.label_fenetre()
                detector.association_label_fenetre_data()
                data = detector.concat_label_fenetre_data()

                # Calcul des statistiques
                stats_processor = Statistics(file_path, data)
                statistiques_resultat = stats_processor.stats()
                concat_stats_df = pd.concat([concat_stats_df, statistiques_resultat], ignore_index=True)

                # Extraction des caractéristiques
                extraction_features = ExtractionFeatures(data)
                extraction_features.enlever_derniere_ligne_et_colonne_label()
                data_features = extraction_features.dataframe_caracteristiques_final()

                # Sauvegarde des caractéristiques extraites dans un fichier CSV
                features_output_file = os.path.join(extraction_dir, f"{os.path.splitext(file_name)[0]}_extraction_features.csv")
                data_features.to_csv(features_output_file, index=False)
                print(f"Features CSV saved to {features_output_file}")
            except Exception as e:
                print(f"An error occurred while processing file {file_name}: {e}")

    # Sauvegarde des statistiques finales dans un fichier CSV
    os.makedirs(statistics_directory, exist_ok=True)
    final_stats_file = os.path.join(statistics_directory, 'final_statistics_dernier_fichier.csv')
    concat_stats_df.to_csv(final_stats_file, index=False)
    print(f"Final statistics have been saved to {final_stats_file}")
    print("All files have been processed.\n")

    return data_features, concat_stats_df

def combine_csv_files(source_directory, output_directory, category, patient_id):
    """
    Combine plusieurs fichiers CSV d'un répertoire source en un seul fichier CSV de sortie.

    Args:
        source_directory (str): Répertoire contenant les fichiers CSV à combiner.
        output_directory (str): Répertoire où le fichier CSV combiné sera sauvegardé.
        category (str): Catégorie des fichiers (par exemple, 'ON' ou 'OFF').
        patient_id (str): Identifiant du patient.
    """
    combined_df = pd.DataFrame()
    extraction_features_dir = os.path.join(source_directory, 'extraction_features_by_file')
    print(f"Combining {category} CSV files from: {extraction_features_dir}")

    # Parcourt tous les fichiers CSV dans le répertoire des caractéristiques extraites
    for file_name in os.listdir(extraction_features_dir):
        if file_name.endswith('_extraction_features.csv'):
            file_path = os.path.join(extraction_features_dir, file_name)
            print(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Sauvegarde le DataFrame combiné dans un fichier CSV
    print(f"All {category} CSV files have been combined. Saving to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    combined_file_path = os.path.join(output_directory, f'{patient_id}_all_extraction_{category}.csv')
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Combined {category} CSV saved to {combined_file_path}")

def combine_on_off_files(on_output_directory, off_output_directory, combined_directory, patient_id):
    """
    Combine les fichiers CSV 'ON' et 'OFF' en un seul fichier CSV combiné.

    Args:
        on_output_directory (str): Répertoire contenant les fichiers CSV 'ON'.
        off_output_directory (str): Répertoire contenant les fichiers CSV 'OFF'.
        combined_directory (str): Répertoire où le fichier CSV combiné sera sauvegardé.
        patient_id (str): Identifiant du patient.
    """
    on_file_path = os.path.join(on_output_directory, f'{patient_id}_all_extraction_ON.csv')
    off_file_path = os.path.join(off_output_directory, f'{patient_id}_all_extraction_OFF.csv')
    print(f"Starting to combine ON and OFF CSV files:\nON: {on_file_path}\nOFF: {off_file_path}")

    # Lit les fichiers CSV 'ON' et 'OFF'
    on_df = pd.read_csv(on_file_path)
    off_df = pd.read_csv(off_file_path)

    # Combine les deux DataFrames
    combined_df = pd.concat([on_df, off_df], ignore_index=True)
    os.makedirs(combined_directory, exist_ok=True)
    combined_file_path = os.path.join(combined_directory, f'{patient_id}_all_extraction_ON_OFF.csv')
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Finished combining ON and OFF CSV saved to {combined_file_path}\n")

def process_patient(root_directory, statistics_directory, top_n_values, methods):
    """
    Traite un patient donné, en effectuant toutes les étapes de prétraitement,
    extraction de caractéristiques, combinaison de fichiers et évaluation de modèles.

    Args:
        root_directory (str): Répertoire racine contenant les données du patient.
        statistics_directory (str): Répertoire où les statistiques seront stockées.
        top_n_values (list): Liste des valeurs pour les caractéristiques principales à considérer.
        methods (list): Liste des méthodes d'extraction de caractéristiques à utiliser.
    """
    patient_id = os.path.basename(root_directory)
    print(f"Processing patient with ID: {patient_id}")

    # Étape 1 : Traitement par lot et stockage des fichiers C3D
    print("Starting batch processing and storing C3D files...")
    data_features, concat_stats_df = batch_process_and_store_c3d_files(root_directory, statistics_directory)
    print("Finished batch processing and storing C3D files.")

    # Étape 2 : Combinaison des fichiers CSV 'ON'
    on_source_directory = os.path.join(root_directory, '2_ON')
    on_output_directory = os.path.join(root_directory, '2_ON', 'all_extraction_ON')
    print("Starting to combine ON CSV files...")
    combine_csv_files(on_source_directory, on_output_directory, 'ON', patient_id)
    print("Finished combining ON CSV files.\n")

    # Étape 3 : Combinaison des fichiers CSV 'OFF'
    off_source_directory = os.path.join(root_directory, '1_OFF')
    off_output_directory = os.path.join(root_directory, '1_OFF', 'all_extraction_OFF')
    print("Starting to combine OFF CSV files...")
    combine_csv_files(off_source_directory, off_output_directory, 'OFF', patient_id)
    print("Finished combining OFF CSV files.\n")

    # Étape 4 : Combinaison des fichiers CSV 'ON' et 'OFF'
    combined_directory = os.path.join(root_directory, '3_ON_OFF')
    combine_on_off_files(on_output_directory, off_output_directory, combined_directory, patient_id)

    # Étape 5 : Traitement et classement des caractéristiques
    combined_file_path = os.path.join(combined_directory, f'{patient_id}_all_extraction_ON_OFF.csv')
    processor_over = FileProcessor(combined_file_path)
    processor_over.process_file_over(root_directory, os.path.join(root_directory, 'best_combinations_over100.csv'))
    processor_optimise = FileProcessor(combined_file_path)
    processor_optimise.process_file_optimise(root_directory, os.path.join(root_directory, 'best_combinations_optimize.csv'))
    processor_raw = FileProcessor(combined_file_path)
    processor_raw.save_raw_data_splits(root_directory)

    # Étape 6 : Classement des caractéristiques
    train_folder = os.path.join(root_directory, '4_train_ON_OFF')
    output_folder = os.path.join(root_directory, '6_classement_features_ON_OFF')
    feature_ranking_processor = FeatureRankingProcessor(train_folder, output_folder)
    for method in methods:
        feature_ranking_processor.apply_relief(data_type=method)

    # Étape 7 : Entraînement et évaluation des modèles de machine learning
    test_folder = os.path.join(root_directory, '5_test_ON_OFF')
    output_folder = os.path.join(root_directory, '7_resultats_machine_learning')
    model_trainer = ModelTraining(train_folder, test_folder, output_folder, output_folder)
    for method in methods:
        train_data = model_trainer.load_train(method)
        test_data = model_trainer.load_test(method)
        grouped_data = model_trainer.group_data([train_data, test_data])
        feature_importances = model_trainer.load_feature_importances(method)
        print(f"Feature importances loaded: {feature_importances.keys()}")  # Ajout de l'impression de débogage
        results = model_trainer.train_models(grouped_data, feature_importances, top_n_values, method)

def process_patients(patients_directories, statistics_directory, top_n_values, methods):
    """
    Traite une liste de patients en utilisant les répertoires de patients spécifiés,
    les méthodes d'extraction de caractéristiques et les valeurs top_n.

    Args:
        patients_directories (list): Liste des répertoires des patients à traiter.
        statistics_directory (str): Répertoire où les statistiques seront stockées.
        top_n_values (list): Liste des valeurs pour les caractéristiques principales à considérer.
        methods (list): Liste des méthodes d'extraction de caractéristiques à utiliser.
    """
    for patient_directory in patients_directories:
        process_patient(patient_directory, statistics_directory, top_n_values, methods)

# def process_new_patient(patient_directory, statistics_directory, methods, top_n_values):
#     """
#     Traite un nouveau patient en utilisant le répertoire du patient, les méthodes d'extraction de caractéristiques et les valeurs top_n spécifiés.

#     Args:
#         patient_directory (str): Répertoire contenant les données du nouveau patient.
#         statistics_directory (str): Répertoire où les statistiques seront stockées.
#         methods (list): Liste des méthodes d'extraction de caractéristiques à utiliser.
#         top_n_values (list): Liste des valeurs pour les caractéristiques principales à considérer.
#     """
#     process_patient(patient_directory, statistics_directory, top_n_values, methods)

if __name__ == "__main__":
    # Définir les répertoires des nouveaux patients à traiter
    new_patients_directories = ['C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01']
    
    # Définir les méthodes d'extraction de caractéristiques à utiliser
    methods = ['brut']
    
    # Définir les valeurs top_n à utiliser pour les caractéristiques principales
    top_n_values = [10, 20, 30]
    
    # Définir le répertoire où les statistiques seront stockées
    statistics_directory = 'C:/Users/antho/Documents/MEMOIRE_M2/statistiques_audeline/'
    
    # Traiter les patients spécifiés
    process_patients(new_patients_directories, statistics_directory, top_n_values, methods)
