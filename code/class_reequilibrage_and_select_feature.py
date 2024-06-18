import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import time
from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

class DataLoader:
    """
    Classe responsable du chargement et de la préparation des données à partir d'un fichier CSV.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_prepare_data(self):
        """
        Charge les données à partir du fichier CSV, filtre les colonnes avec des valeurs manquantes,
        supprime les lignes avec la valeur 'transitionNoFog' dans la colonne 'label', et prépare
        les données pour le modèle.
        
        Retourne :
        - X : Features.
        - y : Labels binarisés (1 pour 'fog' et 'transitionFog', 0 pour les autres).
        """
        print(f"Chargement et préparation des données pour {self.filepath}")
        data = pd.read_csv(self.filepath)
        data_filtered = data.dropna(axis=1)
        data_filtered = data_filtered[data_filtered['label'] != 'transitionNoFog'] # Suppression du label transitionNoFog
        X = data_filtered.drop('label', axis=1) # On enlève la colonne label
        y = data_filtered['label'].apply(lambda x: 1 if x in ['fog', 'transitionFog'] else 0) # On binarise les labels
        print(f"Données chargées et préparées pour {self.filepath}")
        return X, y

class ResamplingPipeline:
    """
    Classe responsable de la configuration des pipelines de suréchantillonnage et de resampling.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.smote_strategy_over = 1
        self.smote_strategy_optimise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.under_strategy_optimise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.X_train, self.X_test, self.y_train, self.y_test = self.standardize_and_split()

    def standardize_and_split(self):
        """
        Standardise les données et divise le jeu de données en ensembles d'entraînement et de test.
        
        Retourne :
        - X_train : Features d'entraînement.
        - X_test : Features de test.
        - y_train : Labels d'entraînement.
        - y_test : Labels de test.
        """
        print("Standardisation et division des données")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns) # Standardisation des features avec conservation des noms de colonnes
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, test_size=0.3, shuffle=True, stratify=self.y, random_state=42) # Division des données en ensembles d'entraînement et de test
        return X_train, X_test, y_train, y_test

    def configure_pipeline_over(self):
        """
        Configure un pipeline avec SMOTE pour suréchantillonner les données minoritaires.
        
        Retourne :
        - pipeline : Pipeline de suréchantillonnage avec SMOTE.
        """
        print("Configuration du pipeline de suréchantillonnage")
        steps = [('smote', SMOTE(sampling_strategy=self.smote_strategy_over, random_state=42)), 
                 ('model', DecisionTreeClassifier(random_state=42))]
        return ImPipeline(steps)

    def configure_pipeline_optimise(self):
        """
        Configure un pipeline avec SMOTE et RandomUnderSampler pour optimiser le resampling.
        
        Retourne :
        - pipeline : Pipeline optimisé avec SMOTE et RandomUnderSampler.
        """
        print(f"Configuration du pipeline avec SMOTE={self.smote_strategy_optimise} et UNDER={self.under_strategy_optimise}")
        steps = [('smote', SMOTE(sampling_strategy=self.smote_strategy_optimise, random_state=42))]
        steps.append(('under_sampler', RandomUnderSampler(sampling_strategy=self.under_strategy_optimise, random_state=42)))
        steps.append(('model', DecisionTreeClassifier(random_state=42)))
        return ImPipeline(steps)
    
    def configure_pipeline_with_best_strategies(self, smote_strategy, under_strategy):
        """
        Configure un pipeline avec les meilleures stratégies trouvées pour SMOTE et RandomUnderSampler.
        
        Paramètres :
        - smote_strategy : Stratégie SMOTE optimale.
        - under_strategy : Stratégie RandomUnderSampler optimale.
        
        Retourne :
        - pipeline : Pipeline configuré avec les meilleures stratégies.
        """
        print(f"Configuration du pipeline avec SMOTE={smote_strategy} et UNDER={under_strategy}")
        steps = [('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
                 ('under_sampler', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
                 ('model', DecisionTreeClassifier(random_state=42))]
        return ImPipeline(steps)

class ModelEvaluator:
    """
    Classe responsable de l'évaluation des modèles.
    """
    def __init__(self, pipeline, X_train, y_train):
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train

    def evaluate_model(self):
        """
        Évalue le modèle en utilisant une validation croisée stratifiée et retourne le score moyen ROC AUC.
        
        Retourne :
        - mean_score : Score moyen ROC AUC.
        """
        print("Évaluation du modèle")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, self.X_train, self.y_train, scoring='roc_auc', cv=cv, n_jobs=1)
        mean_score = np.mean(scores)
        print(f"Score moyen ROC AUC: {mean_score}")
        return mean_score

class DataSaver:
    """
    Classe responsable de la sauvegarde des données traitées.
    """
    def __init__(self, base_filename, directory_path):
        self.base_filename = base_filename
        self.directory_path = directory_path

    def save_data(self, X_train, X_test, y_train, y_test, suffix):
        """
        Sauvegarde les données d'entraînement et de test dans des fichiers CSV.
        
        Paramètres :
        - X_train : Features d'entraînement.
        - X_test : Features de test.
        - y_train : Labels d'entraînement.
        - y_test : Labels de test.
        - suffix : Suffixe à ajouter aux noms de fichiers.
        """
        print(f"Début de la sauvegarde pour {self.base_filename} avec suffixe {suffix}")
        train_path = f"{self.directory_path}/train_ON_OFF/X_train_{self.base_filename}_{suffix}.csv"
        train_label_path = f"{self.directory_path}/train_ON_OFF/y_train_{self.base_filename}_{suffix}.csv"
        test_path = f"{self.directory_path}/test_ON_OFF/X_test_{self.base_filename}_{suffix}.csv"
        test_label_path = f"{self.directory_path}/test_ON_OFF/y_test_{self.base_filename}_{suffix}.csv"
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        pd.DataFrame(X_train).to_csv(train_path, index=False, header=True)
        print(f"Données d'entraînement sauvegardées : {train_path}")
        pd.Series(y_train).to_csv(train_label_path, index=False, header=True)
        print(f"Étiquettes d'entraînement sauvegardées : {train_label_path}")
        pd.DataFrame(X_test).to_csv(test_path, index=False, header=True)
        print(f"Données de test sauvegardées : {test_path}")
        pd.Series(y_test).to_csv(test_label_path, index=False, header=True)
        print(f"Étiquettes de test sauvegardées : {test_label_path}")
        print(f"Fin de la sauvegarde pour {self.base_filename} avec suffixe {suffix}\n")

class FileProcessor:
    """
    Classe responsable du traitement des fichiers individuels.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.base_filename = os.path.basename(filepath).replace('.csv', '')
        self.loader = DataLoader(filepath)

    def process_file_over(self, save_directory, save_best_combinations_filepath):
        """
        Traite un fichier en utilisant la stratégie de suréchantillonnage SMOTE.
        
        Paramètres :
        - save_directory : Répertoire où sauvegarder les données traitées.
        
        Retourne :
        - result_over : Dictionnaire contenant les résultats du traitement.
        """
        X, y = self.loader.load_and_prepare_data()
        resampler = ResamplingPipeline(X, y)
        pipeline = resampler.configure_pipeline_over()
        evaluator = ModelEvaluator(pipeline, resampler.X_train, resampler.y_train)
        mean_score = evaluator.evaluate_model()
        pipeline.fit(resampler.X_train, resampler.y_train)
        X_resampled, y_resampled = pipeline.steps[0][1].fit_resample(resampler.X_train, resampler.y_train) # Suréchantillonnage des données
        saver = DataSaver(self.base_filename, save_directory)
        saver.save_data(X_resampled, resampler.X_test, y_resampled, resampler.y_test, "over100")
        result_over = {
            'File': self.filepath,
            'SMOTE Strategy': 1,
            'ROC AUC Score': mean_score,
            'Note': 'Resampling applied' if 1 != 'None' else 'No resampling due to class 1 >= class 0'
        }
        
        # Sauvegarder les résultats de la VC avec SMOTE  = 1 dans un fichier CSV
        results_df_over = pd.DataFrame([result_over])
        if not os.path.isfile(save_best_combinations_filepath):
            results_df_over.to_csv(save_best_combinations_filepath, index=False)
        else:
            results_df_over.to_csv(save_best_combinations_filepath, index=False)
        
        return result_over

    def process_file_optimise(self, save_directory, save_best_combinations_filepath):
        """
        Traite un fichier en optimisant les stratégies SMOTE et RandomUnderSampler.
        
        Paramètres :
        - save_directory : Répertoire où sauvegarder les données traitées.
        
        Retourne :
        - result_optimise : Dictionnaire contenant les résultats du traitement.
        """
        X, y = self.loader.load_and_prepare_data()
        resampler = ResamplingPipeline(X, y)
        best_score = 0
        best_smote = None
        best_under = None
        for smote_strategy in resampler.smote_strategy_optimise: # Pour chaque stratégie SMOTE
            for under_strategy in resampler.under_strategy_optimise: # Pour chaque stratégie RandomUnderSampler
                try:
                    pipeline = resampler.configure_pipeline_with_best_strategies(smote_strategy, under_strategy) # Configuration du pipeline avec les stratégies actuelles
                    evaluator = ModelEvaluator(pipeline, resampler.X_train, resampler.y_train)
                    score = evaluator.evaluate_model()
                    
                    if score > best_score: 
                        best_score = score
                        best_smote = smote_strategy
                        best_under = under_strategy
                        
                except Exception as e:
                    error_message = "Ratio impossible" if "The specified ratio" in str(e) else str(e)
                    print(f"Erreur lors du traitement : {error_message}")
                        
        if best_smote and best_under:
            pipeline = resampler.configure_pipeline_with_best_strategies(best_smote, best_under)
            X_resampled, y_resampled = pipeline.steps[0][1].fit_resample(resampler.X_train, resampler.y_train) # Resampling des données avec les meilleures stratégies
            resampler.X_train, resampler.y_train = X_resampled, y_resampled
        saver = DataSaver(self.base_filename, save_directory)
        saver.save_data(resampler.X_train, resampler.X_test, resampler.y_train, resampler.y_test, "optimise")
        result_optimise = {
            'File': self.filepath,
            'SMOTE Strategy': best_smote,
            'Under Strategy': best_under,
            'ROC AUC Score': best_score,
            'Note': 'Resampling applied'
        }
        # Sauvegarder les résultats des meilleures combinaisons dans un fichier CSV
        results_df_optimise = pd.DataFrame([result_optimise])
        if not os.path.isfile(save_best_combinations_filepath):
            results_df_optimise.to_csv(save_best_combinations_filepath, index=False)
        else:
            results_df_optimise.to_csv(save_best_combinations_filepath, index=False)
        return result_optimise

    def save_raw_data_splits(self, save_directory):
        """
        Sauvegarde les données brutes après standardisation et division en ensembles d'entraînement et de test.
        
        Paramètres :
        - save_directory : Répertoire où sauvegarder les données brutes.
        """
        X, y = self.loader.load_and_prepare_data()
        resampler = ResamplingPipeline(X, y)
        saver = DataSaver(self.base_filename, save_directory)
        saver.save_data(resampler.X_train, resampler.X_test, resampler.y_train, resampler.y_test, "brut")

class DirectoryProcessor:
    """
    Classe responsable du traitement de tous les fichiers dans un répertoire.
    """
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def process_directory_over(self, save_directory, results_filepath):
        """
        Traite tous les fichiers dans le répertoire en utilisant la stratégie de suréchantillonnage SMOTE.
        
        Paramètres :
        - save_directory : Répertoire où sauvegarder les données traitées.
        - results_filepath : Chemin du fichier CSV pour sauvegarder les résultats.
        """
        results_df_over = pd.DataFrame()
        files = [f for f in os.listdir(self.directory_path) if f.endswith(".csv")]
        for filename in tqdm(files, desc="Traitement des fichiers (over)"):
            filepath = os.path.join(self.directory_path, filename)
            processor = FileProcessor(filepath) 
            result_over = processor.process_file_over(save_directory) # Traitement du fichier avec suréchantillonnage
            results_df_over = pd.concat([results_df_over, pd.DataFrame([result_over])], ignore_index=True) # Ajout des résultats de chaque fichier au DataFrame
            print(f"Ajout des résultats pour {filename} au DataFrame.")
        results_df_over.to_csv(results_filepath, index=False)
        print(f"Résultats sauvegardés dans {results_filepath}.")

    def process_directory_optimise(self, save_directory, results_filepath):
        """
        Traite tous les fichiers dans le répertoire en optimisant les stratégies SMOTE et RandomUnderSampler.
        
        Paramètres :
        - save_directory : Répertoire où sauvegarder les données traitées.
        - results_filepath : Chemin du fichier CSV pour sauvegarder les résultats.
        """
        results_df_optimise = pd.DataFrame()
        files = [f for f in os.listdir(self.directory_path) if f.endswith(".csv")]
        for filename in tqdm(files, desc="Traitement des fichiers (optimise)"):
            filepath = os.path.join(self.directory_path, filename)
            processor = FileProcessor(filepath)
            result_optimise = processor.process_file_optimise(save_directory)
            results_df_optimise = pd.concat([results_df_optimise, pd.DataFrame([result_optimise])], ignore_index=True) # Ajout des résultats de chaque fichier au DataFrame
            print(f"Ajout des résultats pour {filename} avec SMOTE {result_optimise['SMOTE Strategy']} et UNDER {result_optimise['Under Strategy']} au DataFrame.") 
        results_df_optimise.to_csv(results_filepath, index=False)
        print(f"Résultats sauvegardés dans {results_filepath}.")

    def process_directory_raw(self, save_directory):
        """
        Sauvegarde les données brutes pour tous les fichiers dans le répertoire.
        
        Paramètres :
        - save_directory : Répertoire où sauvegarder les données brutes.
        """
        files = [f for f in os.listdir(self.directory_path) if f.endswith(".csv")] # Liste des fichiers CSV dans le répertoire
        for filename in tqdm(files, desc="Traitement des fichiers (raw)"): # Pour chaque fichier
            filepath = os.path.join(self.directory_path, filename)
            processor = FileProcessor(filepath)
            processor.save_raw_data_splits(save_directory)



######## Traitement par répertoire ########
# Début du chronométrage pour le suréchantillonnage
start_time_over = time.time()

# Chemin vers le répertoire des données
#directory_path = 'C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/all_features_by_patient/'

# Processus de suréchantillonnage pour chaque fichier dans le répertoire
# processor_over = DirectoryProcessor(directory_path)
# processor_over.process_directory_over("C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/all_features_by_patient_final_over100_test", # chemin de sauvegarde des données suréchantillonnées
#                                       'C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/resultats_resampling_by_patient/best_combinations_over100_test.csv')

# # Fin du chronométrage pour le suréchantillonnage
# end_time_over = time.time()
# total_time_over = end_time_over - start_time_over
# print(f"Temps d'exécution total pour le suréchantillonnage: {total_time_over:.2f} seconds")

# Début du chronométrage pour le resampling optimisé
# start_time_optimise = time.time()

# # Processus de resampling optimisé pour chaque fichier dans le répertoire
# processor_optimise = DirectoryProcessor(directory_path)
# processor_optimise.process_directory_optimise("C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/all_features_by_patient_final_optimise_test", # chemin de sauvegarde des données optimisées
#                                               'C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/resultats_resampling_by_patient/best_combinations_optimize_test.csv')

# # Fin du chronométrage pour le resampling optimisé
# end_time_optimise = time.time()
# total_time_optimise = end_time_optimise - start_time_optimise
# print(f"Temps d'exécution total pour le resampling optimisé: {total_time_optimise:.2f} seconds")

# Sauvegarde des données brutes pour chaque fichier dans le répertoire
# processor_raw = DirectoryProcessor(directory_path)
# processor_raw.process_directory_raw("D:/detectFog/data/all_features_by_patient_final_data_brute")




######## Traitement par fichier ########
# filepath = "C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/all_features_by_patient/A_P_1956-02-21_all_extraction_features.csv"
# # Processus de resampling optimisé pour chaque fichier dans le répertoire
# processor_optimise = FileProcessor(filepath)
# processor_optimise.process_file_optimise("C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/all_features_by_patient_final_optimise_test", # chemin de sauvegarde des données optimisées
#                                          'C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/resultats_resampling_by_patient/best_combinations_optimize_test.csv')



# Example usage:
root_directory = 'C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01'
patient_id = os.path.basename(root_directory)
combined_file_path = os.path.join(root_directory, 'ON_OFF', f'{patient_id}_all_extraction_ON_OFF.csv')

# Process with over-sampling
processor_over = FileProcessor(combined_file_path)
processor_over.process_file_over(root_directory, os.path.join(root_directory, 'best_combinations_over100.csv'))

# Process with optimized sampling
processor_optimise = FileProcessor(combined_file_path)
processor_optimise.process_file_optimise(root_directory, os.path.join(root_directory, 'best_combinations_optimize.csv'))

# Save raw data splits
processor_raw = FileProcessor(combined_file_path)
processor_raw.save_raw_data_splits(root_directory)


end_time_over = time.time()
total_time_over = end_time_over - start_time_over
print(f"Temps d'exécution total : {total_time_over:.2f} seconds")