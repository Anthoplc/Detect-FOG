import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import time
from skrebate import ReliefF, MultiSURF

# Attribution des resampling
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data_filtered = data.dropna(axis=1)
    data_filtered = data_filtered[data_filtered['label'] != 'transitionNoFog']
    X = data_filtered.drop('label', axis=1)
    y = data_filtered['label'].apply(lambda x: 1 if x in ['fog', 'transitionFog'] else 0)
    return X, y



def evaluate_model(pipeline, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    return np.mean(scores)
    
    
    
##############################Application 100% de suréchantillonnage#####################


def configure_pipeline_over(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)
    
    steps = [('model', DecisionTreeClassifier(random_state=42))]
    if y_train.value_counts().get(1, 0) < y_train.value_counts().get(0, 0):
        steps.insert(0, ('smote', SMOTE(sampling_strategy=1, random_state=42)))
    return ImPipeline(steps), X_train, y_train

def apply_vc_over_resampling_pipeline(filepath):
    print(f"Starting processing for {filepath}")
    try:
        X, y = load_and_prepare_data(filepath)
        pipeline, X_train, y_train = configure_pipeline_over(X, y)
        mean_score = evaluate_model(pipeline, X_train, y_train)
        print(f"Processing completed for {filepath} with mean ROC AUC score of {mean_score}")

        return pd.DataFrame({
            'File': [filepath],
            'SMOTE Strategy': 1,
            'ROC AUC Score': [mean_score],
            'Note': ['Resampling applied' if ['SMOTE Strategy'] != 'None' else 'No resampling due to class 1 >= class 0']
        })
    except Exception as e:
        error_message = "Ratio impossible" if "The specified ratio" in str(e) else str(e)
        print(f"Error processing {filepath}: {error_message}")
        return pd.DataFrame({
            'File': [filepath],
            'Error': [error_message]
        })    
    
##########################VC########################################################
# Path to directory where CSV files are stored
directory_path = 'C:/Users/Bonan/Documents/detectFog/data/ON_OFF_all_features/'
'''
# DataFrame to store results
results_df_over = pd.DataFrame()

# Process each file with each combination of resampling strategies
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory_path, filename)
        result_over = apply_vc_over_resampling_pipeline(filepath)
        results_df_over = pd.concat([results_df_over, result_over], ignore_index=True)
        print(f"Added results for {filename} to the DataFrame.")

# Save results
results_file = 'C:/Users/Bonan/Documents/detectFog/data/resultats_resampling/best_combinations_over100.csv'
results_df_over.to_csv(results_file, index=False)
print(f"Results saved to {results_file}.")



# On applique maintenant les données resample à 100%
best_combinations_path = 'C:/Users/Bonan/Documents/detectFog/data/resultats_resampling/best_combinations_over100.csv'
best_combinations_df = pd.read_csv(best_combinations_path)

# Fonction pour appliquer le resampling et sauvegarder les résultats
def resampling_over_and_save(filepath, smote_strategy):
    print(f"Traitement du fichier : {filepath}")
    
    data = pd.read_csv(filepath)
    data_filtered = data.dropna(axis=1)
    data_filtered = data_filtered[data_filtered['label'] != 'transitionNoFog']
    X = data_filtered.drop('label', axis=1)
    y = data_filtered['label'].apply(lambda x: 1 if x in ['fog', 'transitionFog'] else 0)

    print("Préparation des données...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)

    if y_train.value_counts().get(1, 0) > y_train.value_counts().get(0, 0):
        print("Aucun resampling nécessaire.")
        print("Terminé.")

        # Stocker les datasets
        base_filename = os.path.basename(filepath).replace('.csv', '')
        directory_path = "C:/Users/Bonan/Documents/detectFog/data/ON_OFF_all_features_final"

        # Chemins pour enregistrer les fichiers
        train_path = f"{directory_path}_over100/train/X_train_{base_filename}_over100.csv"
        train_label_path = f"{directory_path}_over100/train/y_train_{base_filename}_over100.csv"
        test_path = f"{directory_path}_over100/test/X_test_{base_filename}_over100.csv"
        test_label_path = f"{directory_path}_over100/test/y_test_{base_filename}_over100.csv"

        # Enregistrement des données
        pd.DataFrame(X_train).to_csv(train_path, index=False, header=True)
        print(f"Données d'entraînement sauvegardées : {train_path}")
        pd.Series(y_train).to_csv(train_label_path, index=False, header=True)
        print(f"Étiquettes d'entraînement sauvegardées : {train_label_path}")
        pd.DataFrame(X_test).to_csv(test_path, index=False, header=True)
        print(f"Données de test sauvegardées : {test_path}")
        pd.Series(y_test).to_csv(test_label_path, index=False, header=True)
        print(f"Étiquettes de test sauvegardées : {test_label_path}")
        
    else:
        print("Application du resampling")
        smote = SMOTE(sampling_strategy=1, random_state=42)
    
        pipeline = Pipeline([
            ('smote', smote)
            ])

        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        print("Resampling terminé.")

        # Stocker les datasets
        base_filename = os.path.basename(filepath).replace('.csv', '')
        directory_path = "C:/Users/Bonan/Documents/detectFog/data/ON_OFF_all_features_final"

        # Chemins pour enregistrer les fichiers
        train_path = f"{directory_path}_over100/train/X_train_{base_filename}_over100.csv"
        train_label_path = f"{directory_path}_over100/train/y_train_{base_filename}_over100.csv"
        test_path = f"{directory_path}_over100/test/X_test_{base_filename}_over100.csv"
        test_label_path = f"{directory_path}_over100/test/y_test_{base_filename}_over100.csv"

        # Enregistrement des données
        pd.DataFrame(X_resampled).to_csv(train_path, index=False, header=True)
        print(f"Données d'entraînement sauvegardées : {train_path}")
        pd.Series(y_resampled).to_csv(train_label_path, index=False, header=True)
        print(f"Étiquettes d'entraînement sauvegardées : {train_label_path}")
        pd.DataFrame(X_test).to_csv(test_path, index=False, header=True)
        print(f"Données de test sauvegardées : {test_path}")
        pd.Series(y_test).to_csv(test_label_path, index=False, header=True)
        print(f"Étiquettes de test sauvegardées : {test_label_path}")



# Début du chronométrage
start_time_over = time.time()

for index, row in best_combinations_df.iterrows():
    file_path = row['File']
    smote_strategy = row['SMOTE Strategy']
    resampling_over_and_save(file_path, smote_strategy)

print(" ")
# Fin du chronométrage
end_time_over = time.time()

# Calcul et affichage du temps d'exécution total
total_time_over = end_time_over - start_time_over
print(f"Total execution time: {total_time_over:.2f} seconds")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
###############################APPLICTION REEQUILLIBRAGE OPTIMISE##############################
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print("Debut Resampling optimise")
print(" ")
print(" ")
print(" ")
print(" ")

def apply_resampling_pipeline(filepath, smote_strategy, under_strategy):
    print(f"Traitement du fichier {filepath}...")
    try:
        data = pd.read_csv(filepath)
        data_filtered = data.dropna(axis=1)
        data_filtered = data_filtered[data_filtered['label'] != 'transitionNoFog']
        X = data_filtered.drop('label', axis=1)
        y = data_filtered['label'].apply(lambda x: 1 if x in ['fog', 'transitionFog'] else 0)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)


        # Setup resampling and model
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
        under_sampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        steps = [('smote', smote), ('under_sampler', under_sampler), ('model', model)]
        pipeline = Pipeline(steps)

        # Evaluate pipeline
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
        mean_score = np.mean(scores)
        return {'File': filepath, 'SMOTE Strategy': smote_strategy, 'Under Strategy': under_strategy, 'ROC AUC Score': mean_score, 'Note': 'Resampling applied'}

    except Exception as e:
        error_message = "Ratio impossible" if "The specified ratio" in str(e) else str(e)
        print(f"Error processing {filepath}: {error_message}")
        return None





##########################VC########################################################


start_time_optimise = time.time()


# Path to directory
directory_path = 'C:/Users/Bonan/Documents/detectFog/data/ON_OFF_all_features/'

# Stratégies de resampling à tester
smote_strategies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
under_strategies = [0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7,0.8,0.9,1]


# DataFrame to store results
results_df = pd.DataFrame()

# Process each file with each combination of resampling strategies
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory_path, filename)
        for smote_strategy in smote_strategies:
            for under_strategy in under_strategies:
                result = apply_resampling_pipeline(filepath, smote_strategy, under_strategy)
                result_df = pd.DataFrame([result])
                results_df = pd.concat([results_df, result_df], ignore_index=True)
                print(f"Added results for {filename} with SMOTE {smote_strategy} and UNDER {under_strategy} to the DataFrame.")

# Filter out the rows where ROC AUC Score is 'N/A'
valid_results = results_df[results_df['ROC AUC Score'] != 'N/A']

# Find the best combination based on ROC AUC Score
if not valid_results.empty:
    best_combination = valid_results.loc[valid_results['ROC AUC Score'].idxmax()]
    print(f"The best resampling combination is SMOTE {best_combination['SMOTE Strategy']} and UNDER {best_combination['Under Strategy']} with an ROC AUC Score of {best_combination['ROC AUC Score']}")

# Save results
results_file = 'C:/Users/Bonan/Documents/detectFog/data/resultats_resampling/results_combinations_optimize.csv'
results_df.to_csv(results_file, index=False)
print(f"Results saved to {results_file}.")



# Grouper les résultats par fichier et trouver la combinaison avec le meilleur ROC AUC Score pour chaque fichier
best_combinations = results_df.loc[results_df.groupby('File')['ROC AUC Score'].idxmax()]

# Chemin du fichier où vous souhaitez sauvegarder les meilleures combinaisons
results_best_combinations_path = 'C:/Users/Bonan/Documents/detectFog/data/resultats_resampling/best_combinations_optimize.csv'

# Utiliser la méthode to_csv sur l'objet DataFrame pour enregistrer les données dans un fichier CSV
best_combinations.to_csv(results_best_combinations_path, index=False)

print("Les meilleures combinaisons ont été sauvegardées avec succès.")





###################### Maintenant on applique les paramètres optimisés#####################
# Chemin vers le DataFrame des meilleures combinaisons
best_combinations_path = 'C:/Users/Bonan/Documents/detectFog/data/resultats_resampling/best_combinations_optimize.csv'
best_combinations_df = pd.read_csv(best_combinations_path)




# DataFrame pour stocker les distributions des classes
class_distribution_df = pd.DataFrame(columns=['File', 'SMOTE Strategy', 'Under Strategy', 'Before Resampling', 'After Resampling'])


# Fonction pour appliquer le resampling et sauvegarder les résultats
def resampling_optimize_and_save(filepath, smote_strategy, under_strategy):
    print(f"Traitement du fichier : {filepath}")
    
    data = pd.read_csv(filepath)
    data_filtered = data.dropna(axis=1)
    data_filtered = data_filtered[data_filtered['label'] != 'transitionNoFog']
    X = data_filtered.drop('label', axis=1)
    y = data_filtered['label'].apply(lambda x: 1 if x in ['fog', 'transitionFog'] else 0)

    print("Préparation des données...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)



    # On récupère la distribtuion avant le resampling
    before_resampling = y_train.value_counts().to_dict()
    
    if smote_strategy == 'None' and under_strategy == 'None':
        print("Aucun resampling nécessaire.")
        
        
        after_resampling = before_resampling
        print("Terminé.")

        # Stocker les datasets
        base_filename = os.path.basename(filepath).replace('.csv', '')
        directory_path = "C:/Users/Bonan/Documents/detectFog/data/ON_OFF_all_features_final"

        # Chemins pour enregistrer les fichiers
        train_path = f"{directory_path}_optimise/train/X_train_{base_filename}_optimise.csv"
        train_label_path = f"{directory_path}_optimise/train/y_train_{base_filename}_optimise.csv"
        test_path = f"{directory_path}_optimise/test/X_test_{base_filename}_optimise.csv"
        test_label_path = f"{directory_path}_optimise/test/y_test_{base_filename}_optimise.csv"

        # Enregistrement des données
        pd.DataFrame(X_train).to_csv(train_path, index=False, header=True)
        print(f"Données d'entraînement sauvegardées : {train_path}")
        pd.Series(y_train).to_csv(train_label_path, index=False, header=True)
        print(f"Étiquettes d'entraînement sauvegardées : {train_label_path}")
        pd.DataFrame(X_test).to_csv(test_path, index=False, header=True)
        print(f"Données de test sauvegardées : {test_path}")
        pd.Series(y_test).to_csv(test_label_path, index=False, header=True)
        print(f"Étiquettes de test sauvegardées : {test_label_path}")
        
    else:
        print(f"Application du resampling: SMOTE={smote_strategy}, Under={under_strategy}")
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
        under_sampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
    
        pipeline = Pipeline([
            ('smote', smote),
            ('under_sampler', under_sampler)
            ])

        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        after_resampling = y_resampled.value_counts().to_dict()
        
        print("Resampling terminé.")
        # Stocker les datasets
        base_filename = os.path.basename(filepath).replace('.csv', '')
        directory_path = "C:/Users/Bonan/Documents/detectFog/data/ON_OFF_all_features_final"

        # Chemins pour enregistrer les fichiers
        train_path = f"{directory_path}_optimise/train/X_train_{base_filename}_optimise.csv"
        train_label_path = f"{directory_path}_optimise/train/y_train_{base_filename}_optimise.csv"
        test_path = f"{directory_path}_optimise/test/X_test_{base_filename}_optimise.csv"
        test_label_path = f"{directory_path}_optimise/test/y_test_{base_filename}_optimise.csv"

        # Enregistrement des données
        pd.DataFrame(X_resampled).to_csv(train_path, index=False, header=True)
        print(f"Données d'entraînement sauvegardées : {train_path}")
        pd.Series(y_resampled).to_csv(train_label_path, index=False, header=True)
        print(f"Étiquettes d'entraînement sauvegardées : {train_label_path}")
        pd.DataFrame(X_test).to_csv(test_path, index=False, header=True)
        print(f"Données de test sauvegardées : {test_path}")
        pd.Series(y_test).to_csv(test_label_path, index=False, header=True)
        print(f"Étiquettes de test sauvegardées : {test_label_path}")
    
    # Enregistrement des distributions dans le DataFrame
    class_distribution_df.loc[len(class_distribution_df)] = [filepath, smote_strategy, under_strategy, before_resampling, after_resampling]
        
    # Sauvegarde des résultats dans un fichier CSV
    class_distribution_df.to_csv('C:/Users/Bonan/Documents/detectFog/data/resultats_resampling/class_distribution_details.csv', index=False)    
        
        
        

# Appliquer le traitement pour chaque fichier
for index, row in best_combinations_df.iterrows():
    file_path = row['File']
    smote_strategy = row['SMOTE Strategy']
    under_strategy = row['Under Strategy']
    resampling_optimize_and_save(file_path, smote_strategy, under_strategy)


print(" ")
print(" ")
# Fin du chronométrage
end_time_optimise = time.time()

# Calcul et affichage du temps d'exécution total
total_time_optimise = end_time_optimise - start_time_optimise
print(f"Temps d'execution du resample optimise : {total_time_optimise:.2f} seconds")'''



########################################################################################
#########################################################################################
########################################################################################


##################################################### RELIEF F ####################################################


'''
print(" ")
print(" VC pour le relief f en optimisé")
print(" ")

start_time_relief_over_vc = time.time()

################ Validation croisée RELIEF F avec OVER ###########################################

# ON crée un dictionnaire dans lequel, on stocker pour chaque fichier X_train et y_train
def load_train(train_folder):
    data_dict = {}
    for file in os.listdir(train_folder):
        file_path = os.path.join(train_folder, file)
        # Extract a common identifier by stripping off 'X_train_' or 'y_train_' and everything after '_over'
        if file.startswith('X_train') or file.startswith('y_train'):
            identifier = file.split('_over')[0].replace('X_train_', '').replace('y_train_', '')
            print("Processing:", identifier)
            if identifier not in data_dict:
                data_dict[identifier] = {}
            if 'X_train' in file:
                data_dict[identifier]['X_train'] = pd.read_csv(file_path)
            elif 'y_train' in file:
                data_dict[identifier]['y_train'] = pd.read_csv(file_path).squeeze()  # Convert DataFrame to Series directly
    print("Data loaded for identifiers:", list(data_dict.keys()))
    return data_dict


# On appliquer une foncion de VC  pour avoir les paramètres ed Relief F OVER
def vc_relief(data_dict, n_neighbors_list, n_features_list):
    results = []
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    
    for identifier, data in data_dict.items():
        X = data['X_train']
        y = data['y_train']
        
        print(f"Running cross-validation for {identifier}")
        for n_neighbors in n_neighbors_list:
            for n_features_to_select in n_features_list:
                print(f"Processing {identifier} with n_neighbors={n_neighbors}, n_features_to_select={n_features_to_select}")

                relief = ReliefF(n_neighbors=n_neighbors)
                relief.fit(X.values, y.values)
                # Obtention des indices des caractéristiques sélectionnées
                top_features_indices = relief.top_features_[:n_features_to_select]

                # Sélection manuelle des caractéristiques
                X_transformed = X.iloc[:, top_features_indices]
                
                model = DecisionTreeClassifier(random_state=42)
                scores = cross_val_score(model, X_transformed, y, scoring='roc_auc', cv=cv, n_jobs=-1)
                mean_score = np.mean(scores)
                
                results.append({
                    'File': identifier,
                    'n_neighbors': n_neighbors,
                    'n_features': n_features_to_select,
                    'ROC AUC Score': mean_score
                })

    return pd.DataFrame(results)

# Paths and parameters
train_folder_over = 'D:/detectFog/data/ON_OFF_all_features_final_over100/train'
data_dict = load_train(train_folder_over)
#n_neighbors_list = [5, 10, 50, 100, 200, 300]
#n_features_list = [1, 5, 10, 20, 30, 50, 80, 100, 200, 500]

n_neighbors_list = [5, 10]
n_features_list = [5, 10]


# Run validation
results_df = vc_relief(data_dict, n_neighbors_list, n_features_list)
results_df.to_csv('D:/detectFog/data/resultats_relief_f/tous_resultats_vc_hyperparametres_relief_over.csv', index=False) 


best_result = results_df.loc[results_df.groupby('File')['ROC AUC Score'].idxmax()]
best_result.to_csv('D:/detectFog/data/resultats_relief_f/best_resultats_vc_hyperparametres_relief_over.csv', index=False) 
print('Le fichier avec les hyperparamètres de relief f over a bien été enregistré')  

stop_time_relief_over_vc = time.time()
total_time_relief_over_vc = stop_time_relief_over_vc - start_time_relief_over_vc
print(f"Temps d'execution du resample optimise : {total_time_relief_over_vc:.2f} seconds")

            
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")

print("Création des scores de Relief F over")
print(" ")
print(" ")
print(" ")
print(" ")



start_time_score_relief_over = time.time()

# On applique relief f, avec les paramètres optimaux
def Application_relief_over(best_result, train_folder, output_folder):
    data_dict = load_train(train_folder)

    # Parcourir les fichiers et les paramètres de best_result ensemble
    for (key, data), (_, params) in zip(data_dict.items(), best_result.iterrows()):
        X = data['X_train']
        y = data['y_train']

        # Récupération des paramètres directement de la ligne correspondante
        n_neighbor = params['n_neighbors']
        n_features_to_select = params['n_features']

        print(f"\nProcessing dataset: {key}")
        print(f"Applying ReliefF with {n_neighbor} neighbors and selecting {n_features_to_select} features...")
        
        relief = ReliefF(n_neighbors=n_neighbor)
        relief.fit(X.values, y.values)

        # Obtention des indices des caractéristiques sélectionnées
        top_features_indices = relief.top_features_[:n_features_to_select]

        # Obtenez les scores d'importance des fonctionnalités sélectionnées
        feature_importances = relief.feature_importances_
        feature_names = X.columns.tolist()

        # Filtrer pour obtenir seulement les scores des caractéristiques sélectionnées
        selected_features_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_features_indices],
                'Score': [feature_importances[i] for i in top_features_indices]
        })

        # Trier les fonctionnalités sélectionnées par score d'importance décroissant
        selected_features_importance = selected_features_importance.sort_values(by='Score', ascending=False)

        # Créer le dossier de sortie s'il n'existe pas déjà
        output_path = os.path.join(output_folder)
        os.makedirs(output_path, exist_ok=True)
            
        # Enregistrer les scores d'importance des fonctionnalités dans un fichier CSV
        selected_features_importance.to_csv(os.path.join(output_path,f'{key}_score_relief_over.csv'), index=False) 

# Utilisation de la fonction Application_relief_optimise avec le dossier de sortie spécifié
output_folder = 'D:/detectFog/data/score_importance_relief_condition'
Application_relief_over(best_result, train_folder_over, output_folder)



stop_time_score_relief_over = time.time()
total_time_score_relief_over = stop_time_score_relief_over - start_time_score_relief_over
print(f"Temps d'execution du resample optimise : {total_time_score_relief_over:.2f} seconds")

######################################
####################################
################################### Avec optimisé######################################################



start_time_vc_optimise = time.time()

# Paths and parameters
train_folder = 'D:/detectFog/data/ON_OFF_all_features_final_optimise/train'
data_dict = load_train(train_folder)


# Run validation
results_df = vc_relief(data_dict, n_neighbors_list, n_features_list)
results_df.to_csv('D:/detectFog/data/resultats_relief_f/tous_resultats_vc_hyperparametres_relief_optimise.csv', index=False) 


best_result = results_df.loc[results_df.groupby('File')['ROC AUC Score'].idxmax()]
best_result.to_csv('D:/detectFog/data/resultats_relief_f/best_resultats_vc_hyperparametres_relief_optimise.csv', index=False) 
print('Le fichier avec les hyperparamètres de relief f over a bien été enregistré')  


print('Le fichier avec les hyperparamètres de relief f optimisé a bien été enregistré')  

stop_time_vc_optimise = time.time()
total_time_score_vc_optimise = stop_time_vc_optimise - start_time_vc_optimise
print(f"Temps d'execution du resample optimise : {total_time_score_vc_optimise:.2f} seconds")
            
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")

print("Création des scores de Relief F optimisé")
print(" ")
print(" ")
print(" ")
print(" ")



start_time_score_relief_optimise = time.time()


# On applique relief f, avec les paramètres optimaux
def Application_relief_optimise(best_result, train_folder, output_folder):
    data_dict = load_train(train_folder)

    # Parcourir les fichiers et les paramètres de best_result ensemble
    for (key, data), (_, params) in zip(data_dict.items(), best_result.iterrows()):
        X = data['X_train']
        y = data['y_train']

        # Récupération des paramètres directement de la ligne correspondante
        n_neighbor = params['n_neighbors']
        n_features_to_select = params['n_features']

        print(f"\nProcessing dataset: {key}")
        print(f"Applying ReliefF with {n_neighbor} neighbors and selecting {n_features_to_select} features...")
        
        relief = ReliefF(n_neighbors=n_neighbor)
        relief.fit(X.values, y.values)

        # Obtention des indices des caractéristiques sélectionnées
        top_features_indices = relief.top_features_[:n_features_to_select]

        # Obtenez les scores d'importance des fonctionnalités sélectionnées
        feature_importances = relief.feature_importances_
        feature_names = X.columns.tolist()

        # Filtrer pour obtenir seulement les scores des caractéristiques sélectionnées
        selected_features_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_features_indices],
                'Score': [feature_importances[i] for i in top_features_indices]
        })

        # Trier les fonctionnalités sélectionnées par score d'importance décroissant
        selected_features_importance = selected_features_importance.sort_values(by='Score', ascending=False)


        # Créer le dossier de sortie s'il n'existe pas déjà
        output_path = os.path.join(output_folder)
        os.makedirs(output_path, exist_ok=True)
            
        # Enregistrer les scores d'importance des fonctionnalités dans un fichier CSV
        selected_features_importance.to_csv(os.path.join(output_path,f'{key}_score_relief_optimise.csv'), index=False) 

# Utilisation de la fonction Application_relief_optimise avec le dossier de sortie spécifié
output_folder = 'D:/detectFog/data/score_importance_relief_condition'
Application_relief_optimise(best_result, train_folder, output_folder)


stop_time_score_relief_optimise =time.time()
total_time_score_relief_optimise = stop_time_score_relief_optimise - start_time_score_relief_optimise
print(f"Temps d'execution du resample optimise : {total_time_score_relief_optimise:.2f} seconds")'''


print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' RELIEF AVEC DONNEES BRUTES')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')

###############################################################################################################
########################################### RELIEF F avec données brutes ######################################

start_time_relief_data_brute_vc = time.time()

def load_train_data_brute(train_folder):
    data_dict = {}
    for file in os.listdir(train_folder):
        file_path = os.path.join(train_folder, file)
        # Extract a common identifier by stripping off 'X_train_' or 'y_train_' and everything after '_over'
        if file.startswith('X_train') or file.startswith('y_train'):
            identifier = file.split('_brut')[0].replace('X_train_', '').replace('y_train_', '')
            print("Processing:", identifier)
            if identifier not in data_dict:
                data_dict[identifier] = {}
            if 'X_train' in file:
                print(f"Loading X_train for {identifier}")
                data_dict[identifier]['X_train'] = pd.read_csv(file_path)
            elif 'y_train' in file:
                print(f"Loading y_train for {identifier}")
                data_dict[identifier]['y_train'] = pd.read_csv(file_path).squeeze()  # Convert DataFrame to Series directly
    print("Data loaded for identifiers:", list(data_dict.keys()))
    return data_dict


#VC pour relief f avec données brutes
def vc_relief_data_brute(train_folder, n_features_list):
    data_dict = load_train_data_brute(train_folder)
    results = []
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    for identifier, data in data_dict.items():
        X = data['X_train']
        y = data['y_train']
        
        print(f"Running cross-validation for {identifier}")
        for n_features_to_select in n_features_list:
            print(f"Processing {identifier}, n_features_to_select={n_features_to_select}")

            relief = MultiSURF()
            relief.fit(X.values, y.values)
            # Obtention des indices des caractéristiques sélectionnées
            top_features_indices = relief.top_features_[:n_features_to_select]

            # Sélection manuelle des caractéristiques
            X_transformed = X.iloc[:, top_features_indices]
                
            model = DecisionTreeClassifier(random_state=42)
            scores = cross_val_score(model, X_transformed, y, scoring='roc_auc', cv=cv, n_jobs=-1)
            mean_score = np.mean(scores)
                
            results.append({
                'File': identifier,
                'n_features': n_features_to_select,
                'ROC AUC Score': mean_score
            })
                
        print(f"Completed processing {identifier}")

    return pd.DataFrame(results)









# def vc_relief_data_brute(directory_path, n_neighbors_list, n_features_list):
#     results = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".csv"):
#             filepath = os.path.join(directory_path, filename)
#             X, y = load_and_prepare_data(filepath)
#             scaler = StandardScaler()
#             X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
#             X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)

#             for n_neighbors in n_neighbors_list:
#                 for n_features_to_select in n_features_list:
#                     print(f"Processing {filename} with n_neighbors={n_neighbors}, n_features_to_select={n_features_to_select}")
#                     relief = ReliefF(n_neighbors=n_neighbors)
#                     relief.fit(X_train.values, y_train.values)

#                     top_features_indices = relief.top_features_[:n_features_to_select]
#                     X_transformed = X_scaled.iloc[:, top_features_indices]

#                     model = DecisionTreeClassifier(random_state=42)
#                     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
#                     scores = cross_val_score(model, X_transformed, y, scoring='roc_auc', cv=cv, n_jobs=-1)
#                     mean_roc_auc = np.mean(scores)

#                     results.append({
#                         'File': filename,
#                         'n_neighbors': n_neighbors,
#                         'n_features': n_features_to_select,
#                         'ROC AUC': mean_roc_auc
#                     })
#             print(f"Completed processing {filename}")

#     return pd.DataFrame(results)

directory_path = 'D:/detectFog/data/ON_OFF_all_features_final_data_brute/train/'
n_features_list = [5, 10, 20, 50, 100, 200, 500]
# Run validation
results_df_data_brute = vc_relief_data_brute(directory_path, n_features_list)
results_df_data_brute.to_csv('D:/detectFog/data/resultats_relief_f/tous_resultats_vc_hyperparametres_data_brute.csv', index=False) 


best_result_data_brute = results_df_data_brute.loc[results_df_data_brute.groupby('File')['ROC AUC'].idxmax()]
best_result_data_brute.to_csv('D:/detectFog/data/resultats_relief_f/best_resultats_vc_hyperparametres_relief_data_brute.csv', index=False) 
print('Le fichier avec les hyperparamètres de relief f des datas brutes a bien été enregistré')  

# stop_time_relief_data_brute_vc = time.time()
# total_time_relief_data_brute_vc = stop_time_relief_data_brute_vc - start_time_relief_data_brute_vc
# print(f"Temps d'execution du resample optimise : {total_time_relief_data_brute_vc:.2f} seconds")


############# On applique RELIEF F DATA BRUTE ##########################################################


start_time_score_relief_data_brute = time.time()


def Application_relief_data_brute(best_result, train_folder, output_folder):
    data_dict = load_train_data_brute(train_folder)

    # Parcourir les fichiers et les paramètres de best_result ensemble
    for (key, data), (_, params) in zip(data_dict.items(), best_result.iterrows()):
        X = data['X_train']
        y = data['y_train']

        # Récupération des paramètres directement de la ligne correspondante
        n_features_to_select = params['n_features']

        print(f"\nProcessing dataset: {key}")
        print(f"Applying ReliefF with {n_features_to_select} features...")
        
        relief = MultiSURF()
        relief.fit(X.values, y.values)

        # Obtention des indices des caractéristiques sélectionnées
        top_features_indices = relief.top_features_[:n_features_to_select]

        # Obtenez les scores d'importance des fonctionnalités sélectionnées
        feature_importances = relief.feature_importances_
        feature_names = X.columns.tolist()

        # Filtrer pour obtenir seulement les scores des caractéristiques sélectionnées
        selected_features_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_features_indices],
                'Score': [feature_importances[i] for i in top_features_indices]
        })

        # Trier les fonctionnalités sélectionnées par score d'importance décroissant
        selected_features_importance = selected_features_importance.sort_values(by='Score', ascending=False)


        # Créer le dossier de sortie s'il n'existe pas déjà
        output_path = os.path.join(output_folder)
        os.makedirs(output_path, exist_ok=True)
            
        # Enregistrer les scores d'importance des fonctionnalités dans un fichier CSV
        selected_features_importance.to_csv(os.path.join(output_path,f'{key}_score_relief_brut.csv'), index=False) 

# Utilisation de la fonction Application_relief_optimise avec le dossier de sortie spécifié
output_folder = 'D:/detectFog/data/score_importance_relief_condition'
best_result_data_brute = 'D:/detectFog/data/resultats_relief_f/best_resultats_vc_hyperparametres_relief_data_brute.csv'
train_folder = 'D:/detectFog/data/ON_OFF_all_features_final_data_brute/train/'
Application_relief_data_brute(best_result_data_brute, train_folder, output_folder)



























def Application_relief_data_brute(best_result, directory_path, output_folder):
    print("Starting application of ReliefF on raw data...")

    # Charger les données de chaque fichier et appliquer les paramètres correspondants de ReliefF
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    best_result = best_result.reset_index(drop=True)  # Réinitialiser l'index pour la synchronisation

    for index, filename in enumerate(files):
        filepath = os.path.join(directory_path, filename)
        X, y = load_and_prepare_data(filepath)

        if index < len(best_result):
            row = best_result.iloc[index]
            n_neighbor = row['n_neighbors']
            n_features_to_select = row['n_features']
            print(f"Applying ReliefF for file {filename} with n_neighbors={n_neighbor}, n_features={n_features_to_select}")

            relief = ReliefF(n_neighbors=n_neighbor)
            relief.fit(X.values, y.values)

            # Obtention des indices des caractéristiques sélectionnées
            top_features_indices = relief.top_features_[:n_features_to_select]

            # Obtenez les scores d'importance des fonctionnalités sélectionnées
            feature_importances = relief.feature_importances_
            feature_names = X.columns.tolist()

            # Filtrer pour obtenir seulement les scores des caractéristiques sélectionnées
            selected_features_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_features_indices],
                'Score': [feature_importances[i] for i in top_features_indices]
            })

            # Trier les fonctionnalités sélectionnées par score d'importance décroissant
            selected_features_importance = selected_features_importance.sort_values(by='Score', ascending=False)

            # Chemin du fichier pour enregistrer les résultats
            output_file_path = os.path.join(output_folder, f'{filename[:-4]}_score_relief_data_brute.csv')
            
            # Enregistrer les scores d'importance des fonctionnalités dans un fichier CSV
            selected_features_importance.to_csv(output_file_path, index=False)
            print(f"Saved importance scores to {output_file_path}")




'''def Application_relief_data_brute(best_result, directory_path, output_folder):
    print("Starting application of ReliefF on raw data...")
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory_path, filename)
            X, y = load_and_prepare_data(filepath)

            for _, row in best_result.iterrows():
                n_neighbor = row['n_neighbors']
                n_features_to_select = row['n_features']
                print(f"Applying ReliefF for file {filename} with n_neighbors={n_neighbor}, n_features={n_features_to_select}")
                
                relief = ReliefF(n_neighbors=n_neighbor)
                relief.fit(X.values, y.values)

                # Obtention des indices des caractéristiques sélectionnées
                top_features_indices = relief.top_features_[:n_features_to_select]

                # Obtenez les scores d'importance des fonctionnalités sélectionnées
                feature_importances = relief.feature_importances_
                feature_names = X.columns.tolist()

                # Filtrer pour obtenir seulement les scores des caractéristiques sélectionnées
                selected_features_importance = pd.DataFrame({
                    'Feature': [feature_names[i] for i in top_features_indices],
                    'Score': [feature_importances[i] for i in top_features_indices]
                })

                # Trier les fonctionnalités sélectionnées par score d'importance décroissant
                selected_features_importance = selected_features_importance.sort_values(by='Score', ascending=False)

                # Chemin du fichier pour enregistrer les résultats
                # Notez que nous ne créons pas de sous-dossier ici
                output_file_path = os.path.join(output_folder, f'{filename[:-4]}_score_relief_data_brute.csv')
                
                # Enregistrer les scores d'importance des fonctionnalités dans un fichier CSV
                selected_features_importance.to_csv(output_file_path, index=False)
                print(f"Saved importance scores to {output_file_path}")'''

# Utilisation de la fonction Application_relief_optimise avec le dossier de sortie spécifié
output_folder = 'D:/detectFog/data/score_importance_relief_condition'
directory_path = 'D:/detectFog/data/ON_OFF_all_features/'

Application_relief_data_brute(best_result_data_brute, directory_path, output_folder)


stop_time_score_relief_data_brute =time.time()
total_time_score_relief_data_brute = stop_time_score_relief_data_brute - start_time_score_relief_data_brute
print(f"Temps d'exécution du ReliefF sur données brutes : {total_time_score_relief_data_brute:.2f} seconds")










###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################



