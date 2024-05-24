import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_with_top_features(num_features):
    # Charger les fichiers CSV
    X_train = pd.read_csv('C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/ON_OFF_all_features_final_optimise/train/X_train_A_P_1956-02-21_OFF_OFF_all_extraction_features_optimise.csv')
    y_train = pd.read_csv('C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/ON_OFF_all_features_final_optimise/train/y_train_A_P_1956-02-21_OFF_OFF_all_extraction_features_optimise.csv')
    feature_importances = pd.read_csv('C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/score_importance_relief_condition/A_P_1956-02-21_OFF_OFF_all_extraction_features_optimise_feature_importances.csv')
    X_test = pd.read_csv('C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/ON_OFF_all_features_final_optimise/test/X_test_A_P_1956-02-21_OFF_OFF_all_extraction_features_optimise.csv')
    y_test = pd.read_csv('C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/ON_OFF_all_features_final_optimise/test/y_test_A_P_1956-02-21_OFF_OFF_all_extraction_features_optimise.csv')


    # On transforme les étiquettes en Series
    y_train = y_train.iloc[:, -1]
    y_test = y_test.iloc[:, -1]
    # Sélectionner les caractéristiques les plus importantes
    top_features = feature_importances['Feature'].head(num_features).tolist()

    # Créer les nouveaux ensembles de données avec les caractéristiques sélectionnées
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    # Initialiser le modèle
    model = RandomForestClassifier(random_state=42)

    # Former le modèle
    model.fit(X_train_top, y_train)

    # Prédictions
    y_pred = model.predict(X_test_top)

    # Évaluer les performances
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

# Exemple d'utilisation
num_features_to_select = 20
accuracy, report = evaluate_model_with_top_features(num_features_to_select)
print(f"Accuracy with top {num_features_to_select} features: {accuracy}")
print(f"Classification report with top {num_features_to_select} features:\n{report}")

################################################################
################################################################
################################################################

import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_with_top_features(num_features, train_file, test_file, feature_importance_file):
    # Charger les fichiers CSV
    X_train = pd.read_csv(train_file)
    y_train = pd.read_csv(train_file.replace("X_train", "y_train"))
    feature_importances = pd.read_csv(feature_importance_file)
    X_test = pd.read_csv(test_file)
    y_test = pd.read_csv(test_file.replace("X_test", "y_test"))

    # Assurer que y_train et y_test sont des vecteurs 1D
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Sélectionner les caractéristiques les plus importantes
    top_features = feature_importances['Feature'].head(num_features).tolist()

    # Créer les nouveaux ensembles de données avec les caractéristiques sélectionnées
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    # Initialiser le modèle
    model = RandomForestClassifier(random_state=42)

    # Former le modèle
    model.fit(X_train_top, y_train)

    # Prédictions
    y_pred = model.predict(X_test_top)

    # Évaluer les performances
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

def process_files_in_directory(directory, num_features):
    train_files = glob.glob(os.path.join(directory, 'train', 'X_train_*.csv'))
    test_files = glob.glob(os.path.join(directory, 'test', 'X_test_*.csv'))
    feature_importance_files = glob.glob(os.path.join(directory, 'score_importance_relief_condition', '*.csv'))

    for train_file in train_files:
        test_file = train_file.replace('train', 'test').replace('X_train', 'X_test')
        feature_importance_file = os.path.join(directory, 'score_importance_relief_condition', os.path.basename(train_file).replace('X_train_', ''))
        
        if test_file in test_files and feature_importance_file in feature_importance_files:
            print(f"Processing {train_file}")
            accuracy, report = evaluate_model_with_top_features(num_features, train_file, test_file, feature_importance_file)
            print(f"Accuracy with top {num_features} features: {accuracy}")
            print(f"Classification report with top {num_features} features:\n{report}")

# Exemple d'utilisation
directory = 'C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/ON_OFF_all_features_final_optimise'
num_features_to_select = 20
process_files_in_directory(directory, num_features_to_select)

################################################################
################################################################
################################################################

import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_with_top_features(num_features, train_file, test_file, feature_importance_file):
    # Charger les fichiers CSV
    X_train = pd.read_csv(train_file)
    y_train = pd.read_csv(train_file.replace("X_train", "y_train"))
    feature_importances = pd.read_csv(feature_importance_file)
    X_test = pd.read_csv(test_file)
    y_test = pd.read_csv(test_file.replace("X_test", "y_test"))

    # Assurer que y_train et y_test sont des vecteurs 1D
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Sélectionner les caractéristiques les plus importantes
    top_features = feature_importances['Feature'].head(num_features).tolist()

    # Créer les nouveaux ensembles de données avec les caractéristiques sélectionnées
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    # Initialiser le modèle
    model = RandomForestClassifier(random_state=42)

    # Former le modèle
    model.fit(X_train_top, y_train)

    # Prédictions
    y_pred = model.predict(X_test_top)

    # Évaluer les performances
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

def process_files_in_directory(directory, num_features):
    train_files = glob.glob(os.path.join(directory, 'train', 'X_train_*.csv'))
    test_files = glob.glob(os.path.join(directory, 'test', 'X_test_*.csv'))
    feature_importance_files = glob.glob(os.path.join(directory, 'score_importance_relief_condition', '*.csv'))

    for train_file in train_files:
        test_file = train_file.replace('train', 'test').replace('X_train', 'X_test')
        feature_importance_file = os.path.join(directory, 'score_importance_relief_condition', os.path.basename(train_file).replace('X_train_', ''))
        
        if test_file in test_files and feature_importance_file in feature_importance_files:
            print(f"Processing {train_file}")
            accuracy, report = evaluate_model_with_top_features(num_features, train_file, test_file, feature_importance_file)
            print(f"Accuracy with top {num_features} features: {accuracy}")
            print(f"Classification report with top {num_features} features:\n{report}")

def process_single_file(train_file, num_features):
    directory = os.path.dirname(train_file)
    test_file = train_file.replace('train', 'test').replace('X_train', 'X_test')
    feature_importance_file = os.path.join(directory, '..', 'score_importance_relief_condition', os.path.basename(train_file).replace('X_train_', ''))
    
    if os.path.exists(test_file) and os.path.exists(feature_importance_file):
        print(f"Processing {train_file}")
        accuracy, report = evaluate_model_with_top_features(num_features, train_file, test_file, feature_importance_file)
        print(f"Accuracy with top {num_features} features: {accuracy}")
        print(f"Classification report with top {num_features} features:\n{report}")
    else:
        print("Corresponding test file or feature importance file not found.")

def main(path, num_features):
    if os.path.isdir(path):
        process_files_in_directory(path, num_features)
    elif os.path.isfile(path):
        process_single_file(path, num_features)
    else:
        print("The provided path is neither a file nor a directory.")

# Exemple d'utilisation
path = 'C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/ON_OFF_all_features_final_optimise'
num_features_to_select = 20
main(path, num_features_to_select)

# Pour traiter un seul fichier
#single_file_path = 'C:/Users/antho/Documents/MEMOIRE_M2/c3d_audeline/ON_OFF_all_features_final_optimise/train/X_train_A_P_1956-02-21_OFF_OFF_all_extraction_features_optimise.csv'
#main(single_file_path, num_features_to_select)

