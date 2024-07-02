import os
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time

class ModelTraining:
    def __init__(self, train_folder, test_folder, top_feature_relief_folder, output_folder):
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.top_feature_relief_folder = top_feature_relief_folder
        self.output_folder = output_folder
        print(f"\n ModelTraining instance created with train_folder: {train_folder}, test_folder: {test_folder}, top_feature_relief_folder: {top_feature_relief_folder}, output_folder: {output_folder}\n")

    def load_train(self, method):
        data_dict = {}
        print(f"Loading training data with method: {method}")
        for file in os.listdir(self.train_folder):
            file_path = os.path.join(self.train_folder, file)
            if file.startswith('X_train') or file.startswith('y_train'):
                if f'_{method}.csv' in file:
                    identifier = file.split(f'_{method}.csv')[0].replace('X_train_', '').replace('y_train_', '')
                    identifier = identifier.split('_all_extraction_ON_OFF')[0]
                    if identifier not in data_dict:
                        data_dict[identifier] = {}
                    if 'X_train' in file:
                        data_dict[identifier]['X_train'] = pd.read_csv(file_path)
                        print(f"Loaded X_train for identifier: {identifier}")
                    elif 'y_train' in file:
                        data_dict[identifier]['y_train'] = pd.read_csv(file_path).squeeze()
                        print(f"Loaded y_train for identifier: {identifier}")
        print(f"Training data loaded with identifiers: {list(data_dict.keys())}")
        return data_dict

    def load_test(self, method):
        data_dict = {}
        print(f"Loading test data with method: {method}")
        for file in os.listdir(self.test_folder):
            file_path = os.path.join(self.test_folder, file)
            if file.startswith('X_test') or file.startswith('y_test'):
                if f'_{method}.csv' in file:
                    identifier = file.split(f'_{method}.csv')[0].replace('X_test_', '').replace('y_test_', '')
                    identifier = identifier.split('_all_extraction_ON_OFF')[0]
                    if identifier not in data_dict:
                        data_dict[identifier] = {}
                    if 'X_test' in file:
                        data_dict[identifier]['X_test'] = pd.read_csv(file_path)
                        print(f"Loaded X_test for identifier: {identifier}")
                    elif 'y_test' in file:
                        data_dict[identifier]['y_test'] = pd.read_csv(file_path).squeeze()
                        print(f"Loaded y_test for identifier: {identifier}")
        print(f"Test data loaded with identifiers: {list(data_dict.keys())}")
        return data_dict

    def load_feature_importances(self, method):
        print(f"Loading feature importances with method: {method}")
        feature_importances = {}
        for file in os.listdir(self.top_feature_relief_folder):
            if file.endswith(f'_all_extraction_ON_OFF_feature_ranking_relief_{method}.csv'):
                file_path = os.path.join(self.top_feature_relief_folder, file)
                identifier = file.split(f'_all_extraction_ON_OFF_feature_ranking_relief_{method}')[0]
                feature_importances[identifier] = pd.read_csv(file_path)
                print(f"Loaded feature importances for identifier: {identifier}")
        print(f"Feature importances loaded with identifiers: {list(feature_importances.keys())}")
        return dict(feature_importances)

    def group_data(self, data_dicts):
        print("Grouping data")
        grouped_data = defaultdict(dict)
        for data_dict in data_dicts:
            for key, value in data_dict.items():
                grouped_data[key].update(value)
        print("Data grouped")
        return dict(grouped_data)

    def select_features(self, feature_importances, X, top_n=10):
        top_features = feature_importances['Feature'].head(top_n).tolist()
        print(f"Selected top {top_n} features: {top_features}")
        return X[top_features]

    def calculate_distribution(self, y):
        distribution = y.value_counts().to_dict()
        print(f"Calculated distribution: {distribution}")
        return distribution

    def train_models(self, data_dict, feature_importances, top_n_values, method='raw'):
        print(f"Training models with top_n_values: {top_n_values} and method: {method}")
        results = []
        for top_n in top_n_values:
            for identifier, data in data_dict.items():
                print(f"Training with identifier: {identifier} and top_n: {top_n}")

                print(f"For train data")
                try:
                    X_train = self.select_features(feature_importances[identifier], data['X_train'], top_n)
                except KeyError as e:
                    print(f"Error: Identifier '{identifier}' not found in feature importances.")
                    continue

                y_train = data['y_train']
                
                print(f"For test data")
                try:
                    X_test = self.select_features(feature_importances[identifier], data['X_test'], top_n)
                except KeyError as e:
                    print(f"Error: Identifier '{identifier}' not found in feature importances.")
                    continue
                
                y_test = data['y_test']

                models = {
                    "RandomForest": RandomForestClassifier(random_state=42),
                }

                for model_name, model in models.items():
                    print(f"Training model: {model_name}")
                    start_train_time = time.time()
                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_train_time
                    print(f"Training time: {train_time} seconds")

                    start_predict_time = time.time()
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    predict_time = time.time() - start_predict_time
                    print(f"Prediction time: {predict_time} seconds")

                    auc = roc_auc_score(y_test, y_pred_proba)
                    cm = confusion_matrix(y_test, y_pred)
                    TN, FP, FN, TP = cm.ravel()

                    sensitivity = round((TP / (TP + FN)) * 100, 1)
                    specificity = round((TN / (TN + FP)) * 100, 1)
                    ppv = round((TP / (TP + FP)) * 100, 1)
                    npv = round((TN / (TN + FN)) * 100, 1)
                    accuracy = round(((TP + TN) / (TP + TN + FP + FN)) * 100, 1)
                    f_score = round(2 * (sensitivity * ppv) / (sensitivity + ppv), 1)
                    youden_index = round(sensitivity + specificity - 100, 1)

                    train_distribution = self.calculate_distribution(y_train)
                    test_distribution = self.calculate_distribution(y_test)

                    result = {
                        'Identifier': identifier,
                        'Model': model_name,
                        'Top N Features': top_n,
                        'CV Accuracy Mean Accuracy': cv_scores.mean(),
                        'Sensitivity': sensitivity,
                        'Specificity': specificity,
                        'PPV': ppv,
                        'NPV': npv,
                        'Accuracy': accuracy,
                        'F-score': f_score,
                        'Youden Index': youden_index,
                        'AUC': auc,
                        'Train Time': train_time,
                        'Predict Time': predict_time,
                        'Train Distribution 0': train_distribution.get(0, 0),
                        'Train Distribution 1': train_distribution.get(1, 0),
                        'Test Distribution 0': test_distribution.get(0, 0),
                        'Test Distribution 1': test_distribution.get(1, 0),
                        'Confusion Matrix TN': TN,
                        'Confusion Matrix FP': FP,
                        'Confusion Matrix FN': FN,
                        'Confusion Matrix TP': TP
                    }
                    results.append(result)

                    # Sauvegarde des résultats pour chaque modèle et patient
                    results_df = pd.DataFrame([result])
                    output_path = os.path.join(self.output_folder, f'{identifier}_{model_name}_{method}.csv')
                    if not os.path.exists(output_path):
                        results_df.to_csv(output_path, index=False)
                    else:
                        results_df.to_csv(output_path, mode='a', header=False, index=False)
                    print(f"Saved results to {output_path}\n")

        return results

# Exemple d'utilisation :
# train_folder = "C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01/4_train_ON_OFF"
# test_folder = "C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01/5_test_ON_OFF"
# importances_folder = "C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01/6_classement_features_ON_OFF"
# output_folder = "C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01"


