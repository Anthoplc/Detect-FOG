import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from collections import defaultdict
import time

class MachineLearningProcessor:
    def __init__(self, root_directory, patient_id):
        self.root_directory = root_directory
        self.patient_id = patient_id
        self.result_directory = os.path.join(root_directory, 'resultats_machine_learning')
        os.makedirs(self.result_directory, exist_ok=True)
        print(f"Initialized MachineLearningProcessor for patient: {patient_id}")
        print(f"Results will be saved in: {self.result_directory}")

    def load_train(self, train_folder, method):
        data_dict = {}
        print(f"Loading training data from {train_folder} for method: {method}")
        for file in os.listdir(train_folder):
            if method in file:  # Filter files based on the method
                file_path = os.path.join(train_folder, file)
                identifier = file.split(f'_{method}')[0].replace('X_train_', '').replace('y_train_', '')
                if identifier not in data_dict:
                    data_dict[identifier] = {}
                if file.startswith('X_train'):
                    print(f"Loading X_train for {identifier}_{method}")
                    data_dict[identifier]['X_train'] = pd.read_csv(file_path)
                elif file.startswith('y_train'):
                    print(f"Loading y_train for {identifier}_{method}")
                    data_dict[identifier]['y_train'] = pd.read_csv(file_path).squeeze()
        print(f"Loaded training data for identifiers: {list(data_dict.keys())}\n")
        return data_dict

    def load_test(self, test_folder, method):
        data_dict = {}
        print(f"Loading test data from {test_folder} for method: {method}")
        for file in os.listdir(test_folder):
            if method in file:  # Filter files based on the method
                file_path = os.path.join(test_folder, file)
                identifier = file.split(f'_{method}')[0].replace('X_test_', '').replace('y_test_', '')
                if identifier not in data_dict:
                    data_dict[identifier] = {}
                if file.startswith('X_test'):
                    print(f"Loading X_test for {identifier}_{method}")
                    data_dict[identifier]['X_test'] = pd.read_csv(file_path)
                elif file.startswith('y_test'):
                    print(f"Loading y_test for {identifier}_{method}")
                    data_dict[identifier]['y_test'] = pd.read_csv(file_path).squeeze()
        print(f"Loaded test data for identifiers: {list(data_dict.keys())}\n")
        return data_dict

    def group_data(self, data_dicts):
        print("Grouping training and test data...")
        grouped_data = defaultdict(dict)
        for data_dict in data_dicts:
            for key, value in data_dict.items():
                grouped_data[key].update(value)
        print(f"Grouped data for identifiers: {list(grouped_data.keys())}")
        return dict(grouped_data)

    def load_feature_importances(self, importances_folder, method):
        feature_importances = {}
        print(f"Loading feature importances from {importances_folder} for method: {method}")
        for file in os.listdir(importances_folder):
            if file.endswith(f'classement_relief_{method}.csv'):
                file_path = os.path.join(importances_folder, file)
                identifier = file.split(f'_classement_relief_{method}')[0]
                print(f"Loading feature importances for {identifier}")
                feature_importances[identifier] = pd.read_csv(file_path)
        print(f"Loaded feature importances for identifiers: {list(feature_importances.keys())}\n")
        return feature_importances

    def select_features(self, feature_importances, X, top_n=10):
        top_features = feature_importances['Feature'].head(top_n).tolist()
        return X[top_features]

    def calculate_distribution(self, y):
        distribution = y.value_counts().to_dict()
        print(f"Calculated class distribution: {distribution}")
        return distribution

    def train_models(self, data_dict, feature_importances, top_n_values, method='brut'):
        results = []
        print(f"Starting model training for method: {method}")
        
        for top_n in top_n_values:
            print(f"Training models with top {top_n} features")
            for identifier, data in data_dict.items():
                if identifier not in feature_importances:
                    print(f"Skipping {identifier} as feature importances are not available")
                    continue

                print(f"Processing identifier: {identifier}_{method}")
                X_train = self.select_features(feature_importances[identifier], data['X_train'], top_n)
                y_train = data['y_train']
                X_test = self.select_features(feature_importances[identifier], data['X_test'], top_n)
                y_test = data['y_test']
                
                models = {
                    "RandomForest": RandomForestClassifier(random_state=42)
                }
                for model_name, model in models.items():
                    print(f"Training {model_name} for {identifier} with top {top_n} features")

                    start_train_time = time.time()
                    
                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                    print(f"Performing cross-validation for {model_name}")
                    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
                    
                    print(f"Fitting {model_name} on the entire training set")
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_train_time

                    start_predict_time = time.time()
                    print(f"Making predictions with {model_name}")
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    predict_time = time.time() - start_predict_time

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

                    print(f"Distribution of classes in training :")
                    train_distribution = self.calculate_distribution(y_train)
                    print(f"Distribution of classes in test :")
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
                    print(f"Completed training for {identifier} with {model_name} with {top_n} features\n")
        
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.result_directory, f"{self.patient_id}_RandomForest_{method}.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return results

# Example usage
root_directory = 'C:/Users/antho/Documents/MEMOIRE_M2/P_P_1963-04-01'
patient_id = os.path.basename(root_directory)

ml_processor = MachineLearningProcessor(root_directory, patient_id)

# Load train and test data for 'brut' method
train_data_brut = ml_processor.load_train(os.path.join(root_directory, 'train_ON_OFF'), 'brut')
test_data_brut = ml_processor.load_test(os.path.join(root_directory, 'test_ON_OFF'), 'brut')
data_dict_brut = ml_processor.group_data([train_data_brut, test_data_brut])

# Load feature importances for 'brut' method
feature_importances_brut = ml_processor.load_feature_importances(os.path.join(root_directory, 'classement_features_ON_OFF'), 'brut')

# Train models and save results for 'brut' method
results_brut = ml_processor.train_models(data_dict_brut, feature_importances_brut, top_n_values=[20,30], method='brut')

# You can repeat the above steps for 'optimise' and 'over100' methods as needed
