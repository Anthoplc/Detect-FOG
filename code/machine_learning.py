import os
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time

class ModelTraining:
    def __init__(self, train_folder, test_folder, top_feature_relief_folder, output_folder):
        """
        Initialize the ModelTraining class with the specified folders.

        Args:
            train_folder (str): Path to the folder containing training data.
            test_folder (str): Path to the folder containing test data.
            top_feature_relief_folder (str): Path to the folder containing feature importance files.
            output_folder (str): Path to the folder where the output results will be saved.
        """
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.top_feature_relief_folder = top_feature_relief_folder
        self.output_folder = output_folder
        print(f"\nModelTraining instance created with train_folder: {train_folder}, test_folder: {test_folder}, top_feature_relief_folder: {top_feature_relief_folder}, output_folder: {output_folder}\n")

    def load_train(self, method):
        """
        Load the training data based on the specified method.

        Args:
            method (str): The method to identify the appropriate training files.

        Returns:
            dict: A dictionary containing the training data.
        """
        # Initialize an empty dictionary to store the training data
        data_dict = {}
        print(f"Loading training data with method: {method}")

        # Iterate over all files in the training folder
        for file in os.listdir(self.train_folder):
            file_path = os.path.join(self.train_folder, file)  # Get the full path of the file

            # Check if the file is an X_train or y_train file
            if file.startswith('X_train') or file.startswith('y_train'):
                
                # Check if the file matches the specified method
                if f'_{method}.csv' in file:
                    
                    # Extract the identifier from the file name
                    identifier = file.split(f'_{method}.csv')[0].replace('X_train_', '').replace('y_train_', '')
                    identifier = identifier.split('_all_extraction_ON_OFF')[0]
                    
                    # If the identifier is not already in the dictionary, add it
                    if identifier not in data_dict:
                        data_dict[identifier] = {}
                    
                    # If the file is an X_train file, load it into the dictionary
                    if 'X_train' in file:
                        data_dict[identifier]['X_train'] = pd.read_csv(file_path)
                        print(f"Loaded X_train for identifier: {identifier}")
                    
                    # If the file is a y_train file, load it into the dictionary
                    elif 'y_train' in file:
                        data_dict[identifier]['y_train'] = pd.read_csv(file_path).squeeze()
                        print(f"Loaded y_train for identifier: {identifier}")

        print(f"Training data loaded with identifiers: {list(data_dict.keys())}")
        
        #the aim of this function is to load the training (label and data) in a a common key
        return data_dict

    def load_test(self, method):
        """
        Load the test data based on the specified method.

        Args:
            method (str): The method to identify the appropriate test files.

        Returns:
            dict: A dictionary containing the test data.
        """
        # Initialize an empty dictionary to store the test data
        data_dict = {}
        print(f"Loading test data with method: {method}")

        # Iterate over all files in the test folder
        for file in os.listdir(self.test_folder):
            file_path = os.path.join(self.test_folder, file)  # Get the full path of the file

            # Check if the file is an X_test or y_test file
            if file.startswith('X_test') or file.startswith('y_test'):
                
                # Check if the file matches the specified method
                if f'_{method}.csv' in file:
                    
                    # Extract the identifier from the file name
                    identifier = file.split(f'_{method}.csv')[0].replace('X_test_', '').replace('y_test_', '')
                    identifier = identifier.split('_all_extraction_ON_OFF')[0]
                    
                    # If the identifier is not already in the dictionary, add it
                    if identifier not in data_dict:
                        data_dict[identifier] = {}
                    
                    # If the file is an X_test file, load it into the dictionary
                    if 'X_test' in file:
                        data_dict[identifier]['X_test'] = pd.read_csv(file_path)
                        print(f"Loaded X_test for identifier: {identifier}")
                    
                    # If the file is a y_test file, load it into the dictionary
                    elif 'y_test' in file:
                        data_dict[identifier]['y_test'] = pd.read_csv(file_path).squeeze()
                        print(f"Loaded y_test for identifier: {identifier}")

        print(f"Test data loaded with identifiers: {list(data_dict.keys())}")
        # the aime is the same as the load_train function
        return data_dict

    def load_feature_importances(self, method):
        """
        Load the feature importances based on the specified method.

        Args:
            method (str): The method to identify the appropriate feature importance files.

        Returns:
            dict: A dictionary containing the feature importances.
        """
        print(f"Loading feature importances with method: {method}")
        feature_importances = {}

        # Iterate over all files in the feature importance folder
        for file in os.listdir(self.top_feature_relief_folder):
            
            # Check if the file matches the specified method
            if file.endswith(f'_all_extraction_ON_OFF_feature_ranking_relief_{method}.csv'):
                file_path = os.path.join(self.top_feature_relief_folder, file)  # Get the full path of the file
                
                # Extract the identifier from the file name
                identifier = file.split(f'_all_extraction_ON_OFF_feature_ranking_relief_{method}')[0]
                
                # Load the feature importances into the dictionary
                feature_importances[identifier] = pd.read_csv(file_path)
                print(f"Loaded feature importances for identifier: {identifier}")

        print(f"Feature importances loaded with identifiers: {list(feature_importances.keys())}")  # allow to see the feature importance loaded
        return dict(feature_importances)

    def group_data(self, data_dicts):
        """
        Group train dictionary and test dictionary into a single dictionary to have the same key for train and test data.

        Args:
            data_dicts (list): A list of data dictionaries to be grouped.

        Returns:
            dict: A single dictionary containing the grouped data.
        """
        print("Grouping data")
        grouped_data = defaultdict(dict)

        # Iterate over each data dictionary
        for data_dict in data_dicts:
            
            # Iterate over each key-value pair in the data dictionary
            for key, value in data_dict.items():
                
                # Update the grouped data dictionary with the value
                grouped_data[key].update(value)

        print("Data grouped")
        return dict(grouped_data)

    def select_features(self, feature_importances, X, top_n=10):
        """
        Select the top N features from the dataset based on feature importances to keep only the most important features of train and test data.

        Args:
            feature_importances (pd.DataFrame): The dataframe containing feature importances.
            X (pd.DataFrame): The feature dataset.
            top_n (int): The number of top features to select.

        Returns:
            pd.DataFrame: The dataset with the selected top N features.
        """
        # Get the list of top N features based on their importance
        top_features = feature_importances['Feature'].head(top_n).tolist()
        print(f"Selected top {top_n} features: {top_features}")
        
        # Return the dataset with only the top N features
        return X[top_features]

    def calculate_distribution(self, y):
        """
        Calculate the distribution of target labels.

        Args:
            y (pd.Series): The target labels.

        Returns:
            dict: A dictionary containing the distribution of target labels.
        """
        # Calculate the value counts of the target labels
        distribution = y.value_counts().to_dict()
        print(f"Calculated distribution: {distribution}")
        
        # Return the distribution as a dictionary
        return distribution

    def train_models(self, data_dict, feature_importances, top_n_values, method='raw'):
        """
        Train machine learning models using the provided data and feature importances.

        Args:
            data_dict (dict): The dictionary containing the training and test data.
            feature_importances (dict): The dictionary containing feature importances.
            top_n_values (list): A list of values for the number of top features to select.
            method (str): The method used for processing (default is 'raw').

        Returns:
            list: A list of dictionaries containing the training results.
        """
        print(f"Training models with top_n_values: {top_n_values} and method: {method}")
        results = []

        # Iterate over each value of top N features to select
        for top_n in top_n_values:
            
            # Iterate over each identifier in the data dictionary
            for identifier, data in data_dict.items():
                print(f"Training with identifier: {identifier} and top_n: {top_n}")

                # Load and select the top N features for the training data and keep only the most important features
                print(f"For train data")
                try:
                    X_train = self.select_features(feature_importances[identifier], data['X_train'], top_n)
                except KeyError as e:
                    print(f"Error: Identifier '{identifier}' not found in feature importances.")
                    continue

                y_train = data['y_train']
                
                # Load and select the top N features for the test data and keep only the most important features
                print(f"For test data")
                try:
                    X_test = self.select_features(feature_importances[identifier], data['X_test'], top_n)
                except KeyError as e:
                    print(f"Error: Identifier '{identifier}' not found in feature importances.")
                    continue
                
                y_test = data['y_test']

                # Define the models to be trained
                models = {
                    "RandomForest": RandomForestClassifier(random_state=42),
                }

                # Train and evaluate each model
                for model_name, model in models.items():
                    print(f"Training model: {model_name}")
                    
                    # Measure the training time
                    start_train_time = time.time()
                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_train_time
                    print(f"Training time: {train_time} seconds")

                    # Measure the prediction time
                    start_predict_time = time.time()
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    predict_time = time.time() - start_predict_time
                    print(f"Prediction time: {predict_time} seconds")

                    # Calculate evaluation metrics
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

                    # Calculate the distribution of the target labels
                    train_distribution = self.calculate_distribution(y_train)
                    test_distribution = self.calculate_distribution(y_test)

                    # Store the results in a dictionary
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

                    # Save the results for each model and identifier
                    results_df = pd.DataFrame([result])
                    output_path = os.path.join(self.output_folder, f'{identifier}_{model_name}_{method}.csv')
                    if not os.path.exists(output_path):
                        results_df.to_csv(output_path, index=False)
                    else:
                        results_df.to_csv(output_path, mode='a', header=False, index=False)
                    print(f"Saved results to {output_path}\n")

        return results


