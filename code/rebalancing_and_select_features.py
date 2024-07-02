import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from skrebate import ReliefF
import time
from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm


class DataLoader:
    """
    Class responsible for loading and preparing data from a CSV file.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_prepare_data(self):
        """
        Loads data from the CSV file, filters columns with missing values,
        removes rows with 'transitionNoFog' in the 'label' column, and prepares
        the data for the model.

        Returns:
        - X: Features.
        - y: Binarized labels (1 for 'fog' and 'transitionFog', 0 for others).
        """
        print(f"Loading and preparing data for {self.filepath}")
        data = pd.read_csv(self.filepath)
        data_filtered = data.dropna(axis=1)
        data_filtered = data_filtered[data_filtered['label'] != 'transitionNoFog'] # remove 'transitionNoFog' rows
        X = data_filtered.drop('label', axis=1) # features
        y = data_filtered['label'].apply(lambda x: 1 if x in ['fog', 'transitionFog'] else 0) # binarize labels*
        print(f"Data loaded and prepared for {self.filepath}")
        return X, y


class ResamplingPipeline:
    """
    Class responsible for configuring oversampling and resampling pipelines.
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
        Standardizes the data and splits it into training and test sets. 
        because this allows all the data to be put on the same scale. This is very important for Relief f and Random Forest

        Returns:
        - X_train: Training features.
        - X_test: Test features.
        - y_train: Training labels.
        - y_test: Test labels.
        """
        print("Standardizing and splitting data")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns) # standardize data and keep column names
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, test_size=0.3, shuffle=True, stratify=self.y, random_state=42)
        print("Data standardized and split")
        return X_train, X_test, y_train, y_test

    def configure_pipeline_over(self):
        """
        Configures a pipeline with SMOTE to oversample minority data.

        Returns:
        - pipeline: Oversampling pipeline with SMOTE.
        """
        print("Configuring oversampling pipeline")
        steps = [('smote', SMOTE(sampling_strategy=self.smote_strategy_over, random_state=42)), 
                 ('model', DecisionTreeClassifier(random_state=42))]
        return ImPipeline(steps)

    def configure_pipeline_optimise(self):
        """
        Configures a pipeline with SMOTE and RandomUnderSampler to optimize resampling.

        Returns:
        - pipeline: Optimized pipeline with SMOTE and RandomUnderSampler.
        """
        print(f"Configuring optimized pipeline with SMOTE={self.smote_strategy_optimise} and UNDER={self.under_strategy_optimise}")
        steps = [('smote', SMOTE(sampling_strategy=self.smote_strategy_optimise, random_state=42)),
                 ('under_sampler', RandomUnderSampler(sampling_strategy=self.under_strategy_optimise, random_state=42)),
                 ('model', DecisionTreeClassifier(random_state=42))]
        return ImPipeline(steps)

    def configure_pipeline_with_best_strategies(self, smote_strategy, under_strategy):
        """
        Configures a pipeline with the best found strategies for SMOTE and RandomUnderSampler.

        Parameters:
        - smote_strategy: Optimal SMOTE strategy.
        - under_strategy: Optimal RandomUnderSampler strategy.

        Returns:
        - pipeline: Pipeline configured with the best strategies.
        """
        print(f"Configuring pipeline with SMOTE={smote_strategy} and UNDER={under_strategy}")
        steps = [('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
                 ('under_sampler', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
                 ('model', DecisionTreeClassifier(random_state=42))]
        return ImPipeline(steps)


class ModelEvaluator:
    """
    Class responsible for evaluating models.
    """
    def __init__(self, pipeline, X_train, y_train):
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train

    def evaluate_model(self):
        """
        Evaluates the model using stratified cross-validation and returns the average ROC AUC score.

        Returns:
        - mean_score: Average ROC AUC score.
        """
        print("Evaluating the model")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, self.X_train, self.y_train, scoring='roc_auc', cv=cv, n_jobs=1)
        mean_score = np.mean(scores)
        print(f"Average ROC AUC Score: {mean_score}")
        return mean_score


class DataSaver:
    """
    Class responsible for saving processed data.
    """
    def __init__(self, base_filename, directory_path):
        self.base_filename = base_filename
        self.directory_path = directory_path

    def save_data(self, X_train, X_test, y_train, y_test, suffix):
        """
        Saves training and test data to CSV files.

        Parameters:
        - X_train: Training features.
        - X_test: Test features.
        - y_train: Training labels.
        - y_test: Test labels.
        - suffix: Suffix to add to file names.
        """
        print(f"Starting to save data for {self.base_filename} with suffix {suffix}")
        train_path = f"{self.directory_path}/4_train_ON_OFF/X_train_{self.base_filename}_{suffix}.csv"
        train_label_path = f"{self.directory_path}/4_train_ON_OFF/y_train_{self.base_filename}_{suffix}.csv"
        test_path = f"{self.directory_path}/5_test_ON_OFF/X_test_{self.base_filename}_{suffix}.csv"
        test_label_path = f"{self.directory_path}/5_test_ON_OFF/y_test_{self.base_filename}_{suffix}.csv"

        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        pd.DataFrame(X_train).to_csv(train_path, index=False, header=True)
        print(f"Training data saved: {train_path}")
        pd.Series(y_train).to_csv(train_label_path, index=False, header=True)
        print(f"Training labels saved: {train_label_path}")
        pd.DataFrame(X_test).to_csv(test_path, index=False, header=True)
        print(f"Test data saved: {test_path}")
        pd.Series(y_test).to_csv(test_label_path, index=False, header=True)
        print(f"Test labels saved: {test_label_path}")
        print(f"Data saving complete for {self.base_filename} with suffix {suffix}\n")


class FileProcessor:
    """
    Class responsible for processing individual files.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.base_filename = os.path.basename(filepath).replace('.csv', '')
        self.loader = DataLoader(filepath)

    def process_file_over(self, save_directory, save_best_combinations_filepath):
        """
        Processes a file using the SMOTE oversampling strategy.

        Parameters:
        - save_directory: Directory to save processed data.

        Returns:
        - result_over: Dictionary containing the processing results.
        """
        print(f"Processing file {self.filepath} with oversampling")
        X, y = self.loader.load_and_prepare_data()
        resampler = ResamplingPipeline(X, y)
        pipeline = resampler.configure_pipeline_over()
        evaluator = ModelEvaluator(pipeline, resampler.X_train, resampler.y_train)
        mean_score = evaluator.evaluate_model()
        pipeline.fit(resampler.X_train, resampler.y_train)
        X_resampled, y_resampled = pipeline.steps[0][1].fit_resample(resampler.X_train, resampler.y_train)
        saver = DataSaver(self.base_filename, save_directory)
        saver.save_data(X_resampled, resampler.X_test, y_resampled, resampler.y_test, "over")
        result_over = {
            'File': self.filepath,
            'SMOTE Strategy': 1,
            'ROC AUC Score': mean_score,
            'Note': 'Resampling applied' if 1 != 'None' else 'No resampling due to class 1 >= class 0'
        }

        # Save the results of cross-validation with SMOTE = 1 in a CSV file
        results_df_over = pd.DataFrame([result_over])
        if not os.path.isfile(save_best_combinations_filepath):
            results_df_over.to_csv(save_best_combinations_filepath, index=False)
        else:
            results_df_over.to_csv(save_best_combinations_filepath, index=False)

        print(f"File {self.filepath} processed with oversampling")
        return result_over

    def process_file_optimise(self, save_directory, save_best_combinations_filepath):
        """
        Processes a file by optimizing SMOTE and RandomUnderSampler strategies.

        Parameters:
        - save_directory: Directory to save processed data.

        Returns:
        - result_optimise: Dictionary containing the processing results.
        """
        print(f"Processing file {self.filepath} with optimized resampling")
        X, y = self.loader.load_and_prepare_data()
        resampler = ResamplingPipeline(X, y)
        best_score = 0
        best_smote = None
        best_under = None
        for smote_strategy in resampler.smote_strategy_optimise:
            for under_strategy in resampler.under_strategy_optimise:
                try:
                    pipeline = resampler.configure_pipeline_with_best_strategies(smote_strategy, under_strategy)
                    evaluator = ModelEvaluator(pipeline, resampler.X_train, resampler.y_train)
                    score = evaluator.evaluate_model()

                    if score > best_score:
                        best_score = score
                        best_smote = smote_strategy
                        best_under = under_strategy

                except Exception as e:
                    error_message = "Ratio impossible" if "The specified ratio" in str(e) else str(e)
                    print(f"Error during processing: {error_message}")

        if best_smote and best_under:
            pipeline = resampler.configure_pipeline_with_best_strategies(best_smote, best_under)
            X_resampled, y_resampled = pipeline.steps[0][1].fit_resample(resampler.X_train, resampler.y_train)
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
        # Save the results of the best combinations in a CSV file
        results_df_optimise = pd.DataFrame([result_optimise])
        if not os.path.isfile(save_best_combinations_filepath):
            results_df_optimise.to_csv(save_best_combinations_filepath, index=False)
        else:
            results_df_optimise.to_csv(save_best_combinations_filepath, index=False)
        print(f"File {self.filepath} processed with optimized resampling")
        return result_optimise

    def save_raw_data_splits(self, save_directory):
        """
        Saves raw data after standardization and splitting into training and test sets.

        Parameters:
        - save_directory: Directory to save raw data.
        """
        print(f"Saving raw data splits for {self.filepath}")
        X, y = self.loader.load_and_prepare_data()
        resampler = ResamplingPipeline(X, y)
        saver = DataSaver(self.base_filename, save_directory)
        saver.save_data(resampler.X_train, resampler.X_test, resampler.y_train, resampler.y_test, "raw")
        print(f"Raw data splits saved for {self.filepath}")


class FeatureRankingProcessor:
    """
    Class responsible for applying ReliefF for feature ranking.
    """
    def __init__(self, train_folder, output_folder):
        self.train_folder = train_folder
        self.output_folder = output_folder

    def set_random_seed(self, seed): # set the seed for the random number generator to ensure reproducibility of relief F
        np.random.seed(seed)
        random.seed(seed)

    def load_train_data(self, data_type):
        """
        Loads training data from the specified folder.

        Parameters:
        - data_type: Type of data to load (e.g., 'raw', 'processed').

        Returns:
        - data_dict: Dictionary containing the training data.
        """
        data_dict = {}
        print(f"Loading '{data_type}' training data from folder: {self.train_folder}")

        for file in os.listdir(self.train_folder):
            file_path = os.path.join(self.train_folder, file)
            if file.endswith(f'_{data_type}.csv'):
                identifier = file.split(f'_{data_type}.csv')[0].replace('X_train_', '').replace('y_train_', '') 
                if identifier not in data_dict:
                    data_dict[identifier] = {}
                if 'X_train' in file:
                    print(f"Loading X_train for identifier: {identifier}")
                    data_dict[identifier]['X_train'] = pd.read_csv(file_path)
                elif 'y_train' in file:
                    print(f"Loading y_train for identifier: {identifier}")
                    data_dict[identifier]['y_train'] = pd.read_csv(file_path).squeeze()

        print(f"Data loaded for identifiers: {list(data_dict.keys())}")
        return data_dict

    def apply_relief(self, data_type='raw', random_seed=42):
        """
        Applies ReliefF for feature ranking.

        Parameters:
        - data_type: Type of data to process.
        - random_seed: Seed for random number generator.
        """
        print(f"\nStarting ReliefF application on '{data_type}' data")
        self.set_random_seed(random_seed)
        data_dict = self.load_train_data(data_type)

        for key, data in data_dict.items():
            X = data['X_train']
            y = data['y_train']

            print(f"\nProcessing dataset: {key}")
            start_time = time.time()

            print("Applying ReliefF...")
            relief = ReliefF()
            relief.fit(X.values, y.values)

            feature_importances = relief.feature_importances_ # importances scores of features
            feature_names = X.columns.tolist()

            features_importance = pd.DataFrame({
                'Feature': feature_names,
                'Score': feature_importances
            })

            features_importance_sorted = features_importance.sort_values(by='Score', ascending=False)

            os.makedirs(self.output_folder, exist_ok=True)
            output_file = os.path.join(self.output_folder, f'{key}_feature_ranking_relief_{data_type}.csv')
            features_importance_sorted.to_csv(output_file, index=False)

            stop_time = time.time()
            total_time = stop_time - start_time
            print(f"ReliefF execution time for {key} ({data_type}) = {total_time:.2f} seconds")
            print(f"Feature importances for {key} ({data_type}) saved to: {output_file}")


class DirectoryProcessor:
    """
    Class responsible for processing all files in a directory.
    """
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def process_directory_over(self, save_directory, results_filepath):
        """
        Processes all files in the directory using the SMOTE oversampling strategy.

        Parameters:
        - save_directory: Directory to save processed data.
        - results_filepath: File path to save the results.
        """
        results_df_over = pd.DataFrame()
        files = [f for f in os.listdir(self.directory_path) if f.endswith(".csv")]
        for filename in tqdm(files, desc="Processing files (over)"):
            filepath = os.path.join(self.directory_path, filename)
            processor = FileProcessor(filepath)
            result_over = processor.process_file_over(save_directory, results_filepath)
            results_df_over = pd.concat([results_df_over, pd.DataFrame([result_over])], ignore_index=True)
            print(f"Added results for {filename} to DataFrame.")
        results_df_over.to_csv(results_filepath, index=False)
        print(f"Results saved to {results_filepath}.")

    def process_directory_optimise(self, save_directory, results_filepath):
        """
        Processes all files in the directory by optimizing SMOTE and RandomUnderSampler strategies.

        Parameters:
        - save_directory: Directory to save processed data.
        - results_filepath: File path to save the results.
        """
        results_df_optimise = pd.DataFrame()
        files = [f for f in os.listdir(self.directory_path) if f.endswith(".csv")]
        for filename in tqdm(files, desc="Processing files (optimise)"):
            filepath = os.path.join(self.directory_path, filename)
            processor = FileProcessor(filepath)
            result_optimise = processor.process_file_optimise(save_directory, results_filepath)
            results_df_optimise = pd.concat([results_df_optimise, pd.DataFrame([result_optimise])], ignore_index=True)
            print(f"Added results for {filename} with SMOTE {result_optimise['SMOTE Strategy']} and UNDER {result_optimise['Under Strategy']} to DataFrame.")
        results_df_optimise.to_csv(results_filepath, index=False)
        print(f"Results saved to {results_filepath}.")

    def process_directory_raw(self, save_directory):
        """
        Saves raw data for all files in the directory.

        Parameters:
        - save_directory: Directory to save raw data.
        """
        files = [f for f in os.listdir(self.directory_path) if f.endswith(".csv")]
        for filename in tqdm(files, desc="Processing files (raw)"):
            filepath = os.path.join(self.directory_path, filename)
            processor = FileProcessor(filepath)
            processor.save_raw_data_splits(save_directory)
