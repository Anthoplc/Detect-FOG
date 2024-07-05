
# 0. Installation 

## Creating Environment
    # Go to the Anaconda Prompt command line
    # Clone the repository
    conda env create -f path/to/environment.yml
    conda activate detect_fog

## Directory Structure

    your_project/
    │
    ├── code/
    │   ├── main.py
    │   ├── environment.yml
    │   ├── preprocessing.py
    │   ├── rebalancing_and_select_features.py
    │   └── machine_learning.py
    │
    └── data/
    |    ├── P_1_1900-00-00/
    |    │   ├── 1_OFF/
    |    │   │   └── c3d/
    |    │   │       └── c3d_off_files
    |    │   ├── 2_ON/
    |    │       └── c3d/
    |    │           └── c3d_on_files
    |    └── P_2_1900-00-00/
    |        ├── 1_OFF/
    |        │   └── c3d/
    |        │       └── c3d_off_files
    |        ├── 2_ON/
    |            └── c3d/
    |                └── c3d_on_files
    └── statistics/

    !!!!!! WARNING: It is important that the names of the patient files are written as above. !!!!!

## TO DO LIST
- It is important that each c3d contains a START and END event.
- Check the labelling of events in the c3d, there must be no succession of debut_fog or fin_fog and there must be no two events with exactly the same time.
  

## Running the main.py script
    cd path/to/main.py
    python main.py --patients_directories "path/to/patient_1" "path/to/another_patient" --statistics_directory "path/to/statistics" --top_n_values 10 20 --methods raw over optimise

    # You can choose to include as many top_n_values as you want
    # You can choose to use only one method or two or three
    
# I. Introduction
## Project objectives
The aim of the project is to develop systems for detecting episodes of Freezing of Gait (FOG) in patients suffering from Parkinson's disease using data from inertial units. More specifically, the objectives are :

### 1. individualisation of algorithms
To create individualised algorithms for each patient in order to determine whether there is a difference in performance compared with global algorithms integrating data from several patients.

### 2. Data rebalancing
Demonstrate that data rebalancing can improve model performance by addressing the imbalance between FOG and non-FOG classes, as FOG episodes are generally much less frequent than periods of normal walking, which leads to class imbalance and will create biases in machine learning models.

### 3. FOG diversity
Demonstrate that there is a diversity of FOG episodes between patients, implying different features to be implemented in the models for each patient, which will improve the performance of the algorithms.

# II. General context
The data is collected from C3D files containing information on movements captured by sensors placed on the patients. Each patient is equipped with 7 sensors (pelvis, thighs, shins and feet). Data from the three axes (X, Y, Z) of the gyroscope and accelerometer are retrieved during 18 runs (6 blocks of 3 conditions: simple task, motor task, cognitive task) promoting the FOG over two visits (ON: with full medication and OFF: without medication). 

# III. Data preprocessing (preprocessing.py file)

## 1. Description of C3D files and data
The C3D files contain raw data from the inertial units, from which we will extract the various FOG, path and movement events, etc., which we will store in the 1.OFF and 2.ON directories.

  Associated functions :
- [recuperer_evenements](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L56)


## 2. Filtering and resampling methods
The data is resampled to a target frequency of 50 Hz and a Butterworth low-pass filter is applied to remove high-frequency noise. This makes the data smoother and improves the quality of the extracted features.

  Associated functions :
- [butter_lowpass](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L71)
- [butter_lowpass_filter](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L88)
- [reechantillonnage_fc_coupure_et_association_labels_et_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L105)


## 3. Data normalisation
The data is normalized to ensure that all variables are on the same scale. We've also placed START and END on each c3d, to identify the patient's sitting phase. And we've also added the 3-axis norm for the gyro and accelerometer.

  Associated functions :
- [calcul_norme](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L159)
- [normalize_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L395)

Following these 3 parts, we have created a json function to gather all the information about the patient, the FOG events, the data associated with each sensor and each axis (X,Y,Z,norm): [creation_json_grace_c3d](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L187)

## 4. Segmentation of data into windows
The data is segmented into 2 second time windows with an offset of 0.2 seconds between each window. The labels created are FOG, transitionFOG, transitionNoFog, NoFOG. 

  Associated functions :
  - [decoupage_en_fenetres](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L422)
  - [label_fenetre](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L501)
  - [association_label_fenetre_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L558)
  - [concat_label_fenetre_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L591)

## 5. Characteristic extraction
Time and frequency domain characteristics are extracted for each window, sensor and axis.

  Associated functions :
- [dataframe_caracteristiques_final](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L1071)

## 6. Creating a statistics file
We have also created a Statistics class which allows us to retrieve the FOG time, the recording time and the percentage of FOG in relation to time for each c3d.

  Associated functions :
- [stats](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L669)

# IV. Rebalancing the data (rebalancing_and_select_features.py file)
For the rest of our algorithm, we have grouped the transition-FOG and FOG labels in class FOG (class =1) and the non-FOG labels in class Non-FOG (class = 0) and we have removed the transition-non-FOG labels. [load_and_prepare_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L23)

Given that in the literature we often encounter problems of class imbalance, we have chosen to compare two rebalancing methods against no rebalancing.

## 1. Oversampling (SMOTE)
The Synthetic Minority Over-sampling Technique (SMOTE) is used to generate synthetic examples of the minority class, in order to address class imbalance in the data and present fully balanced classes.

  Associated functions :
- [configure_pipeline_over](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L73)
- [process_file_over](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L188)

## 2. Optimised sampling
We applied a method using SMOTE and sub-sampling, in order to find the balance that would display the best rebalancing and thus limit the number of data items created and get closer to reality. To do this, we tested several over- and under-sampling percentages to select, using cross-validation, the best AUC score corresponding to the best combination of percentages.

  Associated functions :
- [configure_pipeline_optimise](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L85)
- [configure_pipeline_with_best_strategies](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L98)
- [evaluate_model](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L125)
- [process_file_optimise](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L225)

## 3. Raw sampling
In order to be able to compare the effectiveness of the rebalancing methods, we have also retained the raw balancing.

  Associated functions :
- [save_raw_data_splits](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L279)

# V. Feature selection (rebalancing_and_select_features.py file)

The ReliefF algorithm is used only on the raw (70%) training data, oversampled and optimised to assess the importance of features and select the most relevant for the detection of FOG episodes based on the rebalancing methods. This makes it possible to reduce the dimensionality of the data and focus on the most significant features and to observe whether the selection varies according to the rebalancing methods.

  Associated functions :
- [load_train_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L306)
- [apply_relief](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L335)
- [process_file_over](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L188)
- [process_file_optimise](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L225)
- [save_raw_data_splits](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L279)


# VI. Training and Evaluation of Machine Learning Models (machine_learning.py file)

## 1 Data loading and preparation
Training and test data are loaded and prepared using the features selected by ReliefF.

  Associated functions :
- [load_train](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L26)
- [load_test](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L73)
- [load_feature_importances](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L119)
- [select_features](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L174)

## 2. Model training
Machine learning models, such as RandomForestClassifier, are trained on the training data and evaluated using cross-validation. Each model is evaluated individually for each patient.

  Associated functions :
- [train_models](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L210)

## 3. Model evaluation
Model performance is assessed using metrics such as the area under the ROC curve (AUC), sensitivity and specificity. This measures the models' ability to detect FOG episodes accurately and reliably.
