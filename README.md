
# 0. Installation 

## Création environnement 
    # Aller dans l'indice de commande anacondaPrompt
    #cloner le dépôt 
    conda env create -f chemin d'accès/environment.yml
    conda activate detect_fog

## Structure des répertoires

    votre_projet/
    │
    ├── code/
    │   ├── main.py
    │   ├── environment.yml
    │   ├── preprocessing.py
    │   ├── rebalancing_and_select_features.py
    │   └── machine_learning.py
    │
    └── data/
    |    ├── patient_1/
    |    │   ├── 1_OFF/
    |    │   │   └── c3d/
    |    │   │       └── fichiers_c3d_off
    |    │   ├── 2_ON/
    |    │       └── c3d/
    |    │           └── fichiers_c3d_on
    |    └── patient_2/
    |        ├── 1_OFF/
    |        │   └── c3d/
    |        │       └── fichiers_c3d_off
    |        ├── 2_ON/
    |            └── c3d/
    |                └── fichiers_c3d_on
    └── statistics/

## Exécution du script main.py
    python /chemin/vers/main.py 
    --patients_directories "/chemin/vers/root_directory1" "/chemin/vers/root_directory2" 
    --statistics_directory "/chemin/vers/statistics_directory" 
    --top_n_values 10 20 
    --methods raw over optimise

# I. Introduction
## Objectifs du projet
Le projet vise à développer des systèmes de détection des épisodes de Freezing of Gait (FOG) chez les patients atteints de la maladie de Parkinson en utilisant des données issues de centrales inertielles. Plus précisément, les objectifs sont :

### 1. Individualisation des Algorithmes
Créer des algorithmes individualisés pour chaque patient afin de déterminer s'il y a une différence de performance par rapport à des algorithmes globaux intégrant les données de plusieurs patients.

### 2. Rééquilibrage des Données
Montrer que le rééquilibrage des données permet d'améliorer les performances des modèles en traitant le déséquilibre entre les classes FOG et non-FOG, car Len général les épisodes de FOG sont généralement beaucoup moins fréquents que les périodes de marche normale, ce qui entraîne un déséquilibre des classes et vont créer des biais dans les modèles de machine learning.

### 3. Diversité des FOG
Démontrer qu'il existe une diversité des épisodes de FOG entre les patients, impliquant des caractéristiques différentes à implémenter dans les modèles pour chaque patient, ce qui permet d'améliorer la performance des algorithmes.


# II. Contexte général
Les données sont collectées à partir de fichiers C3D contenant des informations sur les mouvements capturés par des capteurs placés sur les patients. Chaque patient est équipé de 7 capteurs (le pelvis, les cuisses, les tibias et les pieds. Les données provenant des trois axes (X, Y, Z) du gyroscope et de l'accéléromètre sont récupérées lors de 18 passages (6 blocs de 3 conditions : tâche simple, tâche motrice, tâche cognitive) favorisant le FOG sur deux visites (ON : avec pleine médication et OFF : sans médication). 

# III. Prétraitement des Données (fichier preprocessing.py)

## 1. Description des fichiers C3D et des données
Les fichiers C3D contiennent des données brutes provenant des centrales inertiels dans lesquelles, nous allons extraire les différents événements de FOG, de parcours, de mouvement, etc. qu'on va stocker dans le _**répertoire 1.OFF et 2.ON**_

  Fonctions associées :
- [recuperer_evenements](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L56)


## 2. Méthodes de filtrage et de rééchantillonnage
Les données sont rééchantillonnées à une fréquence cible de 50 Hz et un filtre passe-bas de Butterworth est appliqué pour éliminer les bruits haute fréquence. Cela permet de rendre les données plus lisses et d'améliorer la qualité des caractéristiques extraites.

  Fonction associées :
- [butter_lowpass](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L71)
- [butter_lowpass_filter](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L88)
- [reechantillonnage_fc_coupure_et_association_labels_et_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L105)


## 3. Normalisation des données
Les données sont normalisées pour garantir que toutes les variables sont sur la même échelle. Nous avons également placé des START et END sur chaque c3d, afin d'élminer la phase assise du patient. Et nous avons également ajouter la norme provenant des 3 axes pour le gyroscopre et l'accéléromètre

  Fonctions associées : 
- [calcul_norme](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L159)
- [normalize_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L395)

A la suite de ces 3 parties, nous avons créer une fonction json, permettant de réunir toutes les informations du patients, des événements de FOG, des données associées à chaque capteurs et chaque axe (X,Y,Z,norme) : [creation_json_grace_c3d](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L187)

## 4. Segmentation des données en fenêtres
Les données sont segmentées en fenêtres temporelles de 2 secondes avec un décalage de 0,2 seconde entre chaque fenêtre. Les étiquettes créées sont FOG, transitionFOG, transitionNoFog, NoFOG. 

  Fonctions associées :
  - [decoupage_en_fenetres](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L422)
  - [label_fenetre](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L501)
  - [association_label_fenetre_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L558)
  - [concat_label_fenetre_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L591)

## 5. Extraction des Caractéristiques
Des caractéristiques dans le domaine temporel et fréquentiel sont extraites pour chaque fenêtres, capteurs et axes.

  Fonctions associées :
- [dataframe_caracteristiques_final](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L1071)

## 6. Création d'un fichier statistiques
Nous avons également créer une class Statistics qui nous permets de récupérer le temps de FOG, le temps de l'enregistrement et le pourcentage de FOG par rapport au temps pour chaque c3d.

  Fonctions associées :
- [stats](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/preprocessing.py#L669)

# IV. Rééquilibrage des données (fichier rebalancing_and_select_features.py)
Pour toute la suite de notre algorithme nous avons regroupé les étiquettes transition-FOG et FOG en classe FOG (classe =1) et les étiquettes non-FOG en classe Non-FOG (classe = 0) et nous avons supprimé les étiquettes transition-non-FOG. [load_and_prepare_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L23)

Etant donné que dans la littérature, nous rencontrons souvent des problèmes de déséquiblibre de classes, nous avons choisi de comparer deux méthodes de rééquilibrages par rapport à aucun rééquilibrage.

## 1. Sur-échantillonnage (SMOTE)
Le Synthetic Minority Over-sampling Technique (SMOTE) est utilisé pour générer des exemples synthétiques de la classe minoritaire, afin de traiter le déséquilibre des classes dans les données et présenter des classes totalement équilibrées.

  Fonctions associées :
- [configure_pipeline_over](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L73)
- [process_file_over](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L188)

## 2. Echantillonnage optimisé
Nous avons appliqué une méthode utilisant SMOTE et du sous-échantillonnage, afin de trouver l'équilibre qui permettrait d'afficher le meilleurs rééquilibrage et ainsi limité le nombre de données créées et se rapprocher de la réalité. Pour cela, nous avons tester plusieurs pourcentage de sur et sous échantillonnage pour sélectionner grâce à la validation croisée le meilleur score AUC correspondant à la meilleure combinaison de pourcentage.

  Fonctions associées :
- [configure_pipeline_optimise](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L85)
- [configure_pipeline_with_best_strategies](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L98)
- [evaluate_model](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L125)
- [process_file_optimise](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L225)

## 3. Echantillonnage brute
Afin de pouvoir comparer l'efficacité des méthodes de rééquilibrage, nous avons également conservé l'équilibrage brute.

  Fonctions associées :
- [save_raw_data_splits](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L279)

# V. Sélection des caractéristiques (fichier rebalancing_and_select_features.py)

L'algorithme ReliefF est utilisé uniquement sur les données d'entrainement (70%) brutes, sur-échantillonné et optimisé pour évaluer l'importance des caractéristiques et sélectionner les plus pertinentes pour la détection des épisodes de FOG en fonction des méthodes de rééquilibre. Cela permet de réduire la dimensionnalité des données et de se concentrer sur les caractéristiques les plus significatives​​ et d'observer si la sélection varie en fonction des méthodes de rééquilibrage.

  Fonctions associées :
- [load_train_data](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L306)
- [apply_relief](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L335)
- [process_file_over](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L188)
- [process_file_optimise](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L225)
- [save_raw_data_splits](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/rebalancing_and_select_features.py#L279)


# VI. Formation et Évaluation des Modèles de Machine Learning (fichier machine_learning.py)

## 1. Chargement et préparation des données
Les données d'entraînement et de test sont chargées et préparées en utilisant les caractéristiques sélectionnées par ReliefF.

  Fonctions associées :
- [load_train](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L26)
- [load_test](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L73)
- [load_feature_importances](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L119)
- [select_features](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L174)

## 2. Entraînement des modèles
Les modèles de machine learning, tels que RandomForestClassifier, sont entraînés sur les données d'entraînement et évalués à l'aide de validation croisée. Chaque modèle est évalué individuellement pour chaque patient.

  Fonctions associées :
- [train_models](https://github.com/Anthoplc/Detect-FOG/blob/674be32f6b65143bb53ccc18b7cf0e8b94f00846/code/machine_learning.py#L210)

## 3. Évaluation des modèles
Les performances des modèles sont évaluées à l'aide de métriques telles que l'aire sous la courbe ROC (AUC), la sensibilité, la spécificité. Cela permet de mesurer la capacité des modèles à détecter les épisodes de FOG de manière précise et fiable​​.
