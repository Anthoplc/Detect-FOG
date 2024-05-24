import pandas as pd
from sklearn.model_selection import train_test_split
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, accuracy_score
from skrebate import ReliefF



# Charger les données
data = pd.read_csv('C:/Users/antho/Documents/MEMOIRE_M2/data_final.csv')

# Identifier les colonnes contenant des valeurs NA
na_columns = data.columns[data.isna().any()].tolist()

# Supprimer les colonnes avec des NA
data_cleaned = data.dropna(axis=1)

# Supprimer les lignes où la colonne spécifique contient 'transitionNoFog'
data_filtered = data_cleaned[data_cleaned['label'] != 'transitionNoFog']

# Séparer les caractéristiques et les étiquettes
X = data_filtered.drop('label', axis=1)  # données issues desc caractéristiques
y = data_filtered['label'].apply(lambda X: 1 if X in ['fog', 'transitionFog'] else 0) # labels transofmré en cible et non cible

#_______________________________________________________________STANDARDISATION_______________________________________________________________
# Création d'un objet StandardScaler
scaler = StandardScaler()

# Adaptation du scaler aux données (calcul de la moyenne et de l'écart-type)
scaler.fit(X)  # X est votre DataFrame avec les caractéristiques

# Transformation des données de caractéristsiques
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42) # 30 des données sont gardés comme test

# Préparation du pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42)) # Remplacer par RandomForestClassifier()
])
# Avec RandomForest, on a des over = 0.3 et under = 0.6


# Grille de paramètres pour GridSearch
param_grid = {
    'smote__sampling_strategy': [0.3, 0.4, 0.5],
    'under_sampler__sampling_strategy': [0.5, 0.6, 0.7]
}

# Configuration et exécution de la recherche en grille
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print("Meilleurs paramètres : ", grid_search.best_params_)
print("Meilleur score :", grid_search.best_score_)

# Évaluation avec les meilleurs paramètres
predictions = grid_search.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Utilisation des meilleurs paramètres trouvés pour configurer le pipeline final
best_smote_strategy = grid_search.best_params_['smote__sampling_strategy']
best_under_strategy = grid_search.best_params_['under_sampler__sampling_strategy']

#_______________________________________________________________MAJ DES DONNEES AVEC UNDER ET OVER SAMPLING_______________________________________________________________
# Définition du pipeline avec SMOTE suivi de RandomUnderSampler
model_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=best_smote_strategy, random_state=42),  # Sur-échantillonnage à 1:1 ratio
    ('under_sampler', RandomUnderSampler(sampling_strategy=best_under_strategy, random_state=42)))
 # Sous-échantillonnage à ajuster selon besoin
])

# Application du pipeline sur les données d'entraînement
X_resampled, y_resampled = model_pipeline.fit_resample(X_train, y_train)

# Affichage des nouvelles distributions des classes
print("Distribution des classes avant le pipeline:", y_train.value_counts())
print("Nouvelle distribution des classes après le pipeline:", pd.Series(y_resampled).value_counts())




#_______________________________________________________________## Etape 1 : VC pour le nombre de voisin optimal à utiliser dans Relief F_______________________________________________________________

X_resampled_np = X_resampled.values if isinstance(X_resampled, pd.DataFrame) else X_resampled
y_resampled_np = y_resampled.values if isinstance(y_resampled, pd.Series) else y_resampled
grid_search.fit(X_resampled_np, y_resampled_np)

X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

# Création du pipeline incluant ReliefF et un classificateur
pipeline = Pipeline([
    ('feature_selection', ReliefF(n_neighbors=5)),  # Le nombre de voisins sera ajusté via GridSearchCV
    ('classification', RandomForestClassifier(random_state=42))
])

# Définition de la grille de paramètres pour ReliefF
param_grid = {
    'feature_selection__n_neighbors': [1,2,3,4,5,6,7,8,9,10,15,20]  # Différents nombres de voisins à tester
}

# Création de l'objet GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_resampled_np, y_resampled_np)

# Affichage des meilleurs paramètres et du meilleur score
print("Best parameters found:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

# Évaluation du meilleur modèle sur l'ensemble de test
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_np)
print("Test set accuracy:", accuracy_score(y_test_np, predictions))


#_______________________________________________________________## ## Etape 1 bis : VC pour le nombre de voisin optimal à utiliser dans Relief F + nb features_______________________________________________________________

# Creation of the pipeline including ReliefF and a classifier
pipeline = Pipeline([
    ('feature_selection', ReliefF(n_neighbors=5)),  # Initial neighbors, will be adjusted via GridSearchCV
    ('classification', RandomForestClassifier(random_state=42))
])

# Definition of the parameter grid for GridSearchCV
param_grid = {
    'feature_selection__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
    'feature_selection__n_features_to_select': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40, 50]
}

# Creating the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled_np, y_resampled_np)

# Display of the best parameters and best score
print("Best parameters found:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

# Evaluation of the best model on the test set
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_np)
print("Test set accuracy:", accuracy_score(y_test_np, predictions))