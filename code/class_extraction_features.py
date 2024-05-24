import pandas as pd
import matplotlib.pyplot as plt
from main import PreProcessing, Statistics
import numpy as np
import json
from scipy.stats import mode, median_abs_deviation, iqr, trim_mean, entropy as ent, skew, kurtosis
from scipy.signal import welch, correlate, stft
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft, fftfreq
import entropy as ent


class ExtractionFeatures:
    def __init__(self, data):
        ################### Initialisation des attributs pour création JSON ###################
        self.data = data # Données provenant de la classe PreProcessing
        self.fs = 50 # Fréquence d'échantillonnage  
        self.fft_magnitudes = None
        self.frequencies = None

        # On enlève la colonne label et la dernière frame avec les NA
    def enlever_derniere_ligne_et_colonne_label(self):
        for sensor, sensor_data in self.data.items():
            if sensor not in ["metadata", "parcours", "FOG"]:
                for side, side_data in sensor_data.items():
                    for measure, measure_data in side_data.items():
                        for axis, axis_data in measure_data.items():
                            if isinstance(axis_data, pd.DataFrame):
                            
                                # Supprimer la dernière ligne du DataFrame
                                data_moins_derniere_ligne_na = axis_data.drop(axis_data.index[-1])
                                # print(data_moins_derniere_ligne_na)
                                label = data_moins_derniere_ligne_na["label"]
                            
                                # Vérifier si la colonne 'label' existe avant de la supprimer
                                if 'label' in data_moins_derniere_ligne_na.columns:
                                    data_moins_colonne_label = data_moins_derniere_ligne_na.drop(columns=["label"])
                                    # Mise à jour du DataFrame dans le dictionnaire
                                    measure_data[axis] = data_moins_colonne_label
                                    
        return self.data, label


    def transformation_domaine_frequentiel (self, axis_data):
        # Nombre de points de données par fenêtre
        n = axis_data.shape[1]  # ou 100 si c'est connu

        # Créer un tableau de fréquences
        frequences = fftfreq(n, d=1/self.fs)
        frequences = frequences[:n//2] # obligé de laisser la data en série, pour générer le graphique des spectres de magnitudes
        frequencies = fftfreq(n, d=1/self.fs)
        frequencies = frequencies[:n//2]

        # Transposer le tableau de fréquences pour le mettre en colonnes
        frequencies = frequencies.reshape((1, -1))



        # calculer la transformée de Fourier
        fft_result = fft(axis_data, axis = 1)
        fft_magnitudes = np.abs(fft_result)[:,:n//2] # Garder uniquement les valeurs positives, puisque d'après la symétrie de la FFT, les valeurs négatives sont les mêmes que les valeurs positives


        # # Créer un DataFrame pour stocker les magnitudes des fréquences
        fft_magnitudes = pd.DataFrame(fft_magnitudes)
        frequencies = pd.DataFrame(frequencies)

        return self.fft_magnitudes, self.frequencies

    #____________________________________________________________________________________________________________________________

    # On calcul les caractéristiques temporelles
    def extract_temporal_features(self,axis_data):
        
        # Initialise un DataFrame vide pour stocker les caractéristiques
        df_features = pd.DataFrame()
        
        # Moyenne
        df_features['Mean_Temporal'] = np.mean(axis_data, axis=1)
        
        # Écart type
        df_features['Ecart_Type_Temporal'] = np.std(axis_data, axis=1)
        
        # Variance
        df_features['Variance_Temporal'] = np.var(axis_data, axis=1)
        
        # Énergie
        df_features['Energy_Temporal'] = np.sum(np.square(axis_data), axis=1)
        
        # Range
        df_features['Range'] = np.ptp(axis_data, axis=1)
        
        # Root mean square
        df_features['RMS'] = np.sqrt(np.mean(np.square(axis_data), axis=1))
        
        # Médiane
        df_features['Median_Temporal'] = np.median(axis_data, axis=1)
        
        # Trimmed mean
        df_features['Trimmed_Mean'] = trim_mean(axis_data, 0.1, axis=1)
        
        # Mean absolute value
        df_features['Mean_Absolute_Value'] = np.mean(np.abs(axis_data), axis=1)
        
        # Median absolute deviation
        df_features['Median_Absolute_Deviation'] = median_abs_deviation(axis_data, axis=1, nan_policy='omit')
        
        # Percentiles
        df_features['25th_percentile'] = np.percentile(axis_data, 25, axis=1)
        
        df_features['75th_percentile'] = np.percentile(axis_data, 75, axis=1)
        
        # Interquantile range
        df_features['Interquartile_range'] = iqr(axis_data, axis=1, rng=(25,75), nan_policy="omit")
        
        # Skewness
        df_features['Skewness_Temporal'] = skew(axis_data, axis=1)
        
        # Kurtosis
        df_features['Kurtosis_Temporal'] = kurtosis(axis_data, axis=1)
        
        # Incréments moyennes
        mean = np.mean(axis_data, axis=1)
        df_features['Increments_Mean'] = np.diff(mean, prepend=mean[0])
        
        
        
        # # Coefficients d'autorégression
        # fenetres = [np.array(window) for window in data.values]
        # max_order = 9
        # best_orders = {}
        # best_mse = np.inf 
        # tscv = TimeSeriesSplit(n_splits=4)

        # for i, fenetre in enumerate(fenetres):
        #     best_order = None
        #     best_mse = np.inf
        
        #     for p in range(1, max_order + 1):
        #         mse_scores = []
            
        #         for train_index, test_index in tscv.split(fenetre):
        #             train_data, test_data = fenetre[train_index], fenetre[test_index]
                
        #             model = AutoReg(train_data, lags=p)
        #             result = model.fit()
        #             predictions = result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        #             mse = mean_squared_error(test_data, predictions)
        #             mse_scores.append(mse)
            
        #         avg_mse = np.mean(mse_scores)

        #         if avg_mse < best_mse:
        #             best_mse = avg_mse
        #             best_order = p
        
        #     best_orders[i] = best_order

        # coefficients_autoreg = []

        # for i, fenetre in enumerate(fenetres):
        #     modele_ar = AutoReg(fenetre, lags=best_orders[i])
        #     resultat = modele_ar.fit()
        #     coefficients = resultat.params[best_orders[i]]
        #     coefficients_autoreg.append(coefficients) 
        # df_features['Ar_Coefficients'] = coefficients_autoreg 

        # Coefficient de variation
        df_features['Coefficient_Variation'] = np.std(axis_data, axis=1) / np.mean(axis_data, axis=1)
        
        # Normalized signal magnitude area
        # features['normalized_signal_magnitude_area'] = np.sum(np.abs(np.diff(data, axis=1)), axis=1) / data.shape[1]
        
        # Mean crossing rate
        # features['mean_crossing_rate'] = np.mean(np.diff(data > np.mean(data, axis=1, keepdims=True), axis=1), axis=1)
        
        # Signal vector magnitude
        # features['signal_vector_magnitude'] = np.sqrt(np.sum(np.square(data), axis=1))
        
        # # Incréments
        # # Calculer les différences entre les éléments consécutifs de chaque ligne
        # diffs = np.diff(data, axis=1)
        # # Insérer une colonne de zéros au début de chaque ligne
        # features['increments'] = np.hstack((np.zeros((data.shape[0], 1)), diffs))
        # df_features['increments'] = features['increments']
        
        # Entropie
        # features['entropy'] = entropy(data, axis=1)
        
        # Pic de la transformée de Fourier (FFT)
        # f, Pxx = welch(data, axis=1)
        # features['fft_peak'] = f[np.argmax(Pxx, axis=1)]
        # df_features['fft_peak'] = features['fft_peak']

        return df_features
    #____________________________________________________________________________________________________________________________


    # On calcul les caractéristiques fréquentielles
    def calcul_entropie_spectrale (self):
        # Calculer l'entropie spectrale de puissance pour chaque fenêtre
        entropie_spectrale = []

        for index, row in self.fft_magnitudes.iterrows():
            # Calculer les proportions pi de la puissance spectrale
            puissance_totale = np.sum(row**2)
            p_i = (row**2) / puissance_totale
        
            # Filtrer les valeurs de p_i égales à 0 pour éviter les erreurs de log(0)
            p_i = p_i[p_i > 0]

            # Calculer l'entropie spectrale pour la fenêtre actuelle
            H = -np.sum(p_i * np.log(p_i))
            entropie_spectrale.append(H)

        # Convertir la liste d'entropie en un tableau numpy pour une manipulation facile
        df_entropie_spectrale = pd.DataFrame({'Entropie_Spectrale': entropie_spectrale})
        return df_entropie_spectrale


    def calcul_details_harmoniques (self):

        # Initialiser les listes pour stocker les résultats
        premiere_harmonique_mag = []
        deuxieme_harmonique_mag = []
        premiere_harmonique_freq = []
        deuxieme_harmonique_freq = []
        distance_harmonique_frequence = []  # Liste pour stocker la distance entre les harmoniques
        distance_harmonique_magnitude = []
        centre_densite_spectrale = []
        centre_densite_spectrale_puissance = []
        rapport_harmonique_frequence = []
        rapport_harmonique_magnitude = []
        crete_spectrale_puissance_ponderee_gpt = []
        crete_spectrale_puissance_ponderee_borzi = []
        largeurs_harmoniques = []


        # Itérer sur chaque fenêtre
        for index, row in self.fft_magnitudes.iterrows():
            magnitudes = row.values
            frequences = self.frequencies.values.flatten() # Assumer que les fréquences sont constantes et identiques pour toutes les fenêtres
        
            # Trouver les indices des deux plus grandes magnitudes
            indices_harmoniques = np.argsort(magnitudes)[-2:]  # Cela nous donne les indices du second puis du premier
        
            # Assurer que l'indice de la première harmonique est celui de la plus grande magnitude
            if magnitudes[indices_harmoniques[0]] > magnitudes[indices_harmoniques[1]]:
                premiere_harmonique, deuxieme_harmonique = indices_harmoniques[0], indices_harmoniques[1]
            else:
                premiere_harmonique, deuxieme_harmonique = indices_harmoniques[1], indices_harmoniques[0]
        
            # Calculer le centre de densité spectrale (CDS)
            cds = np.sum(frequences * magnitudes) / np.sum(magnitudes)
        
            # Calculer le centre de densité spectrale de puissance
            cds_puissance = np.sum(frequences * magnitudes**2) / np.sum(magnitudes**2)
        
            # Calcul de la crête spectrale de puissance pondérée selon GPT
            cs_puissance_ponderee_gpt = np.max(magnitudes**2) / np.sum(magnitudes**2)
        
            # Calcul de la crête spectrale de puissance pondérée selon Borzi
            cs_puissance_ponderee_borzi = ((magnitudes[premiere_harmonique]**2) * frequences[premiere_harmonique])
        
            # Stocker les résultats
            premiere_harmonique_mag.append(magnitudes[premiere_harmonique])
            deuxieme_harmonique_mag.append(magnitudes[deuxieme_harmonique])
            premiere_harmonique_freq.append(frequences[premiere_harmonique])
            deuxieme_harmonique_freq.append(frequences[deuxieme_harmonique])
            centre_densite_spectrale.append(cds)
            centre_densite_spectrale_puissance.append(cds_puissance)
            crete_spectrale_puissance_ponderee_gpt.append(cs_puissance_ponderee_gpt)
            crete_spectrale_puissance_ponderee_borzi.append(cs_puissance_ponderee_borzi)
        
            # Calculer et stocker la distance de fréquence entre les harmoniques
            distance_harmonique_frequence.append(abs(frequences[premiere_harmonique] - frequences[deuxieme_harmonique]))
        
        
            # Pour éviter Inf, vérifier si le dénominateur est zéro
            if frequences[deuxieme_harmonique] == 0:
                rapport_harmonique_frequence.append(0)
            else:
                rapport_harmonique_frequence.append(frequences[premiere_harmonique] / frequences[deuxieme_harmonique])
        
        
            # Calculer et stocker la distance de magnitude entre les harmoniques
            distance_harmonique_magnitude.append(abs(magnitudes[premiere_harmonique] - magnitudes[deuxieme_harmonique]))
        
            # De même, éviter Inf pour les magnitudes
            if magnitudes[deuxieme_harmonique] == 0:
                rapport_harmonique_magnitude.append(0)
            else:
                rapport_harmonique_magnitude.append(magnitudes[premiere_harmonique] / magnitudes[deuxieme_harmonique])
            
            
            
            # Calculer la largeur des harmoniques
                # Calculer la magnitude de la première harmonique
            premiere_harmonique_magnitude = magnitudes[premiere_harmonique]
        
            # Utiliser la bonne méthode pour trouver les indices gauche et droite
            # Trouver l'indice de gauche
            gauche = np.where(magnitudes[:premiere_harmonique] < premiere_harmonique_magnitude * 0.5)[0]
            if len(gauche) > 0:
                indice_gauche = gauche[-1] + 1  # Prendre le dernier indice sous le seuil et ajouter 1
            else:
                indice_gauche = 0  # S'il n'y a pas de valeur sous le seuil, prendre le début du signal
        
            # Trouver l'indice de droite
            droite = np.where(magnitudes[premiere_harmonique+1:] < premiere_harmonique_magnitude * 0.5)[0]
            if len(droite) > 0:
                indice_droite = droite[0] + premiere_harmonique + 1  # Prendre le premier indice sous le seuil après le pic
            else:
                indice_droite = len(magnitudes) - 1  # S'il n'y a pas de valeur sous le seuil, prendre la fin du signal
        
            # Calculer la largeur en Hz
            largeur_hz = frequences[indice_droite] - frequences[indice_gauche]
            largeurs_harmoniques.append(largeur_hz)

        # Créer un DataFrame pour les résultats
        df_resultats = pd.DataFrame({
            'Premiere_Harmonique_Magnitude': premiere_harmonique_mag,
            'Deuxieme_Harmonique_Magnitude': deuxieme_harmonique_mag,
            'Premiere_Harmonique_Frequence': premiere_harmonique_freq,
            'Deuxieme_Harmonique_Frequence': deuxieme_harmonique_freq,
            'Distance_Harmonique_Frequence': distance_harmonique_frequence,
            'Distance_Harmonique_Amplitude':  distance_harmonique_magnitude,
            'Rapport_Harmonique_Frequence': rapport_harmonique_frequence,
            'Rapport_Harmonique_Amplitude':  rapport_harmonique_magnitude,
            'Centre_Densite_Spectrale': centre_densite_spectrale,
            'Centre_Densite_Spectrale_Puissance': centre_densite_spectrale_puissance,
            'Crete_Spectrale_Puissance_Ponderee_GPT': crete_spectrale_puissance_ponderee_gpt,
            'Crete_Spectrale_Puissance_Ponderee_Borzi': crete_spectrale_puissance_ponderee_borzi,
            'Largeur_Harmonique': largeurs_harmoniques
            })
        
        return df_resultats


    def ecart_type_borne (self):
        # Définissons les bandes de fréquences spécifiées
        bandes_frequence = {
            'ecart_type': (0, 50),
            'ecart_type_0.04_0.68_Hz': (0.04, 0.68),
            'ecart_type_0.68_3_Hz': (0.68, 3),
            'ecart_type_3_8_Hz': (3, 8),
            'ecart_type_8_20_Hz': (8, 20),
            'ecart_type_0.1_8_Hz': (0.1, 8)
        }

        # Extrayons les fréquences depuis le fichier frequence.csv pour l'associer à chaque colonne de magnitude_frequence.csv
        frequences = self.frequencies.values.flatten()

        # Créons un DataFrame pour stocker les écarts types calculés pour chaque bande de fréquence et pour chaque ligne (fenêtre)
        ecarts_types = pd.DataFrame()

        # Pour chaque bande de fréquence, filtrons les données et calculons l'écart type
        for nom_bande, (freq_min, freq_max) in bandes_frequence.items():
        
            # Identifions les colonnes correspondant à la bande de fréquence
            colonnes_bande = (frequences >= freq_min) & (frequences <= freq_max)
        
            # Filtrons les magnitudes pour cette bande de fréquence
            magnitudes_bande = self.fft_magnitudes.loc[:, colonnes_bande]
        
            # Calculons l'écart type pour cette bande de fréquence et ajoutons les résultats au DataFrame
            ecarts_types[nom_bande] = magnitudes_bande.std(axis=1)
            
        return ecarts_types


    def calculer_freeze_index(self):
        """
        Calcule le Freeze Index pour chaque fenêtre de données.
        
        :param magnitudes: Un DataFrame ou un numpy array des magnitudes du spectre de puissance pour chaque fenêtre.
        :param frequences: Un numpy array des fréquences correspondant aux colonnes de magnitudes.
        :return: Un numpy array contenant le Freeze Index pour chaque fenêtre.
        """
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        frequences = self.frequencies.values.flatten()
        
        # Définissons une fonction interne pour calculer l'aire sous le spectre de puissance
        def calculer_aire_sous_spectre(frequences, magnitudes, freq_min, freq_max):
            indices_bande = (frequences >= freq_min) & (frequences <= freq_max)
            magnitudes_bande = magnitudes[:, indices_bande]
            aire_sous_spectre = np.trapz(magnitudes_bande, x=frequences[indices_bande], axis=1)
            return aire_sous_spectre

        # Bandes de fréquences pour le Freeze Index
        bande_freeze = (3, 8)  # Bande "freeze"
        bande_locomotrice = (0.5, 3)  # Bande "locomotrice"

        # Calcul de l'aire sous le spectre pour chaque bande
        aire_freeze = calculer_aire_sous_spectre(frequences, magnitudes, *bande_freeze)
        aire_locomotrice = calculer_aire_sous_spectre(frequences, magnitudes, *bande_locomotrice)

        # Calcul du Freeze Index
        freeze_index = (aire_freeze ** 2) / (aire_locomotrice ** 2)
        freeze_index_df = pd.DataFrame({'Freeze_Index': freeze_index})

        return freeze_index_df


    ## Fréquence de faible Puissance pour une bande fréquence entre 0 et 2 Hz
    def ratio_faible_puissance_entre_0_2Hz (self):
        # Assurez-vous que `magnitude_frequence_df` et `frequences` sont définis et chargés correctement
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        frequences = self.frequencies.values.flatten()

        ratios = []  # Pour stocker le ratio de chaque fenêtre
        psd = np.abs(magnitudes)**2  # Calcul de la densité spectrale de puissance
        puissance_totale = np.sum(psd, axis=1)  # Calcul de la puissance totale du signal pour chaque fenêtre
        
        # Filtrer pour obtenir la puissance dans la bande 0-2 Hz
        bande_indices = (frequences >= 0) & (frequences <= 2)
        psd_band= psd[:, bande_indices]
        puissance_bande = np.sum(psd_band, axis = 1)

        ratios = puissance_bande / puissance_totale
        ratios_df = pd.DataFrame({'Ratio_Faible_Puissance_0_2Hz': ratios})
        
        return ratios_df



    def skewness_band_freq (self):
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        frequences = self.frequencies.values.flatten()


        # Définissons les bandes de fréquences spécifiées
        bandes_frequence = {
            'skewness': (0, 50),
            'skewness_0.04_0.68_Hz': (0.04, 0.68),
            'skewness_0.68_3_Hz': (0.68, 3),
            'skewness_3_8_Hz': (3, 8),
            'skewness_8_20_Hz': (8, 20),
            'skewness_0.1_8_Hz': (0.1, 8)
        }

        # Créons un DataFrame pour stocker les écarts types calculés pour chaque bande de fréquence et pour chaque ligne (fenêtre)
        skwenesss = pd.DataFrame()

        # Pour chaque bande de fréquence, filtrons les données et calculons l'écart type
        for nom_bande, (freq_min, freq_max) in bandes_frequence.items():
        
            # Identifions les colonnes correspondant à la bande de fréquence
            colonnes_bande = (frequences >= freq_min) & (frequences <= freq_max)
        
            # Filtrons les magnitudes pour cette bande de fréquence
            magnitudes_bande = magnitudes[:, colonnes_bande]
        
            # Calculons l'écart type pour cette bande de fréquence et ajoutons les résultats au DataFrame
            skwenesss[nom_bande] = skew(magnitudes_bande,axis=1)
            
        return skwenesss



    def kurtosis_band_freq (self) :
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        frequences = self.frequencies.values.flatten()
        # Définissons les bandes de fréquences spécifiées
        bandes_frequence = {
            'kurtosis': (0, 50),
            'kurtosis_0.04_0.68_Hz': (0.04, 0.68),
            'kurtosis_0.68_3_Hz': (0.68, 3),
            'kurtosis_3_8_Hz': (3, 8),
            'kurtosis_8_20_Hz': (8, 20),
            'kurtosis_0.1_8_Hz': (0.1, 8)
        }

        # Créons un DataFrame pour stocker les écarts types calculés pour chaque bande de fréquence et pour chaque ligne (fenêtre)
        kurtosiss = pd.DataFrame()

        # Pour chaque bande de fréquence, filtrons les données et calculons l'écart type
        for nom_bande, (freq_min, freq_max) in bandes_frequence.items():
        
            # Identifions les colonnes correspondant à la bande de fréquence
            colonnes_bande = (frequences >= freq_min) & (frequences <= freq_max)
        
            # Filtrons les magnitudes pour cette bande de fréquence
            magnitudes_bande = magnitudes[:, colonnes_bande]
        
            # Calculons l'écart type pour cette bande de fréquence et ajoutons les résultats au DataFrame
            kurtosiss[nom_bande] = kurtosis(magnitudes_bande,axis=1)
        
        return kurtosiss

    def calcul_locomotion_band_power (self):
    # Locomotion band power
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        frequences = self.frequencies.values.flatten()
        # Filtrer pour obtenir la puissance dans la bande de locomotion (0.5-3 Hz)
        bande_locomotion_power_list = []
        psd = np.abs(magnitudes)**2 
        bande_locomotion = (frequences >= 0.5) & (frequences <= 3)
        psd_bande_locomotion = psd[:, bande_locomotion]
        puissance_bande_locomotion = np.sum(psd_bande_locomotion, axis=1)

        for window in puissance_bande_locomotion:
            bande_locomotion_power = window / 50
            bande_locomotion_power_list.append(bande_locomotion_power)
        
        df_bande_locomotion_power = pd.DataFrame({'Locomotion_Band_Power': bande_locomotion_power_list})
        return df_bande_locomotion_power


    def calcul_freeze_band_power (self):
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        frequences = self.frequencies.values.flatten()
        # Filtrer pour obtenir la puissance dans la bande de locomotion (3-8 Hz)
        bande_freeze_power_list = []
        psd = np.abs(magnitudes)**2 


        bande_freeze = (frequences >= 3) & (frequences <= 8)
        psd_bande_freeze = psd[:, bande_freeze]
        puissance_bande_freeze = np.sum(psd_bande_freeze, axis=1)

        for window in puissance_bande_freeze:
            bande_freeze_power = window / 50
            bande_freeze_power_list.append(bande_freeze_power)
        
        df_bande_freeze_power = pd.DataFrame({'Freeze_Band_Power': bande_freeze_power_list})
        return df_bande_freeze_power

    def calcul_band_power(self):
    # Locomotion band power
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        frequences = self.frequencies.values.flatten()
        bande_power_list = []
        psd = np.abs(magnitudes)**2 

        # Filtrer pour obtenir la puissance dans la bande de locomotion et de freezing (0.5-8 Hz)
        bande_power = (frequences >= 0.5) & (frequences <= 8)
        psd_bande_power = psd[:, bande_power]
        puissance_bande_power= np.sum(psd_bande_power, axis=1)

        for window in puissance_bande_power:
            bande_power = window / 50
            bande_power_list.append(bande_power)
        
        df_bande_power = pd.DataFrame({'Band_Power': bande_power_list})
        return df_bande_power



    def calcul_energie (self):
        magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
        energie_liste = []
        
        for window in magnitudes:
            # Calculer l'énergie de la fenêtre comme la somme des carrés des valeurs du signal divisée par la longueur de la fenêtre
            energie = np.sum(np.square(magnitudes)) / len(window)
            energie_liste.append(energie)
        
        # Créer un DataFrame pour stocker les résultats
        df_energie = pd.DataFrame({'Energie': energie_liste})
            
        return df_energie
    #____________________________________________________________________________________________________________________________


    # On récupére toutes les caractéristiques de chaque fenêtre, de chaque capteur dans undatframe commun
    def dataframe_caracteristiques_final(self):
        data_collect = []

        for sensor, sensor_data in self.data.items():
            if sensor not in ["metadata", "parcours", "FOG"]:
                for side, side_data in sensor_data.items():
                    for measure, measure_data in side_data.items():
                        for axis, axis_data in measure_data.items():
                            if isinstance(axis_data, pd.DataFrame):
                                # Application des fonctions pour calculer les caractéristiques temporelles
                                features_temporelles = self.extract_temporal_features(axis_data)
                                fft_magnitude, frequencies = self.transformation_domaine_frequentiel(axis_data)
                                entropie_spectrale = self.calcul_entropie_spectrale(fft_magnitude)
                                details_harmoniques = self.calcul_details_harmoniques(fft_magnitude, frequencies)
                                ecart_types = self.ecart_type_borne(fft_magnitude, frequencies)
                                freeze_index = self.calculer_freeze_index(fft_magnitude, frequencies)
                                ratio_faible_puissance = self.ratio_faible_puissance_entre_0_2Hz(fft_magnitude, frequencies)
                                skewness = self.skewness_band_freq(fft_magnitude, frequencies)
                                kurtosis = self.kurtosis_band_freq(fft_magnitude, frequencies)
                                locomotion_band_power = self.calcul_locomotion_band_power(fft_magnitude, frequencies)
                                freeze_band_power = self.calcul_freeze_band_power(fft_magnitude, frequencies)
                                band_power = self.calcul_band_power(fft_magnitude, frequencies)
                                energie = self.calcul_energie(fft_magnitude, frequencies)
                                

                                # Fusionner toutes les caractéristiques dans un seul DataFrame pour simplification
                                caract_features = pd.concat([features_temporelles,
                                                            entropie_spectrale,
                                                            details_harmoniques, 
                                                            ecart_types,
                                                            freeze_index,
                                                            ratio_faible_puissance,
                                                            skewness,
                                                            kurtosis,
                                                            locomotion_band_power,
                                                            freeze_band_power,
                                                            band_power,
                                                            energie], axis=1)

                                # Renommer les colonnes ici
                                caract_features.rename(columns={feature_name: f"{sensor}_{side}_{measure}_{axis}_{feature_name}" for feature_name in caract_features.columns}, inplace=True)
                                
                                data_collect.append(caract_features)

        # Concaténer toutes les données collectées en alignant les colonnes
        df_final = pd.concat(data_collect, axis=1)
        return df_final