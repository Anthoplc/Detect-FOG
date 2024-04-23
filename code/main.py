import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import resample,butter, lfilter, freqz
import ezc3d
import numpy as np
import json
from scipy.stats import mode, median_abs_deviation, iqr, trim_mean, entropy as ent, skew, kurtosis
from scipy.signal import welch, correlate, stft
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft, fftfreq
import entropy as ent


class PreProcessing:
    def __init__(self, file_path):
        
        ################### Initialisation des attributs pour création JSON ###################
        self.file_path = file_path # Chemin du fichier c3d
        self.c3d = ezc3d.c3d(file_path) # Lire le fichier c3d
        self.events_dict = None # Dictionnaire pour stocker les événements
        self.resampled_times = None # Temps rééchantillonné
        self.filtered_fusion_label_data = None # Données filtrées
        self.labels_data_filtre = None  # Données filtrées avec uniquement les données de GYRO et ACC excluant MAG
        self.labels_data_filtre_modifies = None # Données filtrées avec ajout d'un côté avec pelvis
        self.normes = None
        self.json_data = None
        ################### Fin Initialisation des attributs pour création JSON ###################
        
        
        ################### Initialisation des attributs pour données entre START et END ###################
        self.data_interval = None
        ################### Fin Initialisation des attributs pour données entre START et END ###################
        
        
        ################### Initialisation des attributs de normalisation ###################
        self.normalized_data = None
        ################### Fin Initialisation des attributs de normalisation ###################
        
        ################### Initialisation des attributs pour découpage en fenêtres ###################
        self.taille_fenetre = 2 # Taille de la fenêtre en secondes
        self.decalage = 0.2 #20% de la taille de la fenêtre
        self.taux_echantillonnage = 50    
        self.fenetres_data = None
        self.infos_fenetres = None    
        ################### Fin Initialisation des attributs pour découpage en fenêtres ###################
        
        ################### Initialisation des attributs pour labelisation des fenêtres ###################
        self.temps_prefog = 3
        self.debuts_prefog = None
        ################### Fin Initialisation des attributs pour labelisation des fenêtres ###################
        
        ################### Initialisation des attributs pour association labels fenetres aux datas ###################
        self.mix_label_fenetre_data = None
        ################### Fin Initialisation des attributs pour association labels fenetres aux datas ###################

        ################### Initialisation des attributs pour concaténation des labels et des données ###################
        self.concat_data = None
        ################### Fin Initialisation des attributs pour concaténation des labels et des données ###################

################### Création JSON ###################

    def recuperer_evenements(self):
        """
    Fonction pour récupérer les événements à partir des données C3D.

    Explications :
    - events : Liste des noms des événements.
    - temps_events : Liste des temps associés à chaque événement.
    - df : DataFrame contenant les événements et leurs temps.
    - events_dict : Dictionnaire associant chaque événement à ses temps.

    Retour :
    - Aucun, mais met à jour l'attribut events_dict de l'instance.
        """
        events = self.c3d['parameters']['EVENT']['LABELS']['value'] # Récupérer les événements
        temps_events = self.c3d['parameters']['EVENT']['TIMES']['value'] # Récupérer les temps des événements
        # df = pd.DataFrame({'events': events, 'frames': temps_events[1]}) # Créer un dataframe avec les événements et les temps
        # df.reset_index(drop=True, inplace=True) # Supprimer la colonne index
        # self.events_dict = df.groupby('events')['frames'].apply(list).to_dict() # Créer un dictionnaire avec les événements et les temps associés


                # Standardiser les noms des événements FoG
        standardized_events = ['FOG_begin' if event == 'FoG_Begin' else event for event in events]
        standardized_events = ['FOG_end' if event == 'FoG_end' else event for event in standardized_events]

        df = pd.DataFrame({'events': standardized_events, 'frames': temps_events[1]})  # Créer un dataframe avec les événements et les temps
        df.reset_index(drop=True, inplace=True)  # Supprimer la colonne index
        self.events_dict = df.groupby('events')['frames'].apply(list).to_dict()    


    def butter_lowpass(self, cutoff, fs, order=2):
        """
    Fonction pour calculer les coefficients du filtre passe-bas Butterworth.

    Args :
    - cutoff : Fréquence de coupure du filtre.
    - fs : Fréquence d'échantillonnage.
    - order : Ordre du filtre Butterworth.

    Retour :
    - b : Coefficients du numérateur du filtre.
    - a : Coefficients du dénominateur du filtre.
        """
        
        nyq = 0.5 * fs # Fréquence de Nyquist
        normal_cutoff = cutoff / nyq # Fréquence de coupure normalisée
        b, a = butter(order, normal_cutoff, btype='low', analog=False) # Calculer les coefficients du filtre
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        """
    Fonction pour appliquer un filtre passe-bas Butterworth à des données.

    Args :
    - data : Données à filtrer.
    - cutoff : Fréquence de coupure du filtre.
    - fs : Fréquence d'échantillonnage.
    - order : Ordre du filtre Butterworth.

    Retour :
    - y : Données filtrées.
        """
        
        
        b, a = self.butter_lowpass(cutoff, fs, order=order) # Calculer les coefficients du filtre
        y = lfilter(b, a, data) # Appliquer le filtre
        return y

    def reechantillonnage_fc_coupure_et_association_labels_et_data(self, cutoff_freq=20, target_freq=50):
        """
    Fonction pour rééchantillonner les données à une fréquence de 50Hz, auxquelles appliquer un filtre passe-bas Butterworth de 20 Hz
    et finir par associer les labels aux données.

    Args :
    - cutoff_freq : Fréquence de coupure du filtre.
    - target_freq : Fréquence cible après rééchantillonnage.

    Explications :
    - labels : Liste des noms des étiquettes.
    - original_freq : Fréquence d'échantillonnage originale.
    - data : Données analogiques.
    - nb_frame : Nombre de frames.
    - nb_samples_target : Nombre d'échantillons cible après rééchantillonnage.
    - resampled_times : Temps rééchantillonné.
    - resampled_data : Données rééchantillonnées.
    - fusion_label_data : Dictionnaire associant chaque étiquette à ses données.

    Retour :
    - Aucun, mais met à jour les attributs resampled_times et filtered_fusion_label_data de l'instance.
        """
        
        labels = self.c3d['parameters']['ANALOG']['LABELS']['value']
        original_freq = self.c3d['parameters']['ANALOG']['RATE']['value'][0]
        data = self.c3d['data']['analogs']
        nb_frame = len(data[0][0])
        nb_samples_target = int(nb_frame * (target_freq / original_freq)) # Nombre d'échantillons cible
        self.resampled_times = np.linspace(0., (nb_frame / original_freq), num=nb_samples_target) # Temps rééchantillonné
        resampled_data = np.zeros((len(labels), nb_samples_target)) # Initialiser un tableau pour stocker les données rééchantillonnées
        for i in range(len(labels)):
            resampled_data[i, :] = resample(data[0][i, :], nb_samples_target) 

        fusion_label_data = {}
        for i, label in enumerate(labels):
            fusion_label_data[label] = resampled_data[i, :] # Associer chaque étiquette à son signal

        self.filtered_fusion_label_data = {}  # Initialiser un dictionnaire pour stocker les données filtrées

        for label, data in fusion_label_data.items():  # Itérer sur chaque étiquette et signal
            filtered_signal = self.butter_lowpass_filter(data, cutoff_freq, target_freq)  # Appliquer le filtre Butterworth
            self.filtered_fusion_label_data[label] = filtered_signal  # Stocker le signal filtré dans le dictionnaire de données filtrées

    def filtrer_labels(self):
        """
    Fonction pour filtrer les étiquettes et garder uniquement les données de GYRO et ACC.

    Explications :
    - labels_filtre : Liste des étiquettes filtrées.
    - labels_data_filtre : Dictionnaire associant chaque étiquette à ses données filtrées.

    Retour :
    - Aucun, mais met à jour les attributs labels_filtre et labels_data_filtre de l'instance.
        """
        
        
        self.labels_filtre = []
        self.labels_data_filtre = {} #avec uniquement les données de GYRO et ACC excluant MAG
        for label, valeurs in self.filtered_fusion_label_data.items(): # Itérer sur chaque étiquette et signal
            if 'ACC' in label or 'GYRO' in label: # Si l'étiquette contient 'ACC' ou 'GYRO'
                self.labels_data_filtre[label] = valeurs
                
                
    def modifier_label_pelvis(self):
        """
        Modifie les étiquettes pour le capteur 'pelvis', ajoutant 'Left_' devant 'Pelvis'.

        Args:
            labels_data_filtre (dict): Dictionnaire contenant les étiquettes et les données associées.

        Returns:
            dict: Un dictionnaire avec les étiquettes modifiées.
        """
        self.labels_data_filtre_modifies = {}
        for key, value in self.labels_data_filtre.items():
            parts = key.split('_')
            if parts[0].lower() == "pelvis":  # Vérifier si l'étiquette commence par 'pelvis'
                # Reconstruire l'étiquette avec 'Left_' ajouté
                new_key = '_'.join(["Left"] + parts)
                self.labels_data_filtre_modifies[new_key] = value
            else:
                self.labels_data_filtre_modifies[key] = value

        return self.labels_data_filtre_modifies
    

    def calcul_norme(self):
        """
        Cette fonction calcule les normes des données filtrées.

        Args:
            labels_data_filtre (dict): Dictionnaire contenant les étiquettes et les données associées, filtrées.

        Returns:
            dict: Un dictionnaire contenant les normes calculées.
        """
        self.normes = {}
        traite = set() # Créer un ensemble vide pour stocker les capteurs, les côtés et les mesures déjà traités
        for key, value in self.labels_data_filtre_modifies.items():
            parts = key.split('_') # Séparer l'étiquette en parties
            sensor = parts[1] # Récupérer le capteur
            side = parts[0] # Récupérer le côté
            measure = parts[2] # Récupérer la mesure (GYRO ou ACC)
        
            if (side, sensor, measure) not in traite: # Si le capteur, le côté et la mesure n'ont pas déjà été traités
                traite.add((side, sensor, measure)) # Ajouter le capteur, le côté et la mesure à l'ensemble des éléments traités
            
                # if "ACC" in measure:
                    # Obtention des axes X, Y et Z
                axe_X = (self.labels_data_filtre_modifies[f'{side}_{sensor}_{measure}_X'])
                axe_Y = (self.labels_data_filtre_modifies[f'{side}_{sensor}_{measure}_Y'])
                axe_Z = (self.labels_data_filtre_modifies[f'{side}_{sensor}_{measure}_Z'])
            
                norme = np.sqrt(axe_X**2 + axe_Y**2 + axe_Z**2) # Calcul de la norme auquelle on soustrait 1 pour enlever la gravité
                nom_cle = f'{side}_{sensor}_{measure}_norme'
                self.normes[nom_cle] = norme

        return self.normes



    # def creer_structure_json(self, patient_id, date_de_naissance, medicaments):
    #     """
    #     Fonction pour créer une structure JSON à partir des données.

    #     Args :
    #     - patient_id : Identifiant du patient.
    #     - date_de_naissance : Date de naissance du patient.
    #     - medicaments : Liste des médicaments du patient.

    #     Explications :
    #     - json_data : Dictionnaire contenant les données dans le format JSON.

    #     Retour :
    #     - json_data : Dictionnaire JSON contenant les données du patient.
    #     """
    #     self.json_data = {
    #         "metadata": {
    #             "details du patient": {
    #                 "identifiant": patient_id,
    #                 "date de naissance": date_de_naissance,
    #                 "medicaments": medicaments
    #             },
    #             "temps": self.resampled_times.tolist()
    #         }
    #     }

    #     for key, value in self.labels_data_filtre_modifies.items():
    #         parts = key.split('_')
    #         sensor = parts[1]
    #         side = parts[0]
    #         measure = parts[2]
    #         axis = parts[3]

    #         if sensor not in self.json_data:
    #             self.json_data[sensor] = {}

    #         if side not in self.json_data[sensor]:
    #             self.json_data[sensor][side] = {}

    #         if measure not in self.json_data[sensor][side]:
    #             self.json_data[sensor][side][measure] = {}

    #         self.json_data[sensor][side][measure][axis] = value.tolist()

    #     for key in self.normes:
    #         parts = key.split('_')
    #         sensor = parts[1]
    #         side = parts[0]
    #         measure = parts[2]
    #         axis = parts[3]

    #         if sensor not in self.json_data:
    #             self.json_data[sensor] = {}

    #         if side not in self.json_data[sensor]:
    #             self.json_data[sensor][side] = {}

    #         if measure not in self.json_data[sensor][side]:
    #             self.json_data[sensor][side][measure] = {}

    #         self.json_data[sensor][side][measure][axis] = value.tolist()
    #         self.json_data[sensor][side][measure]["norme"] = self.normes[key].tolist()

    #     if "FOG_begin" in self.events_dict and "FOG_end" in self.events_dict: # Si les événements FOG sont présents
    #         self.json_data["FOG"] = {
    #             "debut": self.events_dict["FOG_begin"], 
    #             "fin": self.events_dict["FOG_end"]
    #         }
    #         del self.events_dict["FOG_begin"] # Supprimer les événements FOG du dictionnaire pour n'avoir que les évènements de parcours
    #         del self.events_dict["FOG_end"] # Supprimer les événements FOG du dictionnaire pour n'avoir que les évènements de parcours
    #     else: # Si les événements FOG ne sont pas présents
    #         self.json_data["FOG"] = {
    #             "debut": 0, 
    #             "fin": 0
    #         }

    #     self.json_data["parcours"] = self.events_dict

    #     return self.json_data
    
    
    
    def creer_structure_json(self):
        """
        Cette fonction crée une structure JSON à partir des données filtrées et d'autres informations.

        Args:
            labels_data_filtre (dict): Dictionnaire contenant les étiquettes et les données associées, filtrées.
            patient_id (int): Identifiant du patient.
            date_de_naissance (str): Date de naissance du patient.
            medicaments (str): Médicaments pris par le patient.
            resampled_times (ndarray): Temps rééchantillonné.
            events_dict (dict): Dictionnaire contenant les événements et leur temps correspondant.
            normes (dict): Dictionnaire contenant les normes calculées.

        Returns:
            dict: Structure JSON contenant les données et les métadonnées.
        """
        parts = self.file_path.split('/')
        filename = parts[-1].split('_')
        patient_id = '_'.join(filename[:3])
        date_de_naissance = filename[2]
        medicaments = filename[3]
        condition = filename[4].split(' ')[0]  # Nettoie la partie condition pour enlever les espaces et parenthèses
        #     # Extraction du numéro de la vidéo en nettoyant la chaîne
            
        # video_part = filename[4]  # Ajustez l'indice selon votre structure de nom de fichier
        # video = video_part.split(' ')[1]  # Séparer sur l'espace
        # video = video.replace('(', '').replace(').c3d', '')  # Enlever les parenthèses et l'extension
        
        
        
        
        ###################################################################""
        self.json_data = {
            "metadata": {
                "details du patient": {
                    "identifiant": patient_id,
                    "date de naissance": date_de_naissance,
                    "medicaments": medicaments,
                    "condition": condition
                },
                "temps": self.resampled_times.tolist()
        }
    }
    
        for key, value in self.labels_data_filtre_modifies.items():
            parts = key.split('_')
            sensor = parts[1]
            side = parts[0]
            measure = parts[2]
            axis = parts[3]
        
            if sensor not in self.json_data:
                self.json_data[sensor] = {}

            if side not in self.json_data[sensor]:
                self.json_data[sensor][side] = {}

            if measure not in self.json_data[sensor][side]:
                self.json_data[sensor][side][measure] = {}

            if axis not in self.json_data[sensor][side][measure]:
                self.json_data[sensor][side][measure][axis] = value.tolist()
        
        for key in self.normes:
            parts = key.split('_')
            sensor = parts[1]
            side = parts[0]
            measure = parts[2]
            axis = parts[3]

            if sensor not in self.json_data:
                self.json_data[sensor] = {}

            if side not in self.json_data[sensor]:
                self.json_data[sensor][side] = {}

            if measure not in self.json_data[sensor][side]:
                self.json_data[sensor][side][measure] = {}

            # Insérer la norme au même niveau d'indentation que l'axe
            self.json_data[sensor][side][measure]["norme"] = self.normes[key].tolist()
    
            # # Ajouter les événements FOG
            # json_data["FOG"] = {
            #     "Debut": events_dict["FOG_begin"],
            #     "Fin": events_dict["FOG_end"]
            # }
    
            # # Ajouter tous les évènements sauf FOG
            # del events_dict["FOG_begin"]
            # del events_dict["FOG_end"]
            
        if "FOG_begin" in self.events_dict and "FOG_end" in self.events_dict:
            self.json_data["FOG"] = {
                "debut": self.events_dict["FOG_begin"], 
                "fin": self.events_dict["FOG_end"]
            }
            del self.events_dict["FOG_begin"]
            del self.events_dict["FOG_end"]
        else:
            self.json_data["FOG"] = {
                "debut": [0],
                "fin": [0]
            }
    
        self.json_data["parcours"] = self.events_dict
    
        return self.json_data

    

    def creation_json_grace_c3d(self):#, output_path):
        """
        Fonction principale pour créer un fichier JSON à partir des données C3D.

        Args :
        - patient_id : Identifiant du patient.
        - date_de_naissance : Date de naissance du patient.
        - medicaments : Liste des médicaments du patient.

        Explications :
        - json_data : Dictionnaire JSON contenant les données de toutes les centrales inertielles portées
        par le patient.

        Retour :
        - json_data : Dictionnaire JSON contenant les données du patient.
        """
        
        self.recuperer_evenements() # Récupérer les événements
        self.reechantillonnage_fc_coupure_et_association_labels_et_data() # Rééchantillonner, filtrer et associer les données
        self.filtrer_labels() # Filtrer les étiquettes pour ne garder que les données de GYRO et ACC
        self.modifier_label_pelvis() # Modifier les étiquettes pour le capteur 'pelvis'
        self.calcul_norme() # Calculer les normes
        self.json_data = self.creer_structure_json() # Créer la structure JSON
        #with open(output_path, 'w') as outfile: # Ouvrir le fichier de sortie
        return self.json_data
            #json.dump(self.json_data, outfile, indent=4) # Écrire les données dans le fichier de sortie

################### Fin Création JSON ###################

################### Création Visuelle pour observer les FOG ###################
    def plot_data_FOG(self, muscle, side, sensor_type, axis):
        """
        Cette fonction permet de comparer les événements de FOG renseignés entre deux neurologues sur un signal donné.
        Args:
            time (array_like): Vecteur de temps.
            signal (array_like): Signal à tracer.
            fog_events_1 (dict): Dictionnaire contenant les instants des événements FOG évalués par le premier neurologue.
            fog_events_2 (dict): Dictionnaire contenant les instants des événements FOG pour le deuxième neurologue.
        """
        def plot_events_vertical_lines(events, color, linestyle, label):
            if isinstance(events, list):  # Vérifier si events est une liste
                for event in events:
                    plt.axvline(x=event, color=color, linestyle=linestyle, label=label)

        events_1_begin = self.json_data["FOG"].get("debut", [])
        events_1_end = self.json_data["FOG"].get("fin", [])

        data_to_plot = self.json_data[muscle][side][sensor_type][axis]
        plt.figure(figsize=(12, 6))
        plt.plot(self.resampled_times, data_to_plot)
        title = f"{muscle} - {side} - {sensor_type} - {axis}"

        plot_events_vertical_lines(events_1_begin, 'green', '--', f'FOG_begin')
        plot_events_vertical_lines(events_1_end, 'red', '--', f'FOG_end')

        plt.xlabel('Temps (s)')
        plt.ylabel('')
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for i, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handles[i])
        plt.legend(unique_handles, unique_labels)
        plt.tight_layout()
        plt.show()

################### Fin Création Visuelle pour observer les FOG ###################


################### Conserver les données entres les évènements START et END ###################
    def extract_data_interval(self):
        """
        Extrait un intervalle de données à partir du moment où le patient se lève jusqu'au dernier moment debout.

        Parameters:
        - data (dict): Un dictionnaire contenant les données des capteurs.

        Returns:
        - data_interval (dict): Un dictionnaire contenant l'intervalle de données extrait.
        """
        # Extraire les temps de début et de fin du parcours
        start_time = self.json_data["parcours"]["START"][0] # Extraire le temps de début du parcours
        end_time = self.json_data["parcours"]["END"][0] # Extraire le temps de fin du parcours
        epsilon=0.01 # Marge d'erreur

        # Trouver les indices correspondants dans le vecteur de temps pour le début du parcours
        for i, time in enumerate(self.resampled_times): # Pour chaque temps dans le vecteur de temps
            if abs(time - start_time) < epsilon:  # Vérifier si la différence est inférieure à la marge d'erreur
                start_index = i 
            if abs(time - end_time) < epsilon:
                end_index = i

        # Extraire les données d'axes pour la plage de temps START à END
        self.data_interval = {}
        for sensor, sensor_data in self.json_data.items():
            if sensor not in ["metadata", "parcours", "FOG"]:
                self.data_interval[sensor] = {}
                for side, side_data in sensor_data.items():
                    self.data_interval[sensor][side] = {}
                    for measure, measure_data in side_data.items():
                        self.data_interval[sensor][side][measure] = {}
                        for axis, axis_data in measure_data.items():
                            self.data_interval[sensor][side][measure][axis] = axis_data[start_index:end_index+1]

        # Copier les données de "metadata", "Parcours" et "FOG"
        self.data_interval["metadata"] = self.json_data["metadata"]
        self.data_interval["parcours"] = self.json_data["parcours"]
        self.data_interval["FOG"] = self.json_data["FOG"]

        # Extraire la plage de temps START à END pour la liste de temps dans metadata
        metadata_temps_interval = self.resampled_times[start_index:end_index+1] # Extraire la plage de temps START à END

        # Ajouter la plage de temps interval à metadata
        self.data_interval["metadata"]["temps"] = metadata_temps_interval # Ajouter la plage de temps interval à metadata
        return self.data_interval
################### Fin Conserver les données entres les évènements START et END ###################


################### Création Visuelle pour observer les FOG avec START et END ###################
    def plot_data_FOG_start_end(self, muscle, side, sensor_type, axis):
        """
        Cette fonction permet de comparer les événements de FOG renseignés entre deux neurologues sur un signal donné.
        Args:
            time (array_like): Vecteur de temps.
            signal (array_like): Signal à tracer.
            fog_events_1 (dict): Dictionnaire contenant les instants des événements FOG évalués par le premier neurologue.
            fog_events_2 (dict): Dictionnaire contenant les instants des événements FOG pour le deuxième neurologue.
        """
        def plot_events_vertical_lines(events, color, linestyle, label):
            if isinstance(events, list):  # Vérifier si events est une liste
                for event in events:
                    plt.axvline(x=event, color=color, linestyle=linestyle, label=label)

        events_1_begin = self.data_interval["FOG"].get("debut", [])
        events_1_end = self.data_interval["FOG"].get("fin", [])

        data_to_plot = self.data_interval[muscle][side][sensor_type][axis]
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_interval["metadata"]["temps"], data_to_plot)
        title = f"{muscle} - {side} - {sensor_type} - {axis}"

        plot_events_vertical_lines(events_1_begin, 'green', '--', f'FOG_begin')
        plot_events_vertical_lines(events_1_end, 'red', '--', f'FOG_end')

        plt.xlabel('Temps (s)')
        plt.ylabel('')
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for i, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handles[i])
        plt.legend(unique_handles, unique_labels)
        plt.tight_layout()
        plt.show()

################### Fin Création Visuelle pour observer les FOG avec START et END ###################

################### Normaliser les données entre START et END ###################
    def normalize_data(self):
        """
        Normalise les données des capteurs entre le début de la phase debout (START)
        et la fin de la phase debout (END). Ainsi, on ne prend pas les données, lorsque le patient est assis
        ou en transition debout/assis ou assis/debout.

        Parameters:
        - data (dict): Un dictionnaire contenant les données des capteurs.

        Returns:
        - normalized_data (dict): Un dictionnaire contenant les données normalisées.
        """
        self.normalized_data = {}
        for sensor, sensor_data in self.data_interval.items():
            if sensor not in ["metadata", "parcours", "FOG"]:
                self.normalized_data[sensor] = {}
                for side, side_data in sensor_data.items():
                    self.normalized_data[sensor][side] = {}
                    for measure, measure_data in side_data.items():
                        self.normalized_data[sensor][side][measure] = {}
                        for axis, axis_data in measure_data.items():
                            # Calculer la moyenne, le maximum et le minimum des données
                            mean = np.mean(axis_data)
                            max_val = np.max(axis_data)
                            min_val = np.min(axis_data)
                            # Appliquer la normalisation sur tous les axes X,Y,Z et la norme
                            normalized_axis_data = (axis_data - mean) / (max_val - min_val)
                            self.normalized_data[sensor][side][measure][axis] = normalized_axis_data
    
        # Copier les données de "metadata", "Parcours" et "FOG"
        self.normalized_data["metadata"] = self.data_interval["metadata"]
        self.normalized_data["parcours"] = self.data_interval["parcours"]
        self.normalized_data["FOG"] = self.data_interval["FOG"]
        return self.normalized_data
################### Fin Normaliser les données entre START et END ###################


################### Découpage en fenêtres ###################

    def decoupage_en_fenetres(self):
        """
    Fonction pour découper les données en fenêtres d'un certains temps avec un chevauchement de 80%.

    Explications :
    - fenetres_data : Dictionnaire contenant les données découpées en fenêtres.
    - infos_fenetres : Dictionnaire contenant des informations sur chaque fenêtre découpée.

    Retour :
    - fenetres_data : Dictionnaire contenant les données découpées en fenêtres.
    - infos_fenetres : Dictionnaire contenant des informations sur chaque fenêtre découpée.
        """
        
        self.fenetres_data = {} # Initialiser un dictionnaire pour stocker les données découpées en fenêtres
        self.infos_fenetres = {} # Initialiser un dictionnaire pour stocker des informations sur chaque fenêtre découpée

        for sensor, sensor_data in self.normalized_data.items():
            if sensor not in ["metadata", "parcours", "FOG"]: # Si le capteur n'est pas "metadata", "Parcours" ou "FOG"
                self.fenetres_data[sensor] = {}
                self.infos_fenetres[sensor] = {}

                for side, side_data in sensor_data.items():
                    self.fenetres_data[sensor][side] = {}
                    self.infos_fenetres[sensor][side] = {}

                    for measure, measure_data in side_data.items():
                        self.fenetres_data[sensor][side][measure] = {}
                        self.infos_fenetres[sensor][side][measure] = {}

                        for axis, axis_data in measure_data.items():
                            taille_signal = len(axis_data) # Taille du signal
                            taille_fenetre_echantillons = int(self.taille_fenetre * self.taux_echantillonnage) # Taille de la fenêtre en échantillons
                            decalage_fenetre = int(self.decalage * taille_fenetre_echantillons) # Décalage de la fenêtre

                            fenetres = []
                            debut = 0
                            fin = taille_fenetre_echantillons
                            nb_fenetres = 0
                            
                            # Découpage des données en fenêtres avec décalage
                            while fin <= taille_signal: # Tant que la fin de la fenêtre est inférieure à la taille du signal
                                fenetre = axis_data[debut:fin] 
                                fenetres.append(fenetre)

                                debut = debut + decalage_fenetre 
                                fin = fin + decalage_fenetre
                                nb_fenetres += 1

                            # Ajout de la dernière fenêtre si la taille ne correspond pas exactement
                            if debut < taille_signal:
                                fenetre = axis_data[debut:]
                                fenetres.append(fenetre)

                            self.fenetres_data[sensor][side][measure][axis] = fenetres
                            self.infos_fenetres[sensor][side][measure][axis] = {
                                "nombre_fenetres": nb_fenetres,
                                "taille_fenetre": taille_fenetre_echantillons,
                                "decalage_fenetre": decalage_fenetre
                            }

        # Traitement des nouvelles données de temps
        temps = self.normalized_data["metadata"]["temps"]
        taille_signal_temps = len(temps)
        taille_fenetre_temps = int(self.taille_fenetre * self.taux_echantillonnage)
        decalage_fenetre_temps = int(self.decalage * taille_fenetre_temps)

        fenetres_temps = []
        debut_temps = 0
        fin_temps = taille_fenetre_temps

        while fin_temps <= taille_signal_temps:
            fenetre_temps = temps[debut_temps:fin_temps]
            fenetres_temps.append(fenetre_temps)

            debut_temps += decalage_fenetre_temps
            fin_temps += decalage_fenetre_temps

        if debut_temps < taille_signal_temps:
            fenetre_temps = temps[debut_temps:]
            fenetres_temps.append(fenetre_temps)

        # Copie des données de "metadata", "Parcours" et "FOG"
        self.fenetres_data["metadata"] = self.normalized_data["metadata"]
        self.fenetres_data["parcours"] = self.normalized_data["parcours"]
        self.fenetres_data["FOG"] = self.normalized_data["FOG"]
    
        # Remplacement des anciennes données de temps par les nouvelles
        self.fenetres_data["metadata"]["temps"] = fenetres_temps
        return self.fenetres_data

################### Fin du Découpage en fenêtres ###################



################### Obtention des labels de fenêtres par rapport au temps ###################
    def label_fenetre(self):
        """
    Fonction pour étiqueter chaque fenêtre en fonction des événements de début et de fin de FOG.
    Ainsi, si une fenêtre contient un événement de début de FOG, elle sera étiquetée comme "transitionFog" ou "fog" si elle contient un événement de fin de FOG.
    Ensuite, si une fenêtre contient un événement de début de FOG et un événement de fin de FOG, elle sera étiquetée comme "fog".
    De plus, si une fenêtre contient un évènement fin de FOG et que le FOG est inférieur à 50% de la longueur de la fenêtre, elle sera étiquetée comme "transitionNoFog".
    Enfin, si une fenêtre ne contient aucun événement de FOG, elle sera étiquetée comme "noFog".
    
    Explications :
    - temps : Liste des temps de début et de fin des fenêtres.
    - debuts_fog : Liste des temps de début de FOG.
    - fins_fog : Liste des temps de fin de FOG.

    Retour :
    - fenetres_data : Dictionnaire contenant les données découpées en fenêtres avec les labels associés.
        """

        temps = self.fenetres_data["metadata"]["temps"]
        debuts_fog = self.fenetres_data["FOG"]["debut"]
        fins_fog = self.fenetres_data["FOG"]["fin"]
        # debuts_prefog= debuts_fog
        # if isinstance(debuts_fog, int):
        #         debuts_fog = [debuts_fog]  # Transforme l'entier en liste contenant cet entier
        # self.debuts_prefog = [max(0, x - self.temps_prefog) for x in debuts_fog]
        
        # Transformation des entiers en listes s'ils ne sont pas déjà sous forme de liste
        if isinstance(fins_fog, int):
            fins_fog = [fins_fog]  # Transforme l'entier en liste contenant cet entier        
    
        # on stock les données d'évènement dans un dataframe ordonner en fonction du temps
        events=pd.DataFrame({'temps': debuts_fog + fins_fog, #+ self.debuts_prefog, 
                    'events': ["debut_fog"]*len(debuts_fog) + ["fin_fog"]*len(fins_fog)}).sort_values('temps').reset_index(drop=True)

        # On récupère si il y a un évènement de FOG ou plusieurs de présent de la FOG
        statuses = []
        status = "noFog"

        for window in temps:
            w_start = window[0] # premier terme de la fenêtre
            w_end = window[-1] # dernier terme de la fenêtre
            window_events = []  # Liste pour stocker les événements de la fenêtre
            time=[]
            time_pourcent = 1 
            
            # Parcours des événements pour vérifier s'ils se trouvent dans la fenêtre
            for _, row in events.iterrows():
                if row['temps'] >= w_start and row['temps'] <= w_end: #si le temps correspondant à l'évènement se trouve entre début et fin de la fenêtre
                    window_events.append(row['events'])  # Ajouter l'événement à la liste
                    if row['events']== "fin_fog": # si l'évènement est fin_fog
                        time.append(row["temps"])
                        
                        
            if len(window_events)==1 and "fin_fog" in window_events: #si on oa une liste avec uniquement fin_fog
                time_array = np.arange(w_start,w_end,1/50) # on crée un vecteur de temps pour la fenêtre
                time_pourcent = np.sum(time_array<=time)/100 # on calcule le pourcentage de temps de FOG dans la fenêtre
    
    
            if not window_events:  # Si la liste est vide
                window_events = [None]  # Remplir avec None
    
            # if status == "NoFog" and "preFog" in window_events:
            #     status = "transitionPreFog"
        
            # elif status == "transitionPreFog" and None in window_events:
            #     status = "preFog"
            
            if status == "noFog" and "debut_fog" in window_events: # si la fenêtre contient debut_fog et son statut est NoFog
                status = "transitionFog"
            
            elif status == "transitionFog" and None in window_events: #si le FOG est suffisement long pour ne pas rencontrer d'évènement après debut_Fog alors :
                status = "fog"
                
            elif status == "transitionFog" and ("debut_fog" in window_events and "fin_fog" in window_events): #si il est petit alors la fenêtre peut comporter l'évènement de début et de fin
                status = "fog"
            
            elif status =="fog" and ("fin_fog" in window_events and "debut_fog" in window_events): # dans le cas où des FOG sont succints, donc c'est à dire quand la fenêtre comporte fin_fog et debut du fog suivant alors :  
                status= "transitionFog"
        
            elif status == "fog" and "fin_fog" in window_events and time_pourcent <= 0.5: # si on a un FOG inférieur à 50% de la longueur de fenêtre, alors : 
            # on s'en fiche de faire cette opération avant, car dans tous les cas on considère FOG lorsqu'il y a deux évènements dans la fenêtre et on prend pour cible transitionFog,donc que ce soit FOG ou transition ce sera dans cible. 
                status = "transitionNoFog"
            
            elif status == "transitionNoFog" and None in window_events:
                status = "noFog"
        
            statuses.append(status)  # Ajouter le statut à la liste des statuts
        
        # on associe les labels de fenêtre dans notre data
        self.fenetres_data["labels_fenetres"] = statuses
        return self.fenetres_data
        
################### Fin de l'obtention des labels de fenêtres par rapport au temps ###################

################### Debut association labels fenetres aux datas ###################
    def association_label_fenetre_data(self):
        """
    Grâce au temps, nous avons obtenus labels de chacune des fenêtres. Il suffit juste 
    d'associer les labels aux données de chaque fenêtre. Cependant, pour extraire les données associées à chaque labels,
    il est nécessaire de citer des labels uniquent et nous n'avons pas la possibilité d'avoir un Dataframe avec tous les labels.

    Retour :
    - mix_label_fenetre_data : Data.frame contenant les données de chaque fenêtre associées à leur label.
        """
        self.mix_label_fenetre_data = {}
        for sensor, sensor_data in self.fenetres_data.items():
            if sensor not in ["metadata", "parcours", "FOG", "labels_fenetres"]:
                self.mix_label_fenetre_data[sensor] = {}
                for side, side_data in sensor_data.items():
                    self.mix_label_fenetre_data[sensor][side] = {}

                    for measure, measure_data in side_data.items():
                        self.mix_label_fenetre_data[sensor][side][measure] = {}
                    
                        for axis, axis_data in measure_data.items():
                            self.mix_label_fenetre_data[sensor][side][measure][axis] = {}
                            presentWin=np.unique(self.fenetres_data["labels_fenetres"])
                            for label in presentWin:
                                #data_list = axis_data
                                data_frame_axis_data= pd.DataFrame(axis_data)
                                self.mix_label_fenetre_data[sensor][side][measure][axis][label]=data_frame_axis_data[[x==label for x in self.fenetres_data["labels_fenetres"]]]
    
        # Copie des données de "metadata", "Parcours" et "FOG"
        self.mix_label_fenetre_data["metadata"] = self.fenetres_data["metadata"]
        self.mix_label_fenetre_data["metadata"]["temps"] = pd.DataFrame(self.mix_label_fenetre_data["metadata"]["temps"])
        self.mix_label_fenetre_data["parcours"] = self.fenetres_data["parcours"]
        self.mix_label_fenetre_data["FOG"] = self.fenetres_data["FOG"]

        self.mix_label_fenetre_data["FOG"] = {
            "debut": self.fenetres_data["FOG"]["debut"],
            "fin": self.fenetres_data["FOG"]["fin"],
            #"preFog": self.debuts_prefog
    }
        return self.mix_label_fenetre_data
################### Fin association labels fenetres aux datas ###################



################### Debut concaténation des labels et des données ###################
    def concat_label_fenetre_data(self):
        """
    Fonction pour concaténer les données de chaque label de fenêtre. Pour chaque capteur, côté, muscle et axe, avec une colonne supplémentaire pour identifier l'étiquette et qui
    permets d'avoir tous les labels dans un seul Dataframe et ne pas avoir à citer dans le JSON
    [muscle][côté][capteur][axe][label], mais uniquement [muscle][côté][capteur][axe]
    
    Retour :
    - concat_data : Dataframe contenant les données concaténées de chaque fenêtre pour chaque capteur, côté, muscle et axe.
        """
        # Initialisez un dictionnaire pour stocker les données combinées
        self.concat_data = {}

        # Bouclez à travers les données pour chaque muscle, côté, capteur, axe
        for muscle, muscle_data in self.mix_label_fenetre_data.items():
            if muscle not in ["metadata", "parcours", "FOG"]:
                for side, side_data in muscle_data.items():
                    for sensor, sensor_data in side_data.items():
                        for axis, axis_data in sensor_data.items():
                        # Initialisez une liste pour stocker les DataFrames de chaque étiquette
                            dfs = []
                            # Bouclez à travers les étiquettes disponibles
                            for label, label_data in axis_data.items():
                                # Ajoutez une colonne 'label' pour identifier l'étiquette
                                label_data['label'] = label
                                # Ajoutez le DataFrame actuel à la liste
                                dfs.append(label_data)
                            # Concaténez les DataFrames de chaque étiquette (ou un DataFrame vide si aucune étiquette disponible)
                            combined_df = pd.concat(dfs) if dfs else pd.DataFrame()
                            # Réorganisez les colonnes pour mettre 'label' en première position
                            combined_df = combined_df[['label'] + [col for col in combined_df.columns if col != 'label']]
                            # Stockez le DataFrame combiné dans le dictionnaire
                            if muscle not in self.concat_data:
                                self.concat_data[muscle] = {}
                            if side not in self.concat_data[muscle]:
                                self.concat_data[muscle][side] = {}
                            if sensor not in self.concat_data[muscle][side]:
                                self.concat_data[muscle][side][sensor] = {}
                            self.concat_data[muscle][side][sensor][axis] = combined_df.sort_index()  # Réorganiser les lignes dans l'ordre croissant des index
        
        # Copie des données de "metadata", "Parcours" et "FOG"
        self.concat_data["metadata"] = self.mix_label_fenetre_data["metadata"]
        self.concat_data["parcours"] = self.mix_label_fenetre_data["parcours"]
        self.concat_data["FOG"] = self.mix_label_fenetre_data["FOG"]

        return self.concat_data
################### Fin concaténation des labels et des données ###################
    
    def plot_data_FOG_start_end_final(self, muscle, side, sensor_type,axis, window_index):
        """
        Cette fonction permet de comparer les événements de FOG renseignés entre deux neurologues sur un signal donné.
        Args:
            time (array_like): Vecteur de temps.
            signal (array_like): Signal à tracer.
            fog_events_1 (dict): Dictionnaire contenant les instants des événements FOG évalués par le premier neurologue.
            fog_events_2 (dict): Dictionnaire contenant les instants des événements FOG pour le deuxième neurologue.
        """
        def plot_events_vertical_lines(events, color, linestyle, label):
            if isinstance(events, list):  # Vérifier si events est une liste
                for event in events:
                    plt.axvline(x=event, color=color, linestyle=linestyle, label=label)

        events_1_begin = self.concat_data["FOG"].get("debut", [])
        events_1_end = self.concat_data["FOG"].get("fin", [])

        data_to_plot = self.concat_data[muscle][side][sensor_type][axis][window_index]
        data_to_plot = data_to_plot.drop(columns=["label"])
        plt.figure(figsize=(12, 6))
        plt.plot(self.concat_data["metadata"]["temps"][window_index], data_to_plot)
        title = f"{muscle} - {side} - {sensor_type} - {axis} - {window_index}"

        #plot_events_vertical_lines(events_1_begin, 'green', '--', f'FOG_begin')
        #plot_events_vertical_lines(events_1_end, 'red', '--', f'FOG_end')

        plt.xlabel('Temps (s)')
        plt.ylabel('')
        plt.title(title)
        # handles, labels = plt.gca().get_legend_handles_labels()
        # unique_labels = []
        # unique_handles = []
        # for i, label in enumerate(labels):
        #     if label not in unique_labels:
        #         unique_labels.append(label)
        #         unique_handles.append(handles[i])
        # plt.legend(unique_handles, unique_labels)
        # plt.tight_layout()
        # plt.show()


class Statistics:
    def __init__(self, file_path, concat_data):
        self.concat_data = concat_data
        self.taille_fenetre = 2 # Taille de la fenêtre en secondes
        self.file_path = file_path
        
    def stats(self):
        # On calcul la durée totale de l'enregistrement
        first_time= self.concat_data["metadata"]["temps"].iloc[0,0] # on récupère la première donnée
        
        if np.isnan(self.concat_data["metadata"]["temps"].iloc[-1,-1]):
            first_na = np.where(np.isnan(self.concat_data["metadata"]["temps"]))[-1][0]
            last_time = self.concat_data["metadata"]["temps"].iloc[-1,(first_na-1)]
        else:
            last_time = self.concat_data["metadata"]["temps"].iloc[-1,-1]
        
        
        
        temps_total = last_time - first_time

        # On calcul le nombre de FOG et le pourcentage de FOG sur la totalité de l'enregistrement
        if  self.concat_data["FOG"]["debut"] == [0] : #si on a pas de FOG, nous sommes censé avoir une liste avec un seul élément qui est 0
            nb_fog = 0
        else :
            nb_fog = len(self.concat_data["FOG"]["debut"])

        temps_fog = sum(fin - debut for debut, fin in zip(self.concat_data["FOG"]["debut"], self.concat_data["FOG"]["fin"]))
        prct_fog = (temps_fog / temps_total) * 100

        
        nb_fenetre = len(self.concat_data["metadata"]["temps"])
        nb_colonne = len(self.concat_data["metadata"]["temps"].columns)
        
        # Extraction des informations du file_path
        parts = self.file_path.split('/')
        filename = parts[-1].split('_')
        identifiant = '_'.join(filename[:3])
        statut = filename[3]
        condition = filename[4]  # Nettoie la partie condition pour enlever les espaces et parenthèses
        video = filename[5].replace('.c3d', '')  # Ajustez l'indice selon votre structure de nom de fichier


        # Ajout des nouvelles colonnes
        tab_stat = pd.DataFrame({
            "Temps total de l'enregistrement": [temps_total],
            "Nombre de FOG": [nb_fog],
            "Temps total de Freezing": [temps_fog],
            "Pourcentage total de FOG": [prct_fog],
            "Temps de chaque fenêtre": [self.taille_fenetre],
            "Nombre de fenêtres": [nb_fenetre],
            "Longueur des fenêtres": [nb_colonne],
            "Statut": [statut],
            "Condition": [condition],
            "Vidéo": [video],
            "Identifiant": [identifiant]
        })
        
        # tab_stat = pd.DataFrame({"Temps total de l'enregistrement": [temps_total], 
        #                          "Nombre de FOG": [nb_fog],
        #                          "Temps total de Freezing" : [temps_fog],
        #                          "Pourcentage total de FOG": [prct_fog],
        #                          "Temps de chaque fenêtre" : [self.taille_fenetre], 
        #                          "nombre de fenêtres": [nb_fenetre], 
        #                          "longueur des fenêtres": [nb_colonne]})
        
        
        # tab_fog = pd.DataFrame({"Debut": self.concat_data["FOG"]["debut"], 
        #                         "Fin": self.concat_data["FOG"]["fin"], 
        #                         "Total" : temps_fog})
        return tab_stat
    
    



class ExtractionFeatures:
    def __init__(self, concat_data):
        ################### Initialisation des attributs pour création JSON ###################
        self.data = concat_data # Données provenant de la classe PreProcessing
        self.fs = 50 # Fréquence d'échantillonnage  
        self.fft_magnitudes = None
        self.frequencies = None
        self.label = None

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
                                self.label = data_moins_derniere_ligne_na["label"]
                            
                                # Vérifier si la colonne 'label' existe avant de la supprimer
                                if 'label' in data_moins_derniere_ligne_na.columns:
                                    data_moins_colonne_label = data_moins_derniere_ligne_na.drop(columns=["label"])
                                    # Mise à jour du DataFrame dans le dictionnaire
                                    measure_data[axis] = data_moins_colonne_label
                                    
        return self.data, self.label


    def transformation_domaine_frequentiel (self, axis_data):
        # Nombre de points de données par fenêtre
        n = axis_data.shape[1]  # ou 100 si c'est connu

        # Créer un tableau de fréquences
        frequences = fftfreq(n, d=1/self.fs)
        frequences = frequences[:n//2] # obligé de laisser la data en série, pour générer le graphique des spectres de magnitudes
        self.frequencies = fftfreq(n, d=1/self.fs)
        self.frequencies = self.frequencies[:n//2]

        # Transposer le tableau de fréquences pour le mettre en colonnes
        self.frequencies = self.frequencies.reshape((1, -1))



        # calculer la transformée de Fourier
        fft_result = fft(axis_data, axis = 1)
        self.fft_magnitudes = np.abs(fft_result)[:,:n//2] # Garder uniquement les valeurs positives, puisque d'après la symétrie de la FFT, les valeurs négatives sont les mêmes que les valeurs positives


        # # Créer un DataFrame pour stocker les magnitudes des fréquences
        self.fft_magnitudes = pd.DataFrame(self.fft_magnitudes)
        self.frequencies = pd.DataFrame(self.frequencies)

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



    # def skewness_band_freq (self):
    #     magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
    #     frequences = self.frequencies.values.flatten()


    #     # Définissons les bandes de fréquences spécifiées
    #     bandes_frequence = {
    #         'skewness': (0, 50),
    #         'skewness_0.04_0.68_Hz': (0.04, 0.68),
    #         'skewness_0.68_3_Hz': (0.68, 3),
    #         'skewness_3_8_Hz': (3, 8),
    #         'skewness_8_20_Hz': (8, 20),
    #         'skewness_0.1_8_Hz': (0.1, 8)
    #     }

    #     # Créons un DataFrame pour stocker les écarts types calculés pour chaque bande de fréquence et pour chaque ligne (fenêtre)
    #     skwenesss = pd.DataFrame()

    #     # Pour chaque bande de fréquence, filtrons les données et calculons l'écart type
    #     for nom_bande, (freq_min, freq_max) in bandes_frequence.items():
        
    #         # Identifions les colonnes correspondant à la bande de fréquence
    #         colonnes_bande = (frequences >= freq_min) & (frequences <= freq_max)
        
    #         # Filtrons les magnitudes pour cette bande de fréquence
    #         magnitudes_bande = magnitudes[:, colonnes_bande]
        
    #         # Calculons l'écart type pour cette bande de fréquence et ajoutons les résultats au DataFrame
    #         skwenesss[nom_bande] = skew(magnitudes_bande,axis=1)
            
    #     return skwenesss



    # def kurtosis_band_freq (self) :
    #     magnitudes = self.fft_magnitudes.values  # Convertissons le DataFrame en numpy array si ce n'est pas déjà le cas
    #     frequences = self.frequencies.values.flatten()
    #     # Définissons les bandes de fréquences spécifiées
    #     bandes_frequence = {
    #         'kurtosis': (0, 50),
    #         'kurtosis_0.04_0.68_Hz': (0.04, 0.68),
    #         'kurtosis_0.68_3_Hz': (0.68, 3),
    #         'kurtosis_3_8_Hz': (3, 8),
    #         'kurtosis_8_20_Hz': (8, 20),
    #         'kurtosis_0.1_8_Hz': (0.1, 8)
    #     }

    #     # Créons un DataFrame pour stocker les écarts types calculés pour chaque bande de fréquence et pour chaque ligne (fenêtre)
    #     kurtosiss = pd.DataFrame()

    #     # Pour chaque bande de fréquence, filtrons les données et calculons l'écart type
    #     for nom_bande, (freq_min, freq_max) in bandes_frequence.items():
        
    #         # Identifions les colonnes correspondant à la bande de fréquence
    #         colonnes_bande = (frequences >= freq_min) & (frequences <= freq_max)
        
    #         # Filtrons les magnitudes pour cette bande de fréquence
    #         magnitudes_bande = magnitudes[:, colonnes_bande]
        
    #         # Calculons l'écart type pour cette bande de fréquence et ajoutons les résultats au DataFrame
    #         kurtosiss[nom_bande] = kurtosis(magnitudes_bande,axis=1)
        
    #     return kurtosiss

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
        
        # Calcul de l'énergie pour chaque signal (chaque ligne)
        energy = np.sum(np.abs(magnitudes)**2 / len(magnitudes), axis=1)
        
        # Créer un DataFrame pour stocker les résultats
        df_energie = pd.DataFrame({'Energie_Frequentielle': energy})
            
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
                                entropie_spectrale = self.calcul_entropie_spectrale()
                                details_harmoniques = self.calcul_details_harmoniques()
                                ecart_types = self.ecart_type_borne()
                                freeze_index = self.calculer_freeze_index()
                                ratio_faible_puissance = self.ratio_faible_puissance_entre_0_2Hz()
                                # skewness = self.skewness_band_freq()
                                # kurtosis = self.kurtosis_band_freq()
                                locomotion_band_power = self.calcul_locomotion_band_power()
                                freeze_band_power = self.calcul_freeze_band_power()
                                band_power = self.calcul_band_power()
                                energie = self.calcul_energie()
                                

                                # Fusionner toutes les caractéristiques dans un seul DataFrame pour simplification
                                caract_features = pd.concat([features_temporelles,
                                                            entropie_spectrale,
                                                            details_harmoniques, 
                                                            ecart_types,
                                                            freeze_index,
                                                            ratio_faible_puissance,
                                                            # skewness,
                                                            # kurtosis,
                                                            locomotion_band_power,
                                                            freeze_band_power,
                                                            band_power,
                                                            energie], axis=1)

                                # Renommer les colonnes ici
                                caract_features.rename(columns={feature_name: f"{sensor}_{side}_{measure}_{axis}_{feature_name}" for feature_name in caract_features.columns}, inplace=True)
                                
                                data_collect.append(caract_features)

        # Concaténer toutes les données collectées en alignant les colonnes
        df_final = pd.concat(data_collect, axis=1)
        label_dataframe = pd.DataFrame(self.label)
        data_concat = pd.concat([df_final,label_dataframe],axis = 1)
        combined_df = data_concat[['label'] + [col for col in data_concat.columns if col != 'label']]
        # Arrondir toutes les valeurs numériques à quatre décimales
        combined_df = combined_df.round(4)

        return combined_df