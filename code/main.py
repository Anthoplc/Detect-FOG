import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import resample,butter, lfilter, freqz
import json
import ezc3d


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
        df = pd.DataFrame({'events': events, 'frames': temps_events[1]}) # Créer un dataframe avec les événements et les temps
        df.reset_index(drop=True, inplace=True) # Supprimer la colonne index
        self.events_dict = df.groupby('events')['frames'].apply(list).to_dict() # Créer un dictionnaire avec les événements et les temps associés

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
        
            elif "debut_fog" in window_events and "fin_fog" in window_events: #si il est petit alors la fenêtre peut comporter l'évènement de début et de fin
                status = "fog"
            
            elif status =="fog" and ("debut_fog" in window_events and "fin_fog" in window_events): # dans le cas où des FOG sont succints, donc c'est à dire quand la fenêtre comporte fin_fog et debut du fog suivant alors :  
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

        temps_fog = sum([fin - debut for debut, fin in zip(self.concat_data["FOG"]["debut"], self.concat_data["FOG"]["fin"])])
        prct_fog = (temps_fog / temps_total) * 100
        
        
        nb_fenetre = len(self.concat_data["metadata"]["temps"])
        nb_colonne = len(self.concat_data["metadata"]["temps"].columns)
        
        temps_fog = [fin - debut for debut, fin in zip(self.concat_data["FOG"]["debut"], self.concat_data["FOG"]["fin"])]


        # Extraction des informations du file_path
        parts = self.file_path.split('/')
        filename = parts[-1].split('_')
        identifiant = '_'.join(filename[:3])
        anniversaire = filename[2]
        statut = filename[3]
        condition = filename[4]  # Nettoie la partie condition pour enlever les espaces et parenthèses
        video = filename[5].replace('.c3d', '')  # Ajustez l'indice selon votre structure de nom de fichier

        # Calculs existants...
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
    
