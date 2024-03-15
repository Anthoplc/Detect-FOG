import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import resample,butter, lfilter, freqz
import json
import ezc3d

# class DetectFog:
#     def __init__(self, file_path):
#         self.__PreProcessing__ = PreProcessing(file_path)
#         self.__Statistics__ = Statistics(self.__PreProcessing__)

class PreProcessing:
    def __init__(self, file_path):
        
        ################### Initialisation des attributs pour création JSON ###################
        self.file_path = file_path # Chemin du fichier c3d
        self.c3d = ezc3d.c3d(file_path) # Lire le fichier c3d
        self.events_dict = None # Dictionnaire pour stocker les événements
        self.resampled_times = None # Temps rééchantillonné
        self.filtered_fusion_label_data = None # Données filtrées
        self.labels_data_filtre = None  # Données filtrées avec uniquement les données de GYRO et ACC excluant MAG
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

################### Création JSON ###################

    def recuperer_evenements(self):
        events = self.c3d['parameters']['EVENT']['LABELS']['value'] # Récupérer les événements
        temps_events = self.c3d['parameters']['EVENT']['TIMES']['value'] # Récupérer les temps des événements
        df = pd.DataFrame({'events': events, 'frames': temps_events[1]}) # Créer un dataframe avec les événements et les temps
        df.reset_index(drop=True, inplace=True) # Supprimer la colonne index
        self.events_dict = df.groupby('events')['frames'].apply(list).to_dict() # Créer un dictionnaire avec les événements et les temps associés

    def butter_lowpass(self, cutoff, fs, order=2):
        nyq = 0.5 * fs # Fréquence de Nyquist
        normal_cutoff = cutoff / nyq # Fréquence de coupure normalisée
        b, a = butter(order, normal_cutoff, btype='low', analog=False) # Calculer les coefficients du filtre
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        b, a = self.butter_lowpass(cutoff, fs, order=order) # Calculer les coefficients du filtre
        y = lfilter(b, a, data) # Appliquer le filtre
        return y

    def reechantillonnage_fc_coupure_et_association_labels_et_data(self, cutoff_freq=20, target_freq=50):
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
        self.labels_filtre = []
        self.labels_data_filtre = {} #avec uniquement les données de GYRO et ACC excluant MAG
        for label, valeurs in self.filtered_fusion_label_data.items(): # Itérer sur chaque étiquette et signal
            if 'ACC' in label or 'GYRO' in label: # Si l'étiquette contient 'ACC' ou 'GYRO'
                self.labels_data_filtre[label] = valeurs

    def calcul_norme(self):
        self.normes = {}
        traite = set()
        for key, value in self.labels_data_filtre.items(): 
            parts = key.split('_')  # Diviser l'étiquette en parties
            sensor = parts[0]  # Muscle
            side = parts[1] # Côté
            measure = parts[2] # ACC ou GYRO
            
            if (sensor, side, measure) not in traite: # Si le muscle, le côté et measure n'ont pas été traités
                traite.add((sensor, side, measure)) # Ajouter le muscle, le côté et measure à l'ensemble traite
                if "ACC" in measure:
                    axe_X = (self.labels_data_filtre[f'{sensor}_{side}_{measure}_X'])
                    axe_Y = (self.labels_data_filtre[f'{sensor}_{side}_{measure}_Y'])
                    axe_Z = (self.labels_data_filtre[f'{sensor}_{side}_{measure}_Z'])

                    norme = np.sqrt(axe_X**2 + axe_Y**2 + axe_Z**2) - 1 # Calculer la norme en soustrayant 1 pour enlever la gravité
                    nom_cle = f'{sensor}_{side}_{measure}_norme'  # Créer le nom de la clé
                    self.normes[nom_cle] = norme # Stocker la norme dans le dictionnaire normes
                else: # Si measure est GYRO
                    axe_X = self.labels_data_filtre[f'{sensor}_{side}_{measure}_X']
                    axe_Y = self.labels_data_filtre[f'{sensor}_{side}_{measure}_Y']
                    axe_Z = self.labels_data_filtre[f'{sensor}_{side}_{measure}_Z']

                    norme = np.sqrt(axe_X**2 + axe_Y**2 + axe_Z**2)
                    nom_cle = f'{sensor}_{side}_{measure}_norme'
                    self.normes[nom_cle] = norme

    def creer_structure_json(self, patient_id, date_de_naissance, medicaments):
        self.json_data = {
            "metadata": {
                "details du patient": {
                    "identifiant": patient_id,
                    "date de naissance": date_de_naissance,
                    "medicaments": medicaments
                },
                "temps": self.resampled_times.tolist()
            }
        }

        for key, value in self.labels_data_filtre.items():
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

            self.json_data[sensor][side][measure][axis] = value.tolist()
            self.json_data[sensor][side][measure]["norme"] = self.normes[key].tolist()

        if "FOG_begin" in self.events_dict and "FOG_end" in self.events_dict: # Si les événements FOG sont présents
            self.json_data["FOG"] = {
                "debut": self.events_dict["FOG_begin"], 
                "fin": self.events_dict["FOG_end"]
            }
            del self.events_dict["FOG_begin"] # Supprimer les événements FOG du dictionnaire pour n'avoir que les évènements de parcours
            del self.events_dict["FOG_end"] # Supprimer les événements FOG du dictionnaire pour n'avoir que les évènements de parcours
        else: # Si les événements FOG ne sont pas présents
            self.json_data["FOG"] = {
                "debut": 0, 
                "fin": 0
            }

        self.json_data["parcours"] = self.events_dict

        return self.json_data

    def creation_json_grace_c3d(self, patient_id, date_de_naissance, medicaments):#, output_path):
        self.recuperer_evenements() # Récupérer les événements
        self.reechantillonnage_fc_coupure_et_association_labels_et_data() # Rééchantillonner, filtrer et associer les données
        self.filtrer_labels() # Filtrer les étiquettes pour ne garder que les données de GYRO et ACC
        self.calcul_norme() # Calculer les normes
        self.json_data = self.creer_structure_json(patient_id, date_de_naissance, medicaments) # Créer la structure JSON
        #with open(output_path, 'w') as outfile: # Ouvrir le fichier de sortie
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
        Normalise les données des capteurs.

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

################### Fin Normaliser les données entre START et END ###################


################### Découpage en fenêtres ###################

    def decoupage_en_fenetres(self):
        self.fenetres_data = {}
        self.infos_fenetres = {}

        for sensor, sensor_data in self.normalized_data.items():
            if sensor not in ["metadata", "parcours", "FOG"]:
                self.fenetres_data[sensor] = {}
                self.infos_fenetres[sensor] = {}

                for side, side_data in sensor_data.items():
                    self.fenetres_data[sensor][side] = {}
                    self.infos_fenetres[sensor][side] = {}

                    for measure, measure_data in side_data.items():
                        self.fenetres_data[sensor][side][measure] = {}
                        self.infos_fenetres[sensor][side][measure] = {}

                        for axis, axis_data in measure_data.items():
                            taille_signal = len(axis_data)
                            taille_fenetre_echantillons = int(self.taille_fenetre * self.taux_echantillonnage)
                            decalage_fenetre = int(self.decalage * taille_fenetre_echantillons)

                            fenetres = []
                            debut = 0
                            fin = taille_fenetre_echantillons
                            nb_fenetres = 0

                            while fin <= taille_signal:
                                fenetre = axis_data[debut:fin]
                                fenetres.append(fenetre)

                                debut = debut + decalage_fenetre
                                fin = fin + decalage_fenetre
                                nb_fenetres += 1

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

################### Fin du Découpage en fenêtres ###################



################### Obtention des labels de fenêtres par rapport au temps ###################
    def label_fenetre(self):
        temps = self.fenetres_data["metadata"]["temps"]
        debuts_fog = self.fenetres_data["FOG"]["debut"]
        fins_fog = self.fenetres_data["FOG"]["fin"]
        #debuts_prefog= debuts_fog
        if isinstance(debuts_fog, int):
                debuts_fog = [debuts_fog]  # Transforme l'entier en liste contenant cet entier
        self.debuts_prefog = [max(0, x - self.temps_prefog) for x in debuts_fog]
        
        if isinstance(fins_fog, int):
            fins_fog = [fins_fog]  # Transforme l'entier en liste contenant cet entier        
        
        status="noFOG"
    
        # on stock les données d'évènement dans un dataframe ordonner en fonction du temps
        events=pd.DataFrame({'temps': debuts_fog + fins_fog + self.debuts_prefog, 
                    'events': ["debut_fog"]*len(debuts_fog) + ["fin_fog"]*len(fins_fog) + ["preFog"]*len(self.debuts_prefog)}).sort_values('temps').reset_index(drop=True)

        # On récupère si il y a un évènement de FOG ou plusieurs de présent de la FOG
        statuses = []
        status = "noFog"

        for window in temps:
            w_start = window[0]
            w_end = window[-1]
            window_events = []  # Liste pour stocker les événements de la fenêtre
    
            for _, row in events.iterrows():
                if row['temps'] >= w_start and row['temps'] <= w_end:
                    window_events.append(row['events'])  # Ajouter l'événement à la liste
    
            if not window_events:  # Si la liste est vide
                window_events = [None]  # Remplir avec None
    
            if status == "noFog" and "preFog" in window_events:
                status = "transitionPreFog"
        
            elif status == "transitionPreFog" and None in window_events:
                status = "preFog"
                
            elif status == "preFog" and "debut_fog" in window_events:
                status = "transitionFog"
        
            elif status == "transitionFog" and (None in window_events or ("debut_fog" in window_events and "fin_fog" in window_events)):
                status = "fog"
        
            elif status == "fog" and "fin_fog" in window_events and "debut_fog" not in window_events:
                status = "transitionNoFog"
        
            elif status == "transitionNoFog" and None in window_events:
                status = "noFog"
        
            statuses.append(status)  # Ajouter le statut à la liste des statuts
        
        # on associe les labels de fenêtre dans notre data
        self.fenetres_data["labels_fenetres"] = statuses
        
################### Fin de l'obtention des labels de fenêtres par rapport au temps ###################

################### Debut association labels fenetres aux datas ###################
    def association_label_fenetre_data(self):
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
            "preFog": self.debuts_prefog
    }
################### Fin association labels fenetres aux datas ###################


# class Statistics:
#     def __init__(self, __PreProcessing__):
#         self.__PreProcessing__ = __PreProcessing__
#         pass 
    
    