import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample, butter, lfilter
import ezc3d
from scipy.stats import mode, median_abs_deviation, iqr, trim_mean, entropy as ent, skew, kurtosis
from scipy.fft import fft, fftfreq
# from scipy.signal import welch, correlate, stft
# from statsmodels.tsa.ar_model import AutoReg
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import TimeSeriesSplit


class PreProcessing:
    def __init__(self, file_path):
        """
        Initialize the PreProcessing class with the given file path.
        
        Args:
            file_path (str): Path to the C3D file.
        """
        # Initialize attributes for JSON creation
        self.file_path = file_path
        self.c3d = ezc3d.c3d(file_path)
        self.events_dict = None
        self.resampled_times = None
        self.filtered_fusion_label_data = None
        self.labels_data_filtre = None
        self.labels_data_filtre_modifies = None
        self.normes = None
        self.json_data = None

        # Initialize attributes for data between START and END
        self.data_interval = None

        # Initialize attributes for normalization
        self.normalized_data = None

        # Initialize attributes for window segmentation
        self.taille_fenetre = 2
        self.decalage = 0.2
        self.taux_echantillonnage = 50    
        self.fenetres_data = None
        self.infos_fenetres = None    

        # Initialize attributes for window labeling
        self.temps_prefog = 3
        self.debuts_prefog = None

        # Initialize attributes for associating labels with data
        self.mix_label_fenetre_data = None

        # Initialize attributes for concatenating labels and data
        self.concat_data = None

    def recuperer_evenements(self):
        """
        Retrieve events from C3D data and create a dictionary with event times.
        """
        events = self.c3d['parameters']['EVENT']['LABELS']['value']
        temps_events = self.c3d['parameters']['EVENT']['TIMES']['value']

        # Standardize event names
        standardized_events = ['FOG_begin' if event == 'FoG_Begin' else event for event in events]
        standardized_events = ['FOG_end' if event == 'FoG_end' else event for event in standardized_events]

        df = pd.DataFrame({'events': standardized_events, 'frames': temps_events[1]})
        df.reset_index(drop=True, inplace=True)
        self.events_dict = df.groupby('events')['frames'].apply(list).to_dict()

    def butter_lowpass(self, cutoff, fs, order=2):
        """
        Calculate Butterworth lowpass filter coefficients.
        
        Args:
            cutoff (float): Cutoff frequency of the filter.
            fs (float): Sampling frequency.
            order (int): Order of the filter.
        
        Returns:
            tuple: Coefficients of the filter.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        """
        Apply Butterworth lowpass filter to data.
        
        Args:
            data (array-like): Data to filter.
            cutoff (float): Cutoff frequency of the filter.
            fs (float): Sampling frequency.
            order (int): Order of the filter.
        
        Returns:
            array-like: Filtered data.
        """
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def reechantillonnage_fc_coupure_et_association_labels_et_data(self, cutoff_freq=20, target_freq=50):
        """
        Resample data to target frequency, apply lowpass filter, and associate labels with data.
        
        Args:
            cutoff_freq (float): Cutoff frequency of the filter.
            target_freq (float): Target resampling frequency.
        """
        labels = self.c3d['parameters']['ANALOG']['LABELS']['value']
        original_freq = self.c3d['parameters']['ANALOG']['RATE']['value'][0]
        data = self.c3d['data']['analogs']
        nb_frame = len(data[0][0])
        nb_samples_target = int(nb_frame * (target_freq / original_freq))
        self.resampled_times = np.linspace(0., (nb_frame / original_freq), num=nb_samples_target)
        resampled_data = np.zeros((len(labels), nb_samples_target))
        for i in range(len(labels)):
            resampled_data[i, :] = resample(data[0][i, :], nb_samples_target)

        fusion_label_data = {}
        for i, label in enumerate(labels):
            fusion_label_data[label] = resampled_data[i, :]

        self.filtered_fusion_label_data = {}
        for label, data in fusion_label_data.items():
            filtered_signal = self.butter_lowpass_filter(data, cutoff_freq, target_freq)
            self.filtered_fusion_label_data[label] = filtered_signal

    def filtrer_labels(self):
        """
        Filter labels to keep only GYRO and ACC data.
        """
        self.labels_filtre = []
        self.labels_data_filtre = {}
        for label, valeurs in self.filtered_fusion_label_data.items():
            if 'ACC' in label or 'GYRO' in label:
                self.labels_data_filtre[label] = valeurs

    def modifier_label_pelvis(self):
        """
        Modify labels for the 'pelvis' sensor by adding 'Left_' prefix.
        
        Returns:
            dict: Modified labels dictionary.
        """
        self.labels_data_filtre_modifies = {}
        for key, value in self.labels_data_filtre.items():
            parts = key.split('_')
            if parts[0].lower() == "pelvis":
                new_key = '_'.join(["Left"] + parts)
                self.labels_data_filtre_modifies[new_key] = value
            else:
                self.labels_data_filtre_modifies[key] = value
        return self.labels_data_filtre_modifies

    def calcul_norme(self):
        """
        Calculate norms for filtered data.
        
        Returns:
            dict: Dictionary with calculated norms.
        """
        self.normes = {}
        traite = set()
        for key, value in self.labels_data_filtre_modifies.items():
            parts = key.split('_')
            sensor = parts[1]
            side = parts[0]
            measure = parts[2]

            if (side, sensor, measure) not in traite:
                traite.add((side, sensor, measure))

                axe_X = self.labels_data_filtre_modifies[f'{side}_{sensor}_{measure}_X']
                axe_Y = self.labels_data_filtre_modifies[f'{side}_{sensor}_{measure}_Y']
                axe_Z = self.labels_data_filtre_modifies[f'{side}_{sensor}_{measure}_Z']

                norme = np.sqrt(axe_X**2 + axe_Y**2 + axe_Z**2)
                nom_cle = f'{side}_{sensor}_{measure}_norme'
                self.normes[nom_cle] = norme

        return self.normes

    def creer_structure_json(self):
        """
        Create JSON structure from filtered data and other information.
        
        Returns:
            dict: JSON structure containing data and metadata.
        """
        parts = self.file_path.split('/')
        filename = parts[-1].split('_')
        patient_id = '_'.join(filename[:3])
        date_de_naissance = filename[2]
        medicaments = filename[3]
        condition = filename[4].split(' ')[0]

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

            if sensor not in self.json_data:
                self.json_data[sensor] = {}

            if side not in self.json_data[sensor]:
                self.json_data[sensor][side] = {}

            if measure not in self.json_data[sensor][side]:
                self.json_data[sensor][side][measure]["norme"] = self.normes[key].tolist()

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

    def creation_json_grace_c3d(self):
        """
        Main function to create a JSON file from C3D data.
        
        Returns:
            dict: JSON data structure.
        """
        self.recuperer_evenements()
        self.reechantillonnage_fc_coupure_et_association_labels_et_data()
        self.filtrer_labels()
        self.modifier_label_pelvis()
        self.calcul_norme()
        self.json_data = self.creer_structure_json()
        return self.json_data

    def plot_data_FOG(self, muscle, side, sensor_type, axis):
        """
        Plot data with FOG events for visualization.
        
        Args:
            muscle (str): Muscle name.
            side (str): Side (Left or Right).
            sensor_type (str): Sensor type (ACC or GYRO).
            axis (str): Axis (X, Y, Z).
        """
        def plot_events_vertical_lines(events, color, linestyle, label):
            if isinstance(events, list):
                for event in events:
                    plt.axvline(x=event, color=color, linestyle=linestyle, label=label)

        events_1_begin = self.json_data["FOG"].get("debut", [])
        events_1_end = self.json_data["FOG"].get("fin", [])

        data_to_plot = self.json_data[muscle][side][sensor_type][axis]
        plt.figure(figsize=(12, 6))
        plt.plot(self.resampled_times, data_to_plot)
        title = f"{muscle} - {side} - {sensor_type} - {axis}"

        plot_events_vertical_lines(events_1_begin, 'green', '--', 'FOG_begin')
        plot_events_vertical_lines(events_1_end, 'red', '--', 'FOG_end')

        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
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

    def extract_data_interval(self):
        """
        Extract data interval from START to END event.
        
        Returns:
            dict: Dictionary containing the extracted data interval.
        """
        start_time = self.json_data["parcours"]["START"][0]
        end_time = self.json_data["parcours"]["END"][0]
        epsilon = 0.01

        for i, time in enumerate(self.resampled_times):
            if abs(time - start_time) < epsilon:
                start_index = i
            if abs(time - end_time) < epsilon:
                end_index = i

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

        self.data_interval["metadata"] = self.json_data["metadata"]
        self.data_interval["parcours"] = self.json_data["parcours"]
        self.data_interval["FOG"] = self.json_data["FOG"]

        metadata_temps_interval = self.resampled_times[start_index:end_index+1]
        self.data_interval["metadata"]["temps"] = metadata_temps_interval
        return self.data_interval

    def plot_data_FOG_start_end(self, muscle, side, sensor_type, axis):
        """
        Plot data with FOG events and START-END interval for visualization.
        
        Args:
            muscle (str): Muscle name.
            side (str): Side (Left or Right).
            sensor_type (str): Sensor type (ACC or GYRO).
            axis (str): Axis (X, Y, Z).
        """
        def plot_events_vertical_lines(events, color, linestyle, label):
            if isinstance(events, list):
                for event in events:
                    plt.axvline(x=event, color=color, linestyle=linestyle, label=label)

        events_1_begin = self.data_interval["FOG"].get("debut", [])
        events_1_end = self.data_interval["FOG"].get("fin", [])

        data_to_plot = self.data_interval[muscle][side][sensor_type][axis]
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_interval["metadata"]["temps"], data_to_plot)
        title = f"{muscle} - {side} - {sensor_type} - {axis}"

        plot_events_vertical_lines(events_1_begin, 'green', '--', 'FOG_begin')
        plot_events_vertical_lines(events_1_end, 'red', '--', 'FOG_end')

        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
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

    def normalize_data(self):
        """
        Normalize data between START and END events.
        
        Returns:
            dict: Dictionary containing normalized data.
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
                            mean_val = np.mean(axis_data)
                            max_val = np.max(axis_data)
                            min_val = np.min(axis_data)
                            normalized_axis_data = (axis_data - mean_val) / (max_val - min_val)
                            self.normalized_data[sensor][side][measure][axis] = normalized_axis_data

        self.normalized_data["metadata"] = self.data_interval["metadata"]
        self.normalized_data["parcours"] = self.data_interval["parcours"]
        self.normalized_data["FOG"] = self.data_interval["FOG"]
        return self.normalized_data

    def decoupage_en_fenetres(self):
        """
        Segment data into windows with 80% overlap.
        
        Returns:
            dict: Dictionary containing segmented window data.
        """
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

        self.fenetres_data["metadata"] = self.normalized_data["metadata"]
        self.fenetres_data["parcours"] = self.normalized_data["parcours"]
        self.fenetres_data["FOG"] = self.normalized_data["FOG"]

        self.fenetres_data["metadata"]["temps"] = fenetres_temps
        return self.fenetres_data

    def label_fenetre(self):
        """
        Label each window based on FOG events.
        
        Returns:
            dict: Dictionary containing labeled window data.
        """
        temps = self.fenetres_data["metadata"]["temps"]
        debuts_fog = self.fenetres_data["FOG"]["debut"]
        fins_fog = self.fenetres_data["FOG"]["fin"]

        if isinstance(fins_fog, int):
            fins_fog = [fins_fog]

        events = pd.DataFrame({'temps': debuts_fog + fins_fog, 'events': ["debut_fog"]*len(debuts_fog) + ["fin_fog"]*len(fins_fog)}).sort_values('temps').reset_index(drop=True)

        statuses = []
        status = "noFog"

        for window in temps:
            w_start = window[0]
            w_end = window[-1]
            window_events = []
            time = []
            time_pourcent = 1 

            for _, row in events.iterrows():
                if row['temps'] >= w_start and row['temps'] <= w_end:
                    window_events.append(row['events'])
                    if row['events'] == "fin_fog":
                        time.append(row["temps"])

            if len(window_events) == 1 and "fin_fog" in window_events:
                time_array = np.arange(w_start, w_end, 1/50)
                time_pourcent = np.sum(time_array <= time) / 100

            if not window_events:
                window_events = [None]

            if status == "noFog" and "debut_fog" in window_events:
                status = "transitionFog"
            elif status == "transitionFog" and None in window_events:
                status = "fog"
            elif status == "transitionFog" and ("debut_fog" in window_events and "fin_fog" in window_events):
                status = "fog"
            elif status == "fog" and ("fin_fog" in window_events and "debut_fog" in window_events):
                status = "transitionFog"
            elif status == "fog" and "fin_fog" in window_events and time_pourcent <= 0.5:
                status = "transitionNoFog"
            elif status == "transitionNoFog" and None in window_events:
                status = "noFog"

            statuses.append(status)

        self.fenetres_data["labels_fenetres"] = statuses
        return self.fenetres_data

    def association_label_fenetre_data(self):
        """
        Associate labels with window data.
        
        Returns:
            dict: Data with associated labels.
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
                            presentWin = np.unique(self.fenetres_data["labels_fenetres"])
                            for label in presentWin:
                                data_frame_axis_data = pd.DataFrame(axis_data)
                                self.mix_label_fenetre_data[sensor][side][measure][axis][label] = data_frame_axis_data[[x == label for x in self.fenetres_data["labels_fenetres"]]]

        self.mix_label_fenetre_data["metadata"] = self.fenetres_data["metadata"]
        self.mix_label_fenetre_data["metadata"]["temps"] = pd.DataFrame(self.mix_label_fenetre_data["metadata"]["temps"])
        self.mix_label_fenetre_data["parcours"] = self.fenetres_data["parcours"]
        self.mix_label_fenetre_data["FOG"] = self.fenetres_data["FOG"]

        self.mix_label_fenetre_data["FOG"] = {
            "debut": self.fenetres_data["FOG"]["debut"],
            "fin": self.fenetres_data["FOG"]["fin"]
        }
        return self.mix_label_fenetre_data

    def concat_label_fenetre_data(self):
        """
        Concatenate window data with labels.
        
        Returns:
            DataFrame: Dataframe with concatenated data and labels.
        """
        self.concat_data = {}

        for muscle, muscle_data in self.mix_label_fenetre_data.items():
            if muscle not in ["metadata", "parcours", "FOG"]:
                for side, side_data in muscle_data.items():
                    for sensor, sensor_data in side_data.items():
                        for axis, axis_data in sensor_data.items():
                            dfs = []
                            for label, label_data in axis_data.items():
                                label_data['label'] = label
                                dfs.append(label_data)
                            combined_df = pd.concat(dfs) if dfs else pd.DataFrame()
                            combined_df = combined_df[['label'] + [col for col in combined_df.columns if col != 'label']]
                            if muscle not in self.concat_data:
                                self.concat_data[muscle] = {}
                            if side not in self.concat_data[muscle]:
                                self.concat_data[muscle][side] = {}
                            if sensor not in self.concat_data[muscle][side]:
                                self.concat_data[muscle][side][sensor] = {}
                            self.concat_data[muscle][side][sensor][axis] = combined_df.sort_index()

        self.concat_data["metadata"] = self.mix_label_fenetre_data["metadata"]
        self.concat_data["parcours"] = self.mix_label_fenetre_data["parcours"]
        self.concat_data["FOG"] = self.mix_label_fenetre_data["FOG"]

        return self.concat_data

    def plot_data_FOG_start_end_final(self, muscle, side, sensor_type, axis, window_index):
        """
        Plot data with FOG events and START-END interval for a specific window.
        
        Args:
            muscle (str): Muscle name.
            side (str): Side (Left or Right).
            sensor_type (str): Sensor type (ACC or GYRO).
            axis (str): Axis (X, Y, Z).
            window_index (int): Index of the window to plot.
        """
        def plot_events_vertical_lines(events, color, linestyle, label):
            if isinstance(events, list):
                for event in events:
                    plt.axvline(x=event, color=color, linestyle=linestyle, label=label)

        events_1_begin = self.concat_data["FOG"].get("debut", [])
        events_1_end = self.concat_data["FOG"].get("fin", [])

        data_to_plot = self.concat_data[muscle][side][sensor_type][axis][window_index]
        data_to_plot = data_to_plot.drop(columns=["label"])
        plt.figure(figsize=(12, 6))
        plt.plot(self.concat_data["metadata"]["temps"][window_index], data_to_plot)
        title = f"{muscle} - {side} - {sensor_type} - {axis} - {window_index}"

        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title(title)
        plt.tight_layout()
        plt.show()

class Statistics:
    def __init__(self, file_path, concat_data):
        """
        Initialize the Statistics class with the given file path and concatenated data.
        
        Args:
            file_path (str): Path to the C3D file.
            concat_data (dict): Concatenated data dictionary.
        """
        self.concat_data = concat_data
        self.taille_fenetre = 2
        self.file_path = file_path

    def stats(self):
        """
        Calculate various statistics for the data.
        
        Returns:
            DataFrame: DataFrame containing calculated statistics.
        """
        first_time = self.concat_data["metadata"]["temps"].iloc[0, 0]

        if np.isnan(self.concat_data["metadata"]["temps"].iloc[-1, -1]):
            first_na = np.where(np.isnan(self.concat_data["metadata"]["temps"]))[-1][0]
            last_time = self.concat_data["metadata"]["temps"].iloc[-1, (first_na - 1)]
        else:
            last_time = self.concat_data["metadata"]["temps"].iloc[-1, -1]

        temps_total = last_time - first_time

        if self.concat_data["FOG"]["debut"] == [0]:
            nb_fog = 0
            temps_fog = 0
        else:
            nb_fog = len(self.concat_data["FOG"]["debut"])
            temps_fog = sum(fin - debut for debut, fin in zip(self.concat_data["FOG"]["debut"], self.concat_data["FOG"]["fin"]))

        prct_fog = (temps_fog / temps_total) * 100 if temps_total > 0 else 0

        nb_fenetre = len(self.concat_data["metadata"]["temps"])
        nb_colonne = len(self.concat_data["metadata"]["temps"].columns)

        parts = self.file_path.split('/')
        filename = parts[-1].split('_')
        identifiant = '_'.join(filename[:3])
        statut = filename[3]
        condition = filename[4]
        video = filename[5].replace('.c3d', '')

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

        return tab_stat

class ExtractionFeatures:
    def __init__(self, concat_data):
        """
        Initialize the ExtractionFeatures class with the given concatenated data.
        
        Args:
            concat_data (dict): Concatenated data dictionary.
        """
        self.data = concat_data
        self.fs = 50
        self.fft_magnitudes = None
        self.frequencies = None
        self.label = None

    def enlever_derniere_ligne_et_colonne_label(self):
        """
        Remove the last row and 'label' column from data.
        
        Returns:
            tuple: Cleaned data and labels.
        """
        for sensor, sensor_data in self.data.items():
            if sensor not in ["metadata", "parcours", "FOG"]:
                for side, side_data in sensor_data.items():
                    for measure, measure_data in side_data.items():
                        for axis, axis_data in measure_data.items():
                            if isinstance(axis_data, pd.DataFrame):
                                data_moins_derniere_ligne_na = axis_data.drop(axis_data.index[-1])
                                self.label = data_moins_derniere_ligne_na["label"]
                                if 'label' in data_moins_derniere_ligne_na.columns:
                                    data_moins_colonne_label = data_moins_derniere_ligne_na.drop(columns=["label"])
                                    measure_data[axis] = data_moins_colonne_label
        return self.data, self.label

    def transformation_domaine_frequentiel(self, axis_data):
        """
        Perform Fourier Transform on axis data.
        
        Args:
            axis_data (DataFrame): Data for a specific axis.
        
        Returns:
            tuple: FFT magnitudes and frequencies.
        """
        n = axis_data.shape[1]
        frequences = fftfreq(n, d=1/self.fs)
        frequences = frequences[:n//2]
        self.frequencies = fftfreq(n, d=1/self.fs)
        self.frequencies = self.frequencies[:n//2]
        self.frequencies = self.frequencies.reshape((1, -1))

        fft_result = fft(axis_data, axis=1)
        self.fft_magnitudes = np.abs(fft_result)[:, :n//2]
        self.fft_magnitudes = pd.DataFrame(self.fft_magnitudes)
        self.frequencies = pd.DataFrame(self.frequencies)

        return self.fft_magnitudes, self.frequencies

    def extract_temporal_features(self, axis_data):
        """
        Extract temporal features from data.
        
        Args:
            axis_data (DataFrame): Data for a specific axis.
        
        Returns:
            DataFrame: DataFrame containing extracted features.
        """
        df_features = pd.DataFrame()
        df_features['Mean_Temporal'] = np.mean(axis_data, axis=1)
        df_features['Ecart_Type_Temporal'] = np.std(axis_data, axis=1)
        df_features['Variance_Temporal'] = np.var(axis_data, axis=1)
        df_features['Energy_Temporal'] = np.sum(np.square(axis_data), axis=1)
        df_features['Range'] = np.ptp(axis_data, axis=1)
        df_features['RMS'] = np.sqrt(np.mean(np.square(axis_data), axis=1))
        df_features['Median_Temporal'] = np.median(axis_data, axis=1)
        df_features['Trimmed_Mean'] = trim_mean(axis_data, 0.1, axis=1)
        df_features['Mean_Absolute_Value'] = np.mean(np.abs(axis_data), axis=1)
        df_features['Median_Absolute_Deviation'] = median_abs_deviation(axis_data, axis=1, nan_policy='omit')
        df_features['25th_percentile'] = np.percentile(axis_data, 25, axis=1)
        df_features['75th_percentile'] = np.percentile(axis_data, 75, axis=1)
        df_features['Interquartile_range'] = iqr(axis_data, axis=1, rng=(25,75), nan_policy="omit")
        df_features['Skewness_Temporal'] = skew(axis_data, axis=1)
        df_features['Kurtosis_Temporal'] = kurtosis(axis_data, axis=1)
        mean = np.mean(axis_data, axis=1)
        df_features['Increments_Mean'] = np.diff(mean, prepend=mean[0])
        df_features['Coefficient_Variation'] = np.std(axis_data, axis=1) / np.mean(axis_data, axis=1)

        return df_features

    def calcul_entropie_spectrale(self):
        """
        Calculate spectral entropy for each window.
        
        Returns:
            DataFrame: DataFrame containing spectral entropy.
        """
        entropie_spectrale = []

        for index, row in self.fft_magnitudes.iterrows():
            puissance_totale = np.sum(row**2)
            p_i = (row**2) / puissance_totale
            p_i = p_i[p_i > 0]
            H = -np.sum(p_i * np.log(p_i))
            entropie_spectrale.append(H)

        df_entropie_spectrale = pd.DataFrame({'Entropie_Spectrale': entropie_spectrale})
        return df_entropie_spectrale

    def calcul_details_harmoniques(self):
        """
        Calculate harmonic details for each window.
        
        Returns:
            DataFrame: DataFrame containing harmonic details.
        """
        premiere_harmonique_mag = []
        deuxieme_harmonique_mag = []
        premiere_harmonique_freq = []
        deuxieme_harmonique_freq = []
        distance_harmonique_frequence = []
        distance_harmonique_magnitude = []
        centre_densite_spectrale = []
        centre_densite_spectrale_puissance = []
        rapport_harmonique_frequence = []
        rapport_harmonique_magnitude = []
        crete_spectrale_puissance_ponderee_gpt = []
        crete_spectrale_puissance_ponderee_borzi = []
        largeurs_harmoniques = []

        for index, row in self.fft_magnitudes.iterrows():
            magnitudes = row.values
            frequences = self.frequencies.values.flatten()

            indices_harmoniques = np.argsort(magnitudes)[-2:]
            if magnitudes[indices_harmoniques[0]] > magnitudes[indices_harmoniques[1]]:
                premiere_harmonique, deuxieme_harmonique = indices_harmoniques[0], indices_harmoniques[1]
            else:
                premiere_harmonique, deuxieme_harmonique = indices_harmoniques[1], indices_harmoniques[0]

            cds = np.sum(frequences * magnitudes) / np.sum(magnitudes)
            cds_puissance = np.sum(frequences * magnitudes**2) / np.sum(magnitudes**2)
            cs_puissance_ponderee_gpt = np.max(magnitudes**2) / np.sum(magnitudes**2)
            cs_puissance_ponderee_borzi = (magnitudes[premiere_harmonique]**2) * frequences[premiere_harmonique]

            premiere_harmonique_mag.append(magnitudes[premiere_harmonique])
            deuxieme_harmonique_mag.append(magnitudes[deuxieme_harmonique])
            premiere_harmonique_freq.append(frequences[premiere_harmonique])
            deuxieme_harmonique_freq.append(frequences[deuxieme_harmonique])
            centre_densite_spectrale.append(cds)
            centre_densite_spectrale_puissance.append(cds_puissance)
            crete_spectrale_puissance_ponderee_gpt.append(cs_puissance_ponderee_gpt)
            crete_spectrale_puissance_ponderee_borzi.append(cs_puissance_ponderee_borzi)
            distance_harmonique_frequence.append(abs(frequences[premiere_harmonique] - frequences[deuxieme_harmonique]))

            if frequences[deuxieme_harmonique] == 0:
                rapport_harmonique_frequence.append(0)
            else:
                rapport_harmonique_frequence.append(frequences[premiere_harmonique] / frequences[deuxieme_harmonique])

            distance_harmonique_magnitude.append(abs(magnitudes[premiere_harmonique] - magnitudes[deuxieme_harmonique]))

            if magnitudes[deuxieme_harmonique] == 0:
                rapport_harmonique_magnitude.append(0)
            else:
                rapport_harmonique_magnitude.append(magnitudes[premiere_harmonique] / magnitudes[deuxieme_harmonique])

            premiere_harmonique_magnitude = magnitudes[premiere_harmonique]
            gauche = np.where(magnitudes[:premiere_harmonique] < premiere_harmonique_magnitude * 0.5)[0]
            if len(gauche) > 0:
                indice_gauche = gauche[-1] + 1
            else:
                indice_gauche = 0

            droite = np.where(magnitudes[premiere_harmonique+1:] < premiere_harmonique_magnitude * 0.5)[0]
            if len(droite) > 0:
                indice_droite = droite[0] + premiere_harmonique + 1
            else:
                indice_droite = len(magnitudes) - 1

            largeur_hz = frequences[indice_droite] - frequences[indice_gauche]
            largeurs_harmoniques.append(largeur_hz)

        df_resultats = pd.DataFrame({
            'Premiere_Harmonique_Magnitude': premiere_harmonique_mag,
            'Deuxieme_Harmonique_Magnitude': deuxieme_harmonique_mag,
            'Premiere_Harmonique_Frequence': premiere_harmonique_freq,
            'Deuxieme_Harmonique_Frequence': deuxieme_harmonique_freq,
            'Distance_Harmonique_Frequence': distance_harmonique_frequence,
            'Distance_Harmonique_Amplitude': distance_harmonique_magnitude,
            'Rapport_Harmonique_Frequence': rapport_harmonique_frequence,
            'Rapport_Harmonique_Amplitude': rapport_harmonique_magnitude,
            'Centre_Densite_Spectrale': centre_densite_spectrale,
            'Centre_Densite_Spectrale_Puissance': centre_densite_spectrale_puissance,
            'Crete_Spectrale_Puissance_Ponderee_GPT': crete_spectrale_puissance_ponderee_gpt,
            'Crete_Spectrale_Puissance_Ponderee_Borzi': crete_spectrale_puissance_ponderee_borzi,
            'Largeur_Harmonique': largeurs_harmoniques
        })

        return df_resultats

    def ecart_type_borne(self):
        """
        Calculate standard deviation for specific frequency bands.
        
        Returns:
            DataFrame: DataFrame containing calculated standard deviations.
        """
        bandes_frequence = {
            'ecart_type': (0, 50),
            'ecart_type_0.04_0.68_Hz': (0.04, 0.68),
            'ecart_type_0.68_3_Hz': (0.68, 3),
            'ecart_type_3_8_Hz': (3, 8),
            'ecart_type_8_20_Hz': (8, 20),
            'ecart_type_0.1_8_Hz': (0.1, 8)
        }

        frequences = self.frequencies.values.flatten()
        ecarts_types = pd.DataFrame()

        for nom_bande, (freq_min, freq_max) in bandes_frequence.items():
            colonnes_bande = (frequences >= freq_min) & (frequences <= freq_max)
            magnitudes_bande = self.fft_magnitudes.loc[:, colonnes_bande]
            ecarts_types[nom_bande] = magnitudes_bande.std(axis=1)
            
        return ecarts_types

    def calculer_freeze_index(self):
        """
        Calculate the Freeze Index for each window.
        
        Returns:
            DataFrame: DataFrame containing the Freeze Index.
        """
        magnitudes = self.fft_magnitudes.values
        frequences = self.frequencies.values.flatten()
        
        def calculer_aire_sous_spectre(frequences, magnitudes, freq_min, freq_max):
            indices_bande = (frequences >= freq_min) & (frequences <= freq_max)
            magnitudes_bande = magnitudes[:, indices_bande]
            aire_sous_spectre = np.trapz(magnitudes_bande, x=frequences[indices_bande], axis=1)
            return aire_sous_spectre

        bande_freeze = (3, 8)
        bande_locomotrice = (0.5, 3)
        aire_freeze = calculer_aire_sous_spectre(frequences, magnitudes, *bande_freeze)
        aire_locomotrice = calculer_aire_sous_spectre(frequences, magnitudes, *bande_locomotrice)
        freeze_index = (aire_freeze ** 2) / (aire_locomotrice ** 2)
        freeze_index_df = pd.DataFrame({'Freeze_Index': freeze_index})

        return freeze_index_df

    def ratio_faible_puissance_entre_0_2Hz(self):
        """
        Calculate the ratio of low power between 0 and 2 Hz.
        
        Returns:
            DataFrame: DataFrame containing the calculated ratio.
        """
        magnitudes = self.fft_magnitudes.values
        frequences = self.frequencies.values.flatten()
        ratios = []
        psd = np.abs(magnitudes)**2
        puissance_totale = np.sum(psd, axis=1)
        bande_indices = (frequences >= 0) & (frequences <= 2)
        psd_band = psd[:, bande_indices]
        puissance_bande = np.sum(psd_band, axis=1)
        ratios = puissance_bande / puissance_totale
        ratios_df = pd.DataFrame({'Ratio_Faible_Puissance_0_2Hz': ratios})
        
        return ratios_df

    def calcul_locomotion_band_power(self):
        """
        Calculate locomotion band power (0.5-3 Hz).
        
        Returns:
            DataFrame: DataFrame containing locomotion band power.
        """
        magnitudes = self.fft_magnitudes.values
        frequences = self.frequencies.values.flatten()
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

    def calcul_freeze_band_power(self):
        """
        Calculate freeze band power (3-8 Hz).
        
        Returns:
            DataFrame: DataFrame containing freeze band power.
        """
        magnitudes = self.fft_magnitudes.values
        frequences = self.frequencies.values.flatten()
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
        """
        Calculate band power (0.5-8 Hz).
        
        Returns:
            DataFrame: DataFrame containing band power.
        """
        magnitudes = self.fft_magnitudes.values
        frequences = self.frequencies.values.flatten()
        bande_power_list = []
        psd = np.abs(magnitudes)**2
        bande_power = (frequences >= 0.5) & (frequences <= 8)
        psd_bande_power = psd[:, bande_power]
        puissance_bande_power = np.sum(psd_bande_power, axis=1)

        for window in puissance_bande_power:
            bande_power = window / 50
            bande_power_list.append(bande_power)
        
        df_bande_power = pd.DataFrame({'Band_Power': bande_power_list})
        return df_bande_power

    def calcul_energie(self):
        """
        Calculate energy for each signal.
        
        Returns:
            DataFrame: DataFrame containing energy.
        """
        magnitudes = self.fft_magnitudes.values
        energy = np.sum(np.abs(magnitudes)**2 / len(magnitudes), axis=1)
        df_energie = pd.DataFrame({'Energie_Frequentielle': energy})
        return df_energie

    def dataframe_caracteristiques_final(self):
        """
        Extract features from each window and create a final DataFrame.
        
        Returns:
            DataFrame: DataFrame containing extracted features.
        """
        data_collect = []

        for sensor, sensor_data in self.data.items():
            if sensor not in ["metadata", "parcours", "FOG"]:
                for side, side_data in sensor_data.items():
                    for measure, measure_data in side_data.items():
                        for axis, axis_data in measure_data.items():
                            if isinstance(axis_data, pd.DataFrame):
                                features_temporelles = self.extract_temporal_features(axis_data)
                                fft_magnitude, frequencies = self.transformation_domaine_frequentiel(axis_data)
                                entropie_spectrale = self.calcul_entropie_spectrale()
                                details_harmoniques = self.calcul_details_harmoniques()
                                ecart_types = self.ecart_type_borne()
                                freeze_index = self.calculer_freeze_index()
                                ratio_faible_puissance = self.ratio_faible_puissance_entre_0_2Hz()
                                locomotion_band_power = self.calcul_locomotion_band_power()
                                freeze_band_power = self.calcul_freeze_band_power()
                                band_power = self.calcul_band_power()
                                energie = self.calcul_energie()

                                caract_features = pd.concat([features_temporelles,
                                                            entropie_spectrale,
                                                            details_harmoniques, 
                                                            ecart_types,
                                                            freeze_index,
                                                            ratio_faible_puissance,
                                                            locomotion_band_power,
                                                            freeze_band_power,
                                                            band_power,
                                                            energie], axis=1)

                                caract_features.rename(columns={feature_name: f"{sensor}_{side}_{measure}_{axis}_{feature_name}" for feature_name in caract_features.columns}, inplace=True)
                                data_collect.append(caract_features)

        df_final = pd.concat(data_collect, axis=1)
        label_dataframe = pd.DataFrame(self.label)
        data_concat = pd.concat([df_final, label_dataframe], axis=1)
        combined_df = data_concat[['label'] + [col for col in data_concat.columns if col != 'label']]
        
        return combined_df
