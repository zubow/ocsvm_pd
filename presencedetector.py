import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from collections import Counter
import re

class PresenceDetector:
    """
        Device-free presence detection using OC-SVM

        usage: see test_pd.py
        A. Zubow
    """
    def __init__(self, Config, GEN_PAPER_PLOTS = False):
        self.verbose = Config.verbose
        self.csi_folder = Config.csi_folder
        self.csi_sampling_rate = Config.csi_sampling_rate
        self.Ntx = Config.Ntx
        self.Nrx = Config.Nrx
        self.csi_skip_first_n_seconds = Config.csi_skip_first_n_seconds
        self.observed_time_ms = Config.observed_time_ms
        self.downselected_subcarriers = Config.downselected_subcarriers
        self.time_window = Config.time_window
        self.ant_mode = Config.ant_mode
        self.ant_submode = Config.ant_submode
        self.GEN_PAPER_PLOTS = GEN_PAPER_PLOTS
        self.data_store = {}
        self.rooms_to_test = []
        self.rooms_to_train = []
        self.ocsvm = svm.OneClassSVM(kernel=Config.kernel, gamma=Config.gamma, nu=Config.nu, verbose=self.verbose)
        self.post_processing_window_sz = Config.post_processing_window_sz


    def load_data(self, rooms_to_train, rooms_to_test):
        """
            Load training and test data from file
        """
        train_only = list(set(rooms_to_train) - set(rooms_to_test))
        test_only = list(set(rooms_to_test) - set(rooms_to_train))
        train_and_test = set(rooms_to_train).intersection(set(rooms_to_test))

        # room id is encoding room and channel
        temp = re.compile("([a-zA-Z]+)([0-9]+)")

        for room in train_and_test:
            res = temp.match(room).groups()
            fprefix = self.csi_folder + "63000x_" + res[0] + "_" + res[1]
            self.load_data_for_test_and_train(room, fprefix + "_empty.npy", fprefix + "_occupied.npy")

        for room in train_only:
            res = temp.match(room).groups()
            fprefix = self.csi_folder + "63000x_" + res[0] + "_" + res[1]
            self.load_train_data(room, fprefix + "_empty.npy", fprefix + "_occupied.npy")

        for room in test_only:
            res = temp.match(room).groups()
            fprefix = self.csi_folder + "63000x_" + res[0] + "_" + res[1]
            self.load_test_data(room, fprefix + "_empty.npy", fprefix + "_occupied.npy")


    def load_test_data(self, room_name, file_room_empty, file_room_occupied):
        self.load_csi_data(room_name, file_room_empty, file_room_occupied)
        self.rooms_to_test.append(room_name)


    def load_train_data(self, room_name, file_room_empty, file_room_occupied):
        self.load_csi_data(room_name, file_room_empty, file_room_occupied)
        self.rooms_to_train.append(room_name)


    def load_data_for_test_and_train(self, room_name, file_room_empty, file_room_occupied):
        self.load_csi_data(room_name, file_room_empty, file_room_occupied)
        self.rooms_to_test.append(room_name)
        self.rooms_to_train.append(room_name)


    def print_data_store(self):
        print('training in: %s' % ','.join(self.rooms_to_train))
        print('testing in: %s' % ','.join(self.rooms_to_test))


    def load_csi_data(self, room_name, file_room_empty, file_room_occupied):
        # load from file
        data_empty_raw = np.load(file_room_empty)
        data_occupied_raw = np.load(file_room_occupied)

        # skip first seconds
        skip_csi_samples = int(self.csi_skip_first_n_seconds * self.csi_sampling_rate)
        # remove incorrect CSI samples, i.e. zero entries
        data_empty_raw = self._remove_zero_csi_samples(data_empty_raw[skip_csi_samples:])
        data_occupied_raw = self._remove_zero_csi_samples(data_occupied_raw[skip_csi_samples:])

        self.data_store[room_name] = {'empty': data_empty_raw, 'occupied': data_occupied_raw}


    def _remove_zero_csi_samples(self, data):
        """ CSI cleansing """
        to_be_deleted_ids = self._get_ids_of_zero_csi_samples(data)
        #print('... from empty room: %d' % len(to_be_deleted_ids))
        data_no_zeros = np.delete(data, list(to_be_deleted_ids), axis=0)
        return data_no_zeros


    def preprocess(self):
        """ Pre-processing CSI data """
        for room in self.data_store:
            self.data_store[room]['empty_pre'] = self._preprocess(self.data_store[room]['empty'], 'empty')
            self.data_store[room]['occupied_pre'] = self._preprocess(self.data_store[room]['occupied'], 'occupied')


    def train(self):
        """ Train the OC-SVM with preprocessed CSI data from empty rooms only‚ """
        train_data = np.empty(shape=[0, self.Nrx * self.Ntx * self.downselected_subcarriers * (self.time_window + 1)])
        for room in self.rooms_to_train:
            # flatten data for SVM input
            train_data = np.concatenate((train_data, np.reshape(self.data_store[room]['empty_pre'],(self.data_store[room]['empty_pre'].shape[0],-1))))

        print('learning ... #batches: %d' % train_data.shape[0])
        self.ocsvm.fit(train_data)


    def predict(self):
        """ Predict the human presence based on testing data """
        self.cmat = np.zeros(4)
        for room in self.rooms_to_test:
            for room_label_id, room_label in zip([1, -1], ['empty_pre', 'occupied_pre']):
                test_data1 = np.reshape(self.data_store[room][room_label], (self.data_store[room][room_label].shape[0], -1))
                prediction1 = self.ocsvm.predict(test_data1)
                post_prediction1 = self._postprocessing(prediction1)

                label1 = np.full(self.data_store[room][room_label].shape[0] - self.post_processing_window_sz, room_label_id) # 1 and -1
                cmat1 = self._get_confusion_matrix(label1, post_prediction1)

                self.cmat = self.cmat + cmat1


    def _postprocessing(self, prediction):
        res_win = []
        prediction_post = np.empty_like(prediction)
        for i,pred in enumerate(prediction):
            res_win.append(pred)
            if len(res_win) > self.post_processing_window_sz:
                res_win.pop(0)
            prediction_post[i] = Counter(res_win).most_common(1)[0][0]
        return prediction_post[self.post_processing_window_sz:]


    def get_metrics(self):
        """ Compute the performance metrics """
        accuracy = (self.cmat[0] + self.cmat[3]) / np.sum(self.cmat)
        sensitivity = self.cmat[0] / (self.cmat[0] + self.cmat[2])
        specificity = self.cmat[3] / (self.cmat[1] + self.cmat[3])
        print('performance: accuracy: %f, sensitivity: %f, specificity: %f' % (accuracy, sensitivity, specificity))


    def _get_confusion_matrix(self, label, prediction):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, pred in enumerate(prediction):
            if pred == 1 and label[i] == pred:
                tp += 1
            elif pred == 1 and label[i] != pred:
                fp += 1
            elif pred == -1 and label[i] == pred:
                tn += 1
            else:
                fn += 1
        return np.array([tp, fp, fn, tn])


    def plot_empty_preprocessed(self, room, batch_index, antenna_num):
        self._plot_heatmap(self.data_store[room]['empty_pre'], 'empty', batch_index, antenna_num, 'empty.pdf')


    def plot_occupied_preprocessed(self, room, batch_index, antenna_num):
        self._plot_heatmap(self.data_store[room]['occupied_pre'], 'occupied', batch_index, antenna_num, 'occupied.pdf')


    def _plot_heatmap(self, data, title, batch_index, antenna_num, fname=None):
        data = data[batch_index][:, antenna_num, :]
        print(data.shape)

        preprocessed_data = np.transpose(data, [1, 0])
        plt.xlabel("Time [sample]")
        plt.ylabel("Freq [carrier]")
        plt.imshow(preprocessed_data, vmin=preprocessed_data.min(), vmax=preprocessed_data.max(), aspect="auto", cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        if fname:
            plt.savefig('paper/figs/' + fname, bbox_inches="tight")
        plt.show()


    def _preprocess(self, data_raw, title):
        """ CSI pre-processing; see paper for details‚ """

        # calculate number of CSI instances used in one Batch, depending on CSI sampling rate
        n_timestamps = int(self.observed_time_ms * self.csi_sampling_rate / 100)
        # calculate number of Batches, depending on CSI sampling rate
        instances = int(data_raw.shape[0] / n_timestamps)
        # downscale subcarrier spacing depending on available subcarriers
        subcarrier_spacing = int(data_raw.shape[-1] / self.downselected_subcarriers)

        # add batch dimension
        data_preprocessed = np.resize(data_raw, (instances, n_timestamps,) + data_raw.shape[1:])

        # antenna filtering
        if self.ant_mode == 3:
            # MIMO: use all 4 paths
            pass
        elif self.ant_mode == 2:
            # SIMO
            tx_ant_idx = self.ant_submode - 1 # -1 and 0
            data_preprocessed = data_preprocessed[:, :, tx_ant_idx::2, :, :]
        elif self.ant_mode == 1:
            # MISO
            rx_ant_idx = self.ant_submode - 1  # -1 and 0
            data_preprocessed = data_preprocessed[:, :, :, rx_ant_idx::2, :]
        elif self.ant_mode == 0:
            # SISO
            tx_ant_idx = np.floor_divide(self.ant_submode,2) - 1 # -1 and 0
            rx_ant_idx = np.mod(self.ant_submode,2) - 1  # -1 and 0
            assert(tx_ant_idx <= 0)
            assert (rx_ant_idx <= 0)
            data_preprocessed = data_preprocessed[:, :, tx_ant_idx::2, rx_ant_idx::2, :]

        # reduce, merge antenna rx/tx dimension
        data_preprocessed = data_preprocessed.reshape(data_preprocessed.shape[:2]+(-1,)+(data_preprocessed.shape[-1],))

        if self.GEN_PAPER_PLOTS:
            pl_data = 10 * np.log10(np.abs(data_preprocessed))
            self._plot_heatmap(pl_data, title + '_fullfreq', 123, 1, title + '_fullfreq.pdf')

        # reshape, remove subcarriers
        data_preprocessed = data_preprocessed[..., ::subcarrier_spacing]

        # take absolute value (magnitude)
        data_preprocessed = np.abs(data_preprocessed)
        if self.GEN_PAPER_PLOTS:
            pl_data = 10 * np.log10(data_preprocessed)
            self._plot_heatmap(pl_data, title + '_downfreq', 123, 1, title + '_downfreq.pdf')

        # normalizem i.e. substract first sample
        data_preprocessed = data_preprocessed / data_preprocessed[:, :1, ...]

        if self.GEN_PAPER_PLOTS:
            pl_data = 10 * np.log10(data_preprocessed)
            self._plot_heatmap(pl_data, title + '_norm', 123, 1, title + '_norm.pdf')

        num_subcarrier = data_preprocessed.shape[-1]
        # 2D-DFT
        data_preprocessed = np.abs(np.fft.fft2(data_preprocessed, s=(n_timestamps,num_subcarrier), axes=(1, 3)))
        # DFT shift
        data_preprocessed = np.fft.fftshift(data_preprocessed, axes=(1, 3))
        # take logarithm
        data_preprocessed = np.log10(data_preprocessed + 1)

        # crop in time domain
        data_preprocessed = self._crop_batch_dimension(data_preprocessed, self.time_window)

        return data_preprocessed


    def _get_ids_of_zero_csi_samples(self, data):
        """ CSI cleansing """
        to_be_deleted_csi_sample_ids = set()
        for f,frame in enumerate(data):
            for tx in frame:
                for rx in tx:
                    if (rx == 0).any():
                        to_be_deleted_csi_sample_ids.add(f)
        return to_be_deleted_csi_sample_ids


    def _crop_batch_dimension(self, temp_data:np.array, time_window:int):
        # cut batch dimension size, left and right
        if time_window == 0 or time_window >= int(temp_data.shape[1]):
            # batch dimension already lower
            pass
        else:
            time_1 = 1 if time_window % 2 else 0
            time_window_lower = int(temp_data.shape[1] / 2) - int(time_window / 2)
            time_window_upper = int(temp_data.shape[1] / 2) + int(time_window / 2) + time_1
            temp_data = temp_data[:, time_window_lower:time_window_upper, ...]
        return temp_data