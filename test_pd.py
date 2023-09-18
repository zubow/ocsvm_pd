from presencedetector import PresenceDetector

gen_paper_plots = False

# PD parameters
class Config:
    verbose = False

    # CSI capture settings
    csi_folder = "data/"
    csi_sampling_rate = 100 # in Hz
    csi_skip_first_n_seconds = 10
    Ntx = 2
    Nrx = 2

    # pre-processing parameters
    observed_time_ms = 128 # batch size; 128=1.28s
    downselected_subcarriers = 14 # downsampling in frequency domain
    time_window = 14
    ant_mode = 3  # 0=SISO, 1=MISO, 2=SIMO, 3=MIMO
    ant_submode = 0  # SISO: 0-3, MISO/SIMO: 0-1, MIMO: 0

    # OCSVM parameter
    kernel = "rbf"
    nu = 0.005
    gamma = 0.01

    # post-processing parameters
    post_processing_window_sz = 5

# create detector obj
pd = PresenceDetector(Config, False)

# train & test data: room + channel
ch = '36'
rooms_to_train = [s + ch for s in ['a', 'b', 'c']]
rooms_to_test = ['a' + ch]
#rooms_to_train = ['a1']
#rooms_to_test = ['a1']

# load train and test data
pd.load_data(rooms_to_train, rooms_to_test)
pd.print_data_store()

# preprocess
pd.preprocess()

# plot
if gen_paper_plots:
    batch_index = 12
    room = 'a1'
    pd.plot_empty_preprocessed(room, batch_index, antenna_num=1)
    pd.plot_occupied_preprocessed(room, batch_index, antenna_num=1)

# train with data from empty rooms only
pd.train()

# test model with unseen data + post-process
pd.predict()

# show results
pd.get_metrics()