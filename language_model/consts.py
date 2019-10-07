
class global_consts():
    single_gpu = True
    load_model = False
    SDK_PATH = "../../../CMU-MultimodalSDK"

    save_grad = False

    dataset = "mosei_emo"
    data_path = None
    if dataset == "mosi_short":
        data_path = "../../MOSI/"
    else:
        data_path = "../../../data/"
    embed_path = "/Users/cask/Downloads/glove.840B.300d.txt"
    sentiment = "sad" # for IEMOCAP, choose from happy, angry, sad and neutral
    model_path = "../model/"

    log_path = None

    HPID = -1

    batch_size = 20

    padding_len = -1

    lr_decay = False

    def logParameters(self):
        print( "Hyperparameters:")
        for name in dir(global_consts):
            if name.find("__") == -1 and name.find("max") == -1 and name.find("min") == -1:
                print( "\t%s: %s" % (name, str(getattr(global_consts, name))))
