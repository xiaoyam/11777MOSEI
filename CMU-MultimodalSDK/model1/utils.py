"""
Utility functions for loading data, models etc.
"""
import json
import pprint
import sys
import os
import pickle
import numpy as np
import torch
import h5py

def load_config(args):
    config_file = os.path.join('configs', args['config_folder'], 'config_{}.json'.format(args['config_num']))

    config = json.load(open(config_file, 'r'))

    if 'batch_size' in args:
        # command-line arg overrides config
        batch_size = args['batch_size']
        args.update(config)
        args['batch_size'] = batch_size
    else:
        args.update(config)

def print_config(args):
    """
    Pretty-print config
    """
    pprint.pprint(args, stream=sys.stderr, indent=2)

"""
Loading data
"""

def load_data():
    load_fns = {
            'mosei': load_mosei,
    }
    load_fn = load_mosei
    aud_train, aud_test, vid_train, vid_test, txt_train, txt_test, ey_tr, ey_te, txt2E, txt4E, txt6E= load_fn()

    def torchify(arr):
        return torch.tensor(arr, dtype=torch.float, device=torch.device('cuda'))

    aud_train = torchify(aud_train)
    aud_test = torchify(aud_test)
    vid_train = torchify(vid_train)
    vid_test = torchify(vid_test)
    txt_train = torchify(txt_train)
    txt_test = torchify(txt_test)
    txt2E = torchify(txt2E)
    txt4E = torchify(txt4E)
    txt6E = torchify(txt6E)
    ey_tr = torch.reshape(torchify(ey_tr), (-1, 1))
    ey_te = torch.reshape(torchify(ey_te), (-1, 1))
    return (aud_train, aud_test), (vid_train, vid_test), (txt_train, txt_test), (ey_tr, ey_te), (txt2E, txt4E, txt6E)

def load_mosei():
    f = open("../mosei_masked_data/mosei_senti_data.pkl", 'rb')
    f4 = open("../mosei_masked_data/mosei_senti_data_4E-01_mask.pkl", 'rb')
    f2 = open("../mosei_masked_data/mosei_senti_data_2E-01_mask.pkl", 'rb')
    f6 = open("../mosei_masked_data/mosei_senti_data_6E-01_mask.pkl", 'rb')
    data = pickle.load(f)
    data2 = pickle.load(f2)
    data4 = pickle.load(f4)
    data6 = pickle.load(f6)
    aud_train = data['train']['audio']
    aud_test = data['test']['audio']
    vid_train = data['train']['vision']
    vid_test = data['test']['vision']
    txt_train = data['train']['text']
    txt_test = data['test']['text']
    y_tr = data['train']['labels']
    y_te = data['test']['labels']
    aud_train[aud_train == -np.inf] = 0
    aud_test[aud_test == -np.inf] = 0
    txt2E = data2['test']['text']
    txt4E = data4['test']['text']
    txt6E = data6['test']['text']
    print("shapes", aud_train.shape, aud_test.shape, vid_train.shape, vid_test.shape, txt_train.shape,
          txt_test.shape, y_tr.shape, y_te.shape, txt2E, txt4E, txt6E)
    return aud_train, aud_test, vid_train, vid_test, txt_train, txt_test, y_tr, y_te, txt2E, txt4E, txt6E

def load_mosei1():
    # fname = 'drum'
    audio_train = "../processed/data/audio_train.h5"
    audio_valid = "../processed/data/audio_test.h5"
    video_train = "../processed/data/video_train.h5"
    video_valid = "../processed/data/video_test.h5"
    text_train = "../processed/data/text_train_emb.h5"
    text_valid = "../processed/data/text_test_emb.h5"
    ey_train = "../processed/data/y_train.h5"
    ey_valid = "../processed/data/y_test.h5"
    with h5py.File(audio_train, 'r') as f:
        aud_train = f['d1'][:]
        f.close()

    with h5py.File(audio_valid, 'r') as f:
        aud_test = f['d1'][:]
        f.close()

    with h5py.File(video_train, 'r') as f:
        vid_train = f['d1'][:]
        f.close()

    with h5py.File(video_valid, 'r') as f:
        vid_test = f['d1'][:]
        f.close()

    with h5py.File(text_train, 'r') as f:
        txt_train = f['d1'][:]
        f.close()

    with h5py.File(text_valid, 'r') as f:
        txt_test = f['d1'][:]
        f.close()

    with h5py.File(ey_train, 'r') as f:
        ey_tr = f['d1'][:]
        f.close()

    with h5py.File(ey_valid, 'r') as f:
        ey_te = f['d1'][:]
        f.close()
    print("shapes", aud_train.shape, aud_test.shape, vid_train.shape, vid_test.shape, txt_train.shape,
          txt_test.shape, ey_tr.shape, ey_te.shape)
    #print('shapes lol', train_low.shape, test_high.shape, test_low.shape, test_high.shape)
    return aud_train, aud_test, vid_train, vid_test, txt_train, txt_test, ey_tr, ey_te

