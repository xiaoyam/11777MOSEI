"""
Utility functions for loading data, models etc.
"""
import json
import pprint
import sys
import os

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
    aud_train, aud_test, vid_train, vid_test, txt_train, txt_test, ey_tr, ey_te = load_fn()

    def torchify(arr):
        return torch.tensor(arr, dtype=torch.float, device=torch.device('cuda'))

    aud_train = torchify(aud_train)
    aud_test = torchify(aud_test)
    vid_train = torchify(vid_train)
    vid_test = torchify(vid_test)
    txt_train = torchify(txt_train)
    txt_test = torchify(txt_test)
    ey_tr = torchify(ey_tr)
    ey_te = torchify(ey_te)
    return (aud_train, aud_test), (vid_train, vid_test), (txt_train, txt_test), (ey_tr, ey_te)


def load_mosei():
    # fname = 'drum'
    audio_train = "../processed/data/audio_train.h5"
    audio_valid = "../processed/data/audio_test.h5"
    video_train = "../processed/data/video_train.h5"
    video_valid = "../processed/data/video_test.h5"
    text_train = "../processed/data/text_train_emb.h5"
    text_valid = "../processed/data/text_test_emb.h5"
    ey_train = "../processed/data/ey_train.h5"
    ey_valid = "../processed/data/ey_test.h5"
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

