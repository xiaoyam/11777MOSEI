import sys
import math
import os
sys.path.append(os.getcwd())
import argparse
import time
import itertools
import json
import warnings
import csv

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dill as pickle

from transformer_model import Transformer, Encoder, Decoder
from vad_model import LSTM
from utils import *

def optimize(aud,vid,txt,ey,aud_en,txt_en,vid_en,model,c1,c2,multi_c,mode,max_epochs = 100):
    device = torch.device('cuda')
    softmax = nn.Softmax()
    sigmoid = nn.Sigmoid()
    if mode == 'test':
        batch_size = aud.shape[0]
    else:
        batch_size = 50

    n_points = aud.shape[0]
    time_step = aud.shape[1]
    
    if mode == 'test':
        aud_en.freeze_hiddens()
        txt_en.freeze_hiddens()
        vid_en.freeze_hiddens()
        model.freeze_hiddens()
        c1.freeze_hiddens()
        c2.freeze_hiddens()
        multi_c.freeze_hiddens()
    optimizers = []
    if mode == 'train':
        model_lr = 0.0001
        aud_en_op = optim.Adam(aud_en.parameters(), lr = model_lr)
        vid_en_op = optim.Adam(vid_en.parameters(), lr = model_lr)
        txt_en_op = optim.Adam(txt_en.parameters(), lr = model_lr)
        model_op = optim.Adam(model.parameters(), lr = model_lr)
        c1_op = optim.Adam(c1.parameters(), lr = model_lr)
        c2_op = optim.Adam(c2.parameters(), lr = model_lr)
        multi_c_op = optim.Adam(multi_c.parameters(), lr = model_lr)
        optimizers = [aud_en_op, vid_en_op, txt_en_op, model_op, c1_op, c2_op, multi_c_op]
        
    start_time = time.time()
    losses = []
    best_loss = None
    best_epoch = None
    epoch = 0
    loss_ten = None
    while True:
        if mode == 'test' and epoch >= 1:
            break
        epoch += 1
        order = np.random.permutation(n_points)
        cumu_loss = 0
        cumu_ten = torch.tensor(np.random.normal(0, 0.01, size = (1,1,1)),dtype=torch.float, requires_grad = True, device = torch.device('cuda'))
        n_batches = n_points // batch_size
        for i in range(n_batches):
            for op in optimizers:
                op.zero_grad()
            
            idxes = order[i*batch_size: (i+1) * batch_size]
            aud_data = aud[idxes]
            txt_data = txt[idxes]
            vid_data= vid[idxes]
            labels = ey[idxes]
            # Encode all 3 modalities            
            aud_context = aud_en(aud_data)
            txt_context= txt_en(txt_data)
            vid_context= vid_en(vid_data)
            # Prevent backprop
            lan_con_det = txt_context.detach()
            #print("myshape", lstm_out.shape)
            #init_hidden = torch.randn(2, time_step, 256)
            mask_prep, c1_output = c1(lan_con_det)
            #print("myshape", mask_prep.shape, c1_output.shape)
            mask = sigmoid(mask_prep)
            lan_pred_prep, _ = c2(lan_con_det)
            lan_pred = softmax(lan_pred_prep)
            
            multi_context = torch.cat((aud_context, txt_context, vid_context),2)
            multi_seq = model(multi_context)
            multi_pred_prep, _ = multi_c(multi_seq)
            
            multi_pred = softmax(multi_pred_prep)
            pred = softmax(multi_pred * mask)
            
            #a = (pred != labels).type(torch.cuda.FloatTensor)
            #a.requires_grad = True
            #b = (lan_pred != labels).type(torch.cuda.FloatTensor)           
            #b.requires_grad = True 
            #loss_ten = torch.mean(a, 0) + torch.mean(b, 0)
            #loss_ten = loss_ten.type(torch.cuda.FloatTensor)
            #if cumu_ten.shape == (1,1,1):
            #    cumu_ten = loss_ten
            #else:
            #    cumu_ten += loss_ten
            #loss = torch.sum(loss_ten)
            
            a = pred.int() == labels.int()
            b = (lan_pred.int() == labels.int())
            loss_ten = torch.mean(a.float(), 0) + torch.mean(b.float(), 0)
            criterion = nn.L1Loss()
            loss = criterion(pred, labels) + criterion(lan_pred, labels)
            cumu_loss += float(loss)
            if mode == 'train':
                loss.backward()
                for op in optimizers:
                    op.step()
        curr_time = time.time() - start_time
        avg_loss = cumu_loss / n_batches
        print("Epoch {} - Average loss: {:.6f}, Cumulative loss: {:.6f}, ({:.2f} s)".format(epoch, avg_loss, cumu_loss,curr_time))
        if mode == 'test':
            print("loss Tensor", loss_ten)
        if epoch >= max_epochs:
            print("Max number of epochs reached!", file=sys.stderr)
            break

        sys.stderr.flush()
        sys.stdout.flush()

    

def main():
    device = torch.device('cuda')
    data = load_data()
    (aud_train, aud_test), (vid_train, vid_test), (txt_train, txt_test), (ey_tr, ey_te) = data
    txt_dim = txt_train.shape[-1]
    aud_dim = aud_train.shape[-1]
    vid_dim = vid_train.shape[-1]
    print(txt_dim, aud_dim, vid_dim, "dims")
    batch = txt_train.shape[0]
    time_stps = txt_train.shape[1]
    txt_en = Encoder(txt_dim, 2, 6)
    aud_en = Encoder(aud_dim, 2, 2)
    vid_en = Encoder(vid_dim, 2, 5)
    
    model = Encoder(txt_dim+aud_dim+vid_dim, 3, 1)

    classifier1 = LSTM(txt_dim, 2, 6)
    classifier2 = LSTM(txt_dim, 2, 6)
    multi_classi = LSTM(txt_dim+aud_dim+vid_dim, 2, 6)
    
    txt_en = txt_en.to(device)
    aud_en = aud_en.to(device)
    vid_en = vid_en.to(device)
    model = model.to(device)
    classifier1 = classifier1.to(device)
    classifier2 = classifier2.to(device)
    multi_classi = multi_classi.to(device)


    optimize(aud_train, vid_train, txt_train, ey_tr, aud_en, txt_en, vid_en, model, classifier1, classifier2, multi_classi, 'train', 100)
    optimize(aud_test, vid_test, txt_test, ey_te, aud_en, txt_en, vid_en, model, classifier1, classifier2, multi_classi, 'test', 1)
    print("All done!!")
    return

if __name__ == '__main__':
    main()
        
