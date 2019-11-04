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
def normalize(tensor):
    return (tensor - torch.min(tensor, 1)[:,None]) / (torch.max(tensor, 1) - torch.min(tensor, 1))[:, None]

def optimize(aud,vid,txt,ey,aud_en,txt_en,vid_en,model,c1,c2,multi_c,mode,max_epochs, optimizers):
    device = torch.device('cuda')
    softmax = nn.Softmax()
    sigmoid = nn.Sigmoid()
    if mode == 'test':
        batch_size = aud.shape[0]
    else:
        batch_size = 100

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
    #elif mode == 'train' or mode == 'rubi':
    #    aud_en.unfreeze_hiddens()
    #    txt_en.freeze_hiddens()
    #    vid_en.unfreeze_hiddens()
    #    model.unfreeze_hiddens()
    #    c1.unfreeze_hiddens()
    #    c2.unfreeze_hiddens()
    #    multi_c.freeze_hiddens()
        
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
            if mode == 'train' or mode == 'rubi':
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
            if mode == 'rubi':
                lan_con_det = txt_context.detach()
            #print("myshape", lstm_out.shape)
            #init_hidden = torch.randn(2, time_step, 256)
            
                mask_prep, c1_output = c1(lan_con_det)
                #print(c1_output.shape)
            #print("myshape", mask_prep.shape, c1_output.shape)
                mask = sigmoid(mask_prep)
                #mask = sigmoid(mask_prep)
                lan_pred_prep, _ = c2(c1_output)
                lan_pred = lan_pred_prep
            #print((lan_pred > 2).sum()) 
            
                multi_context = torch.cat((aud_context, txt_context, vid_context),2)
                multi_seq = model(multi_context)
                multi_pred_prep, _ = multi_c(multi_seq)
             
                multi_pred = multi_pred_prep
                
                combined = multi_pred * mask
                pred = combined # / torch.sum(combined, dim = 1)[:, None]
            else:
                multi_context = torch.cat((aud_context, txt_context, vid_context),2)
                multi_seq = model(multi_context)
                multi_pred_prep, _ = multi_c(multi_seq)
                pred = multi_pred_prep
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
            
            #a = pred.int() == labels.int()
            
            if mode == 'test':
                acc_list = []
                for i in range(6):
                    
                    labels1 = np.round(labels[:, i].view(-1).cpu().numpy())
                    pred1 = np.round(pred[:, i].view(-1).cpu().numpy())
                    P_indis = np.nonzero(labels1)
                    #print('indis', P_indis, pred[:,i])
                    pred_P = pred1[P_indis]
                    labels_P = labels1[P_indis]
                    a = pred_P == labels_P
                    F_indis = np.nonzero(labels1 == 0)
                    print("labels", len(labels_P), len(labels1[F_indis]))
                    pred_F = pred1[F_indis]
                    labels_F = labels1[F_indis]
                    b = pred_F == labels_F
                    TP = np.sum(a, axis=0).astype(float)
                    TN = np.sum(b, axis=0).astype(float)
                    P = len(labels_P)
                    N = n_points - P
                    #print("see", TP, TN, P, TP * N / P + TN)
                    cur_acc = (TP * N / P + TN)/ (2 * N)
                    #print("cur_acc", cur_acc) 
                    acc_list.append(cur_acc)
                loss_ten = torch.FloatTensor(acc_list)
                
                
                
                #print((a == 0).sum(dim=0))
                #loss_ten = torch.mean(a.float(), 0)
            criterion = nn.L1Loss()
            #criterion = nn.L1Loss()
            if mode == 'train':
                loss = criterion(pred, labels)# + criterion(lan_pred, labels)
            elif mode == 'rubi':
                loss = criterion(pred, labels) + criterion(lan_pred, labels)
            elif mode == 'test':
                loss = criterion(pred, labels)
            cumu_loss += float(loss)
            if mode == 'train' or mode == 'rubi':
                loss.backward()
                for op in optimizers:
                    op.step()
        curr_time = time.time() - start_time
        avg_loss = cumu_loss / n_batches
        print("Epoch {} - Average loss: {:.6f}, Cumulative loss: {:.6f}, ({:.2f} s)".format(epoch, avg_loss, cumu_loss,curr_time))
        if mode == 'test':
            print("loss Tensor", loss_ten)
            print("howmany", (pred.int() == 0).sum(dim = 0), (labels.int() == 0).sum(dim = 0))
            return avg_loss, loss_ten 
        if epoch >= max_epochs:
            #print("Max number of epochs reached!")
            break

    

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

    optimizers = []
    model_lr = 0.0001
    aud_en_op = optim.Adam(aud_en.parameters(), lr = model_lr)
    vid_en_op = optim.Adam(vid_en.parameters(), lr = model_lr)
    txt_en_op = optim.Adam(txt_en.parameters(), lr = model_lr)
    model_op = optim.Adam(model.parameters(), lr = model_lr)
    c1_op = optim.Adam(classifier1.parameters(), lr = model_lr)
    c2_op = optim.Adam(classifier2.parameters(), lr = model_lr)
    multi_c_op = optim.Adam(multi_classi.parameters(), lr = model_lr)
    optimizers = [aud_en_op, vid_en_op, txt_en_op, model_op,multi_c_op]
    
    #print(torch.max(txt_test), torch.min(txt_test), txt_test.mean(), txt_test.std())
    #adding noise to the data
    #txt_rnd_noise =



    #optimizers = [aud_en_op, vid_en_op, txt_en_op, model_op, c1_op, c2_op, multi_c_op]
    best_loss = None
    best_ten = None
    #for i in range(100): 
    optimize(aud_train, vid_train, txt_train, ey_tr, aud_en, txt_en, vid_en, model, classifier1, classifier2, multi_classi, 'train', 100, optimizers)
    cur_loss, cur_ten = optimize(aud_test, vid_test, txt_test, ey_te, aud_en, txt_en, vid_en, model, classifier1, classifier2, multi_classi, 'test', 1, optimizers)
    #    if best_loss == None or best_loss > cur_loss:
    #        best_loss = cur_loss
    #        best_ten = cur_ten
    #print(best_loss, "haha")
    print("best result:", cur_loss, cur_ten) 
    print("All done!!")
    return

if __name__ == '__main__':
    main()
        
