import os
import h5py
import json
import numpy as np

def main():
    labels = open("neg_labels.txt", 'w')
    feat_file = open("X_neg_feat.txt", 'a')
    id_file = open("train_neg.txt", 'r')
    ids = id_file.readlines()
    ey = json.load(open("ey_train_neg.json", 'r'))
    pos_file = h5py.File("X_neg.h5", 'r')
    X = pos_file["X_train"][:]
    (n_points, _, dim) = X.shape
    for i in range(n_points):
        #for j in range(0, 60, 30):
        feat = X[i, 0,...]
        np.savetxt(feat_file, feat, fmt="%.5f", delimiter=' ', newline=' ')
        feat_file.write('\n')
            #feat_file.write(feat)
            #feat_file.write('\n')
        emo_label = ey[ids[i][:-1]]
        emo_tag = emo_label.index(max(emo_label))
        labels.write(str(emo_tag))
        labels.write('\n')
    feat_file.close()
    pos_file.close()
    labels.close()
        

if __name__ == '__main__':
    main()
