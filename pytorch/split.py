import csv
import h5py
import numpy as np
import torch
import json

def main():
    with open('../Labels/Senti_neg.csv', mode = 'r') as labelsFile:
        reader = csv.DictReader(labelsFile)
        line_count = 0
        #used_vid = {}
        trainLabels = {}
        #trainVids = []
        testLabels = {}
        #testVids = []
        emos = ["anger", "disgust", "fear", "happyness", "sadness", "surprise"]
        for row in reader:
            if line_count > 0:
                vidId = row["Input.VIDEO_ID"] + "_" + row["Input.CLIP"]
                if vidId not in trainLabels:
                    #used_vid.add(vidId)
                    anger = int(row["Answer.anger"])
                    disgust = int(row["Answer.disgust"])
                    fear = int(row["Answer.fear"])
                    happy = int(row["Answer.happiness"])
                    sad = int(row["Answer.sadness"])
                    surp = int(row["Answer.surprise"])
                    vals = [anger, disgust, fear, happy, sad, surp]
                    #emoTag = vals.index(max(vals))
                    print(anger, disgust, fear, happy, sad, surp, vidId)
                    #label = np.zeros(6)
                    #label[emoTag] = 1
                    label = tuple(vals) 
                    if len(trainLabels) < 680:
                        trainLabels[vidId] = label
                        #trainVids.append(vidId)
                        #trainLabels.append(label)
                    else:
                        testLabels[vidId] = label
                        #testVids.append(vidId)
                        #testLabels.append(label)
            #if line_count > 10:
            #    return
            line_count += 1
    json1 = json.dumps(trainLabels)
    json2 = json.dumps(testLabels)
    f_train = open("ey_train_neg.json", "w")
    f_train.write(json1)
    f_train.close()
    f_test = open("ey_test_neg.json", "w")
    f_test.write(json2)
    f_test.close()
    #trainLabels = np.asarray(trainLabels)
    #testLabels = np.asarray(testLabels)
    #h5f = h5py.File('sen-3Labels.h5', 'w')
    #h5f.create_dataset('ey_train', data = trainLabels)
    #h5f.create_dataset('ey_test', data = testLabels)
    #h5f.close()
    #print(trainLabels.shape, testLabels.shape)
    #print(len(trainVids), len(testVids))
    trainVids = list(trainLabels.keys())
    testVids = list(testLabels.keys())
    print(trainVids[-1])
    with open('Senti_neg_split.csv', mode = 'w') as split_file:
        split_writer = csv.writer(split_file, delimiter = ",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        split_writer.writerow(["trainVids"] + trainVids)
        split_writer.writerow(["testVids"] + testVids)
if __name__ == '__main__':
    main()
