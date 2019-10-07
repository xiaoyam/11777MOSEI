import cv2
import os
import csv
#import keras
#from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

def main():
    with open('./Senti_neg_split.csv', mode = 'r') as idFile:
        reader = csv.reader(idFile)
        line_count = 0
        for r in reader:
            l = r[1:]
            if line_count == 0:
                folder = 'train_neg/'
            else:
                folder = 'test_neg/'
            for vidId in l:
                try:
                    if not os.path.exists(folder+vidId):
                        os.makedirs(folder+vidId)
                except OSError:
                    print ('Error: Creating directory of data')
                cam = cv2.VideoCapture("../Combined/"+vidId+ ".mp4")
                currentframe = 0
                print(folder)
                while True:
                    ret, frame = cam.read()
                    if ret:
                        name = folder+vidId+'/'+'frame'+str(currentframe)+'.jpg'
                        print('Creating...' + name)
                        cv2.imwrite(name, frame)
                        currentframe += 1
                        
                    else:
                        break
                cam.release()
            line_count += 1
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
