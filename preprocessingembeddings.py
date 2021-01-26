import imutils
from imutils import paths
import numpy as np
import cv2
import os
import time
import pickle

dataset="dataset"
embeddingfile="output/embeddings.pickle"
embedding="openface.nn4.small2.v1.t7"

caffeeprototxt="models/deploy.prototxt"
caffeemodel="models/res10_300x300_ssd_iter_140000.caffemodel"

embedder=cv2.dnn.readNetFromTorch(embedding)
detector=cv2.dnn.readNetFromCaffe(caffeeprototxt,caffeemodel)

images=list(paths.list_images(dataset))

names=[]
embeddings=[]
total=0
confident=0.5

for (i,imagepath) in enumerate(images):
    print("Processing Image {}/{}".format(i+1,len(images)))
    name=imagepath.split(os.path.sep)[-2]
    image=cv2.imread(imagepath)
    image=imutils.resize(image,width=600)
    (h,w)=image.shape[:2]
    imageblob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0),swapRB=False, crop=False)
    detector.setInput(imageblob)
    detections=detector.forward()

    if len(detections)>0:
        i=np.argmax(detections[0,0,:,2])
        confidence=detections[0,0,i,2]
        if confidence>confident:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy)=box.astype("int")
            face=image[starty:endy,startx:endx]
            (fH,fW)=image.shape[:2]
            if fW<20 or fH<20:
                continue
            faceblob=cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
            embedder.setInput(faceblob)
            outputembeddings=embedder.forward()
            names.append(name)
            embeddings.append(outputembeddings.flatten())
            total=total+1
print("Embeddings {}".format(total))
data={"embeddings":embeddings,"names":names}
file=open(embeddingfile,"wb")
file.write(pickle.dumps(data))
file.close()
             
