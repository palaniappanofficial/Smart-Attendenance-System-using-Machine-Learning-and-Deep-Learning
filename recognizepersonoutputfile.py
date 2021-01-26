import cv2
import pickle
import numpy as np
import imutils

embeddingFile="output/embeddings.pickle"
labelEncFile="output/labelencoder.pickle"
recognizerFile="output/recognizer.pickle"

prototxt="models/deploy.prototxt"
model="models/res10_300x300_ssd_iter_140000.caffemodel"
embeddingModel="openface.nn4.small2.v1.t7"

embedder=cv2.dnn.readNetFromTorch(embeddingModel)
detector=cv2.dnn.readNetFromCaffe(prototxt,model)

conf=0.5
recognizer=pickle.loads(open(recognizerFile,"rb").read())
le=pickle.loads(open(labelEncFile,"rb").read())
box=[]

cam=cv2.VideoCapture(0)
while True:
    a,frame=cam.read()
    frame=imutils.resize(frame,width=600)
    (h,w)=frame.shape[:2]
    print(h,w)
    imageBlob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0),swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections=detector.forward()
    print(detections.shape[2])
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy)=box.astype("int")
            face=frame[starty:endy,startx:endx]
            (fH,fW)=face.shape[:2]
            print(fH,fW)
            if fH<20 or fW<20:
                continue
            faceblob=cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
            embedder.setInput(faceblob)
            vec=embedder.forward()
            preds=recognizer.predict_proba(vec)[0]
            print(preds)
            j=np.argmax(preds)
            proba=preds[j]
            print(j)
            name=le.classes_[j]
            text="{}:{:.2f}%".format(name,proba*100)
            y=starty-10 if starty-10>10 else starty+10
            cv2.rectangle(frame,(startx,starty),(endx,endy),(0,0,255),2)
            cv2.putText(frame,text,(startx,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0))
     
    cv2.imshow("Face Captured",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
