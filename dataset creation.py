import cv2
import time
import imutils
import os
import csv

cascade="haarcascade_frontalface_default.xml"
facedetector=cv2.CascadeClassifier(cascade)

Name=str(input("Enter your Name:"))
Rollno=int(input("Enter your Rollno:"))
Dataset="Dataset"
datapath=os.path.join(Dataset,Name)

if not os.path.isdir(datapath):
    os.mkdir(datapath)
    
details=[str(Name),str(Rollno)]
with open("studentdetails.csv","a") as csvfile:
    write=csv.writer(csvfile)
    write.writerow(details)
csvfile.close()

print("Starting Video Stream")
camera=cv2.VideoCapture(0)
time.sleep(1.0)
photocount=0

while photocount<50:
    print(photocount)
    a,frame=camera.read()
    image=imutils.resize(frame,width=400)
    detectfaces=facedetector.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

    for (x,y,w,h) in detectfaces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        files=os.path.sep.join([datapath,"{}.png".format(str(photocount).zfill(5))])
        cv2.imwrite(files,image)
        photocount=photocount+1
    cv2.imshow("Face Captured",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
    
