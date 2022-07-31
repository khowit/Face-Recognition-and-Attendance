import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "FACE RECOGNITION + ATTENDANCE/FaceRec"
images = []
className = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    className.append(os.path.splitext(cls)[0])
# print(className)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('FACE RECOGNITION + ATTENDANCE/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncoding(images)
# print(len(encodeListKnown))
print("Complete")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frameRes = cv2.resize(frame,(0,0),None,0.25,0.25)
    frameRes = cv2.cvtColor(frameRes, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(frameRes)
    encodeCurFrame = face_recognition.face_encodings(frameRes,faceCurFrame)
    
    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break 

cap.release()
cv2.destroyAllWindows()