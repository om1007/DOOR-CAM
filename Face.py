
import cv2
import numpy as np
import face_recognition
import os
import csv
import door
import time
from datetime import datetime

# import time

path = 'Images'

images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    name = os.path.splitext(cl)[0]
    classNames.append(name.split(" ")[0])  # Split by space for names like "Elon Musk"

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# csv file 

def Timings(name):

  with open('Time.csv','a+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%I:%M:%S %p , %m/%d/%Y')
            writer = csv.writer(f)
            writer.writerow([name , dtString])


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

unknown_detected = False 

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 3, y2 - 3), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            Timings(name)
            unknown_detected = False 

        else:
            # Print "UNKNOWN" above the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle for unknown faces
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (0, 0,255), cv2.FILLED)


            cv2.putText(img, "UNKNOWN", (x1 + 3, y2 - 3), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            if not unknown_detected:
                door.sendSms()  # Send the SMS
                unknown_detected = True  
            Timings("UNKNOWN")
           
            # now = datetime.now()
            # dtString = now.strftime('%H:%M:%S')
            # with open('Time.csv', 'a', newline='') as f:  # Use 'a' mode for appending
            #     writer = csv.writer(f)
            #     writer.writerow(['UNKNOWN', dtString])

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    

    cv2.imshow('Webcam', img)



cap.release()
cv2.destroyAllWindows()



