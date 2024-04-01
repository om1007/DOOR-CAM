import cv2
import numpy as np
import face_recognition
import os
import csv
import door
import time
from datetime import datetime
from playsound import playsound 

# Define the base recording directory
recording_dir = "C:/Users/om/Desktop/2/Recording"

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

def Timings(name):
    
    now = datetime.now()
    dtString = now.strftime('%I:%M:%S %p , %m/%d/%Y')
    with open('Time.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, dtString])


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

recording = False  # Flag to track recording state
out = None  # Video writer object
last_face_seen = 0  # Time since last face was detected (for timeout)
face_timeout = 3  # Stop recording after this many seconds without a face
unknown_detected = False

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Trigger sound playback only when the button is pressed
    if cv2.waitKey(1) & 0xFF == ord('p'):  # Change 'p' to your desired button
        playsound(r"C:/Users/om/Desktop/2/alert.wav")
  
        unknown_detected = False  # Reset flag for next playback


    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        y1, x2, y2, x1 = faceLoc

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 3, y2 - 3), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            Timings(name)
            unknown_detected = False
         

        else:
            # Print "UNKNOWN" above the bounding box
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle for unknown faces
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (0, 0,255), cv2.FILLED)


            cv2.putText(img, "UNKNOWN", (x1 + 3, y2 - 3), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            if not unknown_detected:
                # door.send_mms()  # Send the SMS
                door.sendSms()  # Send the SMS
               
                unknown_detected = True 
            # Timings("UNKNOWN")
             
        
        
           

    # Check for faces and handle recording logic
    if facesCurFrame:
         for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if not matches[matchIndex]:  # Only start recording for unknown faces
                if not recording:
                    recording = True
                    out_path = os.path.join(recording_dir, f"rec_{datetime.now().strftime('%d-%m-%Y_%I-%M-%S %p')}.mp4")
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (img.shape[1], img.shape[0]))
                    print("Recording Started (Unknown Face)")
                    last_face_seen = time.time()
                    Timings("UNKNOWN")
                     
            else:  # Known face detected, stop recording if it was recording for an unknown face
                if recording:
                    recording = False
                    out.release()
                    print("Recording Stopped (Known Face)")
    else:
        # No face detected, check for timeout
        if recording and time.time() - last_face_seen > face_timeout:
            recording = False
            out.release()
            print("Recording Stopped")

    # Write frame to video if recording
    if recording:
        out.write(img)
        # Timings("UNKNOWN")  # Log timestamp even for unknown faces

    # Display the frame
    cv2.imshow('Webcam', img)
    

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
if recording:
    out.release()  # Ensure recording is stopped even on program exit with a face present
cv2.destroyAllWindows()
