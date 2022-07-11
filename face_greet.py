import cv2
import numpy as np
import face_recognition
import os
import datetime
import pyttsx3 as pyttsx
import time


currentTime = datetime.datetime.now()
currentTime.hour

engine = pyttsx.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-40)

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)

for cur_img in myList:
    current_Img = cv2.imread(f'{path}/{cur_img}')
    images.append(current_Img)
    personName.append(os.path.splitext(cur_img)[0])
print(personName)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)
print("All Encoding Complete!!! ")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)


    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        
                
        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)


            if 5<= currentTime.hour < 12:
                result = ('Good Morning ' + name)
            elif 12 <= currentTime.hour < 18:
                result = ('Good Afternoon ' + name)
            else:
                result = ('Good Evening ' + name)
            #print(result)

            engine.say(result + ' Nice to meet you')
            time.sleep(0.3)
            engine.runAndWait()


    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()

exit()



