import face_recognition
import cv2
import numpy as np
import os
import datetime


hour = datetime.datetime.now().hour

video_capture = cv2.VideoCapture(0)

image1 = face_recognition.load_image_file(os.path.abspath("Sir1.jpeg"))
image1_face_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file(os.path.abspath("Sir2.jpeg"))
image2_face_encoding = face_recognition.face_encodings(image2)[0]

image3 = face_recognition.load_image_file(os.path.abspath("me.jpg"))
image3_face_encoding = face_recognition.face_encodings(image3)[0]

known_face_encodings = [
    image1_face_encoding,
    image2_face_encoding,
    image3_face_encoding
]
known_face_names = [
    "Sudarshan",
    "Vishal",
    "Alsafa"
]

face_locations = []
face_encodings = []
face_names = []
face_greetings = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    print(rgb_small_frame)
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown" 
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        face_greetings = []
        for face_encoding in face_encoding:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            greeting = "Good morning" if 5<=hour<12 else "Good afternoon" if hour<18 else "Good evening"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                greeting = "Good morning" if 5<=hour<12 else "Good afternoon" if hour<18 else "Good evening"
            face_greetings.append(greeting)
        
    process_this_frame = not process_this_frame
    print ("Face detected -- {}".format(face_names, face_greetings))
    for (top, right, bottom, left), name in zip(face_locations, face_names), greeting in zip(face_locations, face_greetings):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, greeting, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()