import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

def initialize_firebase():
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://face-attendance-d907a-default-rtdb.firebaseio.com/",
        'storageBucket': "face-attendance-d907a.appspot.com"
    })
    return storage.bucket()

def get_student_info(id):
    studentInfo = db.reference(f'Students/{id}').get()
    if studentInfo:
        return studentInfo
    else:
        print(f"No student found for ID {id}")
        return {}

def get_student_image(bucket, id):
    blob = bucket.get_blob(f"images/{id}")
    if blob:
        array = np.frombuffer(blob.download_as_string(), np.uint8)
        imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
    else:
        print(f"No image found for ID {id}")
        imgStudent = cv2.imread('Resources/placeholder.png')  # Default to a placeholder image if not found
        if imgStudent is None:
            print("Placeholder image not found.")
            imgStudent = np.zeros((216, 216, 3), np.uint8)  # Default to a blank image if not found
    return imgStudent

def update_attendance(studentInfo, id):
    last_attendance_time = studentInfo.get('last_attendance_time', '1970-01-01 00:00:00')
    try:
        datetimeObject = datetime.strptime(last_attendance_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        datetimeObject = datetime(1970, 1, 1)
    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
    if secondsElapsed > 1800:  # 30 minutes
        ref = db.reference(f'Students/{id}')
        studentInfo['total_attendance'] += 1
        ref.child('total_attendance').set(studentInfo['total_attendance'])
        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return "Attended"
    else:
        return "Already Attended"

def main():
    bucket = initialize_firebase()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    imgBackground = cv2.imread('Resources/background.png')

    folderModePath = 'Resources/Modes'
    modePathList = os.listdir(folderModePath)
    imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

    print("Loading Encode File ...")
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)
    encodeListKnown, studentIds = encodeListKnownWithIds
    print("Encode File Loaded")

    modeType = 0
    counter = 0
    id = -1
    imgStudent = []

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture video frame.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgResize = cv2.resize(img, (640, 480))
        imgBackground[162:162 + 480, 55:55 + 640] = imgResize
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = studentIds[matchIndex]
                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1

            if counter != 0:
                if counter == 1:
                    studentInfo = get_student_info(id)
                    imgStudent = get_student_image(bucket, id)
                    attendance_status = update_attendance(studentInfo, id)
                    if attendance_status == "Already Attended":
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:
                    if 10 < counter < 20:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= 10:
                        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                    counter += 1

                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        studentInfo = {}  # Change from [] to {}
                        imgStudent = None  # Change from [] to None
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        else:
            modeType = 0
            counter = 0

        cv2.imshow("Face Attendance", imgBackground)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
