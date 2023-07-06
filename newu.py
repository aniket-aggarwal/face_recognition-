import cv2
import face_recognition
import pickle
from datetime import datetime
import streamlit as st
import pandas as pd
import time
from streamlit.components.v1 import html
import os
import cv2
import dlib
import imutils
import streamlit as st
from imutils import face_utils
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@st.cache
def capture_face(name):
    detector = dlib.get_frontal_face_detector()
    number = 0
    frame_count = 0

    folder_name = f"C:/Users/Aniket Aggarwal/OneDrive/Desktop/imgs/imgss/{name}"
    if os.path.exists(folder_name):
        st.write("Folder exists")
    else:
        os.makedirs(folder_name)

    # Open the webcam
    camera = cv2.VideoCapture(0)

    while True:
        if frame_count % 5 == 0:
            st.write("Keyframe")
            grabbed, image = camera.read()

            if not grabbed:
                break

            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale image
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cro = image[y: y + h, x: x + w]
                out_image = cv2.resize(cro, (108, 108))

                fram = os.path.join(folder_name, f"{number}.jpg")
                number += 1
                cv2.imwrite(fram, out_image)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_count += 1
        else:
            frame_count += 1
            st.write("Redundant frame")

        if number > 51:
            break

        cv2.imshow("Output Image", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
directory="C:/Users/Aniket Aggarwal/OneDrive/Desktop/imgs/imgss"
model_save_path="trained_knn_model.clf"
# Train the KNN classifier
def train(directory="C:/Users/Aniket Aggarwal/OneDrive/Desktop/imgs/imgss", model_save_path="trained_knn_model.clf", n_neighbors=2, knn_algo='ball_tree'):
    encodings = []
    names = []

    # Loop through each person's folder in the training directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                face_image_path = os.path.join(root, file)
                name = os.path.basename(root)

                # Load the image and encode the face
                face_image = face_recognition.load_image_file(face_image_path)
                face_enc = face_recognition.face_encodings(face_image)[0]

                # Append the encoding and name to the training data
                encodings.append(face_enc)
                names.append(name)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings, names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

# Capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture Image", frame)
        if cv2.waitKey(1) == ord('q'):  # Check if 'q' key is pressed to stop capturing
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

def draw_boxes(frame, boxes, names, current_dt):
    for (top, right, bottom, left), name in zip(boxes, names):
        color = (0, 0, 255) if name == 'unknown' else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, current_dt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def start_model():
    try:
        stop_model = False
        result_placeholder = st.empty()  # Placeholder for results
        results_df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Present'])

        while not stop_model:
            image = capture_image()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_dt = now.strftime("%d-%m-%y")
            imgc = cv2.resize(image, (0, 0), None, 0.25, 0.25)
            imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)
            fasescurrent = face_recognition.face_locations(imgc)
            encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)
            results = []
            with open("trained_knn_model.clf", 'rb') as f:
                knn_clf = pickle.load(f)

            for encode_face, face_location in zip(encode_fasescurrent, fasescurrent):
                matches = knn_clf.predict([encode_face])
                name = matches[0]
                resarray = {'name': name, 'top': face_location[0], 'right': face_location[1], 'bottom': face_location[2],
                            'left': face_location[3]}
                if name == 'unknown':
                    resarray['time'] = current_time
                    resarray['date'] = current_dt
                    results.append(resarray)
                    # results_df = results_df.append({'Name': name, 'Date': current_dt, 'Time': current_time, 'Present': 'No'}, ignore_index=True)
                else:
                    resarray['time'] = current_time
                    resarray['date'] = current_dt
                    if name not in results_df['Name']:
                        results.append(resarray)
                        results_df = results_df.append({'Name': name, 'Date': current_dt, 'Time': current_time, 'Status': 'Marked'}, ignore_index=True)
            
            result_placeholder.json({ 'results': results_df})

            draw_boxes(image, fasescurrent, [result['name'] for result in results], current_dt)
            st.image(image, channels="RGB")

            if st.button("Close Model", key=f'close_button_{current_dt}_{current_time}') or cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                stop_model = True
                break

            time.sleep(2)

        st.dataframe(results_df)

    except Exception as e:
        result_placeholder.json({'success': False, 'error': str(e)})

import streamlit as st

def main():
    st.set_page_config(
        layout="wide",
        page_title="Face Recognition App",
        page_icon=":smiley:",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )  # Set layout to wide and customize page title and icon

    st.title("Face Recognition App")

    css = '''
        .stApp {
            background: url('https://wallpapercave.com/w/wp10281416.jpg');
            background-size: cover;
            background-position: center;
            text-align: left;
        }

        .stApp > header {
            background-color: transparent;
        }

        h1 {
            color: white;
            text-decoration: underline;
            text-align: center;
        }

        .button-container {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;

        }

        .button-container button {
            width: 200px;
            height: 50px;
            background-color: orange;
            color: black;
        }
    '''

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    button_container = st.empty()
    with button_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            
            st.image("https://wallpapercave.com/w/wp7106485.jpg",width=100,use_column_width=True)
            if st.button("Mark Attendance", key='Mark Attendance',use_container_width=True):
                start_model()
        with col2:
            st.image("https://wallpapercave.com/w/wp4578479.jpg",width=400,use_column_width=True)
            if st.button("Register face", key='register_face', use_container_width=True):
                name = st.text_input('Name')
                if name:
                    capture_face(name)

            
        with col3:
            st.image("https://wallpapercave.com/wp/wp3205403.jpg",width=400,use_column_width=True)
            if st.button("Train the algo", key='train_algorithm',use_container_width=True):
                train()


if __name__ == '__main__':
    main()

