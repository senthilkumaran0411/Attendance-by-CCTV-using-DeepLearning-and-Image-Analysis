import streamlit as st
import face_recognition
import numpy as np
import cv2
import tempfile
from datetime import datetime
from PIL import Image

# Helper functions
def load_and_encode_images(image_files):
    encodings = []
    names = []
    for image_file in image_files:
        image = face_recognition.load_image_file(image_file)
        encoding = face_recognition.face_encodings(image)[0]
        encodings.append(encoding)
        names.append(image_file.name.split('.')[0])
    return encodings, names

def make_attendance_entry(name):
    try:
        with open('attendance_list.csv', 'a+') as FILE:
            FILE.seek(0)
            all_lines = FILE.readlines()
            attendance_list = [line.split(',')[0] for line in all_lines]

            if name not in attendance_list:
                now = datetime.now()
                dt_string = now.strftime('%d/%b/%Y, %H:%M:%S')
                FILE.write(f'{name},{dt_string}\n')
    except Exception as e:
        st.error(f"Error writing to attendance list: {e}")

# Streamlit UI
st.title('Face Recognition with Streamlit')

# Upload images for training
uploaded_files = st.file_uploader("Choose images for training", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    st.write("Images uploaded successfully.")
    with st.spinner('Encoding images...'):
        known_face_encodings, known_face_names = load_and_encode_images(uploaded_files)
    st.success("Training complete!")

# Test with webcam
if st.button('Start Webcam'):
    st.write("Starting webcam...")

    # Create a temporary file to store the webcam feed
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Log attendance
            make_attendance_entry(name)

        # Display the resulting image
        stframe.image(frame, channels='BGR', use_column_width=True)

        if st.button('Stop Webcam'):
            break

    video_capture.release()
    stframe.empty()
    st.success("Webcam stopped.")

# Ensure the app is running
if __name__ == "__main__":
    st.write("Running Streamlit app...")
