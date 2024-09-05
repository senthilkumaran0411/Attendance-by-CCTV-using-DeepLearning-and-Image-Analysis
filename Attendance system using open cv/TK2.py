import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import cv2
from datetime import datetime
import os
import csv

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")  # Set the window size to 800x600

        self.known_face_encodings = []
        self.known_face_names = []

        self.create_widgets()
        self.video_source = 1
        self.video_capture = cv2.VideoCapture(self.video_source)

        # Load the student database
        self.load_student_database()

    def create_widgets(self):
        # Upload Image Button
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=5)

        # Train Model Button
        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=5)

        # Start Webcam Button
        self.start_webcam_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam)
        self.start_webcam_button.pack(pady=5)

        # Stop Webcam Button
        self.stop_webcam_button = tk.Button(self.root, text="Stop Webcam", command=self.stop_webcam)
        self.stop_webcam_button.pack(pady=5)

        # Status Label
        self.status_label = tk.Label(self.root, text="Status: Idle")
        self.status_label.pack(pady=5)

        # Create a Frame for the image panel
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Image Panel
        self.image_panel = tk.Label(self.image_frame)
        self.image_panel.pack(fill=tk.BOTH, expand=True)

    def load_student_database(self):
        """Load student information from the CSV file."""
        if os.path.exists('students_db.csv'):
            with open('students_db.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    name, class_section, encoding_str = row
                    encoding = np.fromstring(encoding_str[1:-1], sep=' ')
                    self.known_face_names.append(f"{name}, {class_section}")
                    self.known_face_encodings.append(encoding)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            class_section = simpledialog.askstring("Input", "Enter Class and Section (e.g., 10-A):")
            if class_section:
                self.status_label.config(text=f"Image {os.path.basename(file_path)} uploaded.")
                # Process the uploaded image
                self.process_uploaded_image(file_path, class_section)

    def process_uploaded_image(self, file_path, class_section):
        try:
            image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(image)[0]
            name = os.path.splitext(os.path.basename(file_path))[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(f"{name}, {class_section}")

            # Store the data in the CSV database
            with open('students_db.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, class_section, np.array_str(encoding)])

            messagebox.showinfo("Success", f"Image {name} added to training data.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")

    def train_model(self):
        if self.known_face_encodings:
            messagebox.showinfo("Training", "Model trained with uploaded images.")
        else:
            messagebox.showwarning("No Data", "No images uploaded for training.")

    def start_webcam(self):
        self.status_label.config(text="Status: Webcam running...")
        self.update_frame()

    def stop_webcam(self):
        self.status_label.config(text="Status: Idle")
        self.video_capture.release()
        self.image_panel.config(image='')

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                self.make_attendance_entry(name)

            # Convert the frame to ImageTk format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.image_panel.imgtk = imgtk
            self.image_panel.configure(image=imgtk)

            self.root.after(10, self.update_frame)
        else:
            self.status_label.config(text="Status: Failed to capture frame")

    def make_attendance_entry(self, name):
        try:
            with open('attendance_list.csv', 'a+', newline='') as file:
                file.seek(0)
                all_lines = file.readlines()
                attendance_list = [line.split(',')[0] for line in all_lines]

                if name not in attendance_list:
                    now = datetime.now()
                    dt_string = now.strftime('%d/%b/%Y, %H:%M:%S')
                    file.write(f'{name},{dt_string}\n')
        except Exception as e:
            print(f"Error writing to attendance list: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
