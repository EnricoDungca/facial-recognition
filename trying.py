import cv2
import face_recognition
import numpy as np
import os
import multiprocessing as mp
from Anti_Spoofing import anti_spoofing  # Import anti_spoofing function from Anti_Spoofing.py

import_queue = mp.Queue()

increment = 0

class Face_Recognition:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 320)
        self.cap.set(4, 240)
        self.known_face_encodings = []
        self.known_face_names = []
        self.datasets()
        
    def datasets(self):
        pictures_dir = r"Z:\Face_Recognition\Another try\Pictures"  # Assuming the pictures folder is in the same directory as the script
        print("Contents of pictures directory:")
        print(os.listdir(pictures_dir))
        
        for filename in os.listdir(pictures_dir):
            if filename.endswith(".jpg"):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(pictures_dir, filename)
                print(f"Processing image: {image_path}")
                image_of_person = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image_of_person)
                if face_encoding:
                    self.known_face_encodings.append(face_encoding[0])
                    self.known_face_names.append(name)
                    print(f"Face encoding added for {name}")
                else:
                    print(f"No face found in {filename}")
        
        print(f"Total known face encodings: {len(self.known_face_encodings)}")
                
    def faceMatch(self, face_encoding):
        if not self.known_face_encodings:
            return "Unknown"
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.6:
            return self.known_face_names[best_match_index]
        return "Unknown"
    
    
    def Camera(self):
        while True:
            ret, frame = self.cap.read()
            cv2.imwrite(r"Z:\Face_Recognition\Another try\Logs\Frame.jpg", frame)
            
            if not ret:
                print("Failed to grab frame")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = self.faceMatch(face_encoding)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1)
                
                # Anti-spoofing check
                is_real = anti_spoofing()
                if not is_real:
                    cv2.putText(frame, "Spoofing detected", (60, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, .8, (0, 0, 255), 1)
                    # Optionally handle spoofing detection, e.g., log, alert, etc.
            
            cv2.putText(frame, f"There are {len(face_locations)} people detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, .8, (0, 255, 255), 1)
            
            cv2.imshow("Frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        self.cap.release()
        cv2.destroyAllWindows()
        
    def main(self):
        import_queue.get(self.Camera())

if __name__ == "__main__":
    Face_Recognition().main()
