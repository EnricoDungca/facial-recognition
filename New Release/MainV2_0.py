# import libraries
import cv2
import face_recognition
import numpy as np
import os 
from Anti_Spoofing import anti_spoofing  # Import anti_spoofing function from Anti_Spoofing.py
import multiprocessing as mp
import datetime as dt
run = mp.Queue()


class Face_Recognition:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 320)
        self.cap.set(4, 240)
        self.known_face_encodings: list = []
        self.known_face_names: list = []
        self.datasets()
        
    # Define datasets
    def datasets(self):
        pictures_dir: str = r"Z:\Face_Recognition\Another try\Pictures"  # Assuming the pictures folder is in the same directory as the script
        print("Contents of pictures directory:")
        print(os.listdir(pictures_dir))
        
        for filename in os.listdir(pictures_dir): # Loop through all the files in the folder
            if filename.endswith(".jpg"): # Check if the file is an image
                name: str = os.path.splitext(filename)[0] # Get the name without the extension
                image_path: str = os.path.join(pictures_dir, filename) # Get the full path of the image
                print(f"Processing image: {image_path}") # Print the name of the image
                image_of_person: np.ndarray = face_recognition.load_image_file(image_path) # Load the image
                face_encoding: list = face_recognition.face_encodings(image_of_person) # Get the face encoding
                if face_encoding: # Check if the face encoding is not empty
                    self.known_face_encodings.append(face_encoding[0]) # Append the face encoding
                    self.known_face_names.append(name) # Append the name
                    print(f"Face encoding added for {name}") # Print the name
                else:
                    print(f"No face found in {filename}") 
        
        print(f"Total known face encodings: {len(self.known_face_encodings)}") # Print the total number of face encodings
    
       
    def faceMatch(self, face_encoding):
        if not self.known_face_encodings:
            return "Unknown"
        face_distances: list = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index: int = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.6:
            return self.known_face_names[best_match_index]
        return "Unknown"
    
    
    def Camera(self):
        while True:
            ret, frame = self.cap.read() 
            cv2.imwrite(r"Z:\Face_Recognition\Another try\Logs\Frame.jpg", frame) #this is for anti spoofing frame by frame capture
            
            if not ret:
                print("Failed to grab frame")
                break
            
            rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB
            face_locations: list = face_recognition.face_locations(rgb_frame) # Get the face locations
            face_encodings: list = face_recognition.face_encodings(rgb_frame, face_locations) # Get the face encodings
            
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name: str = self.faceMatch(face_encoding) # Call faceMatch function here to get the name of the person
                color: tuple = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Set the color based on whether the person is recognized
                cv2.rectangle(frame, (left, top), (right, bottom), color, 1) # Draw a rectangle around the face
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1) # Write the name of the person
                
                # anti spoofing
                is_real = anti_spoofing() # Call anti_spoofing function
                if not is_real: # If anti_spoofing returns False
                    spoofName: str = "Spoof" if not anti_spoofing() else "not spoofed" # Set spoofName to "Spoof" or "not spoofed"
                    TextspoofColor: tuple = (0, 0, 255) if not anti_spoofing() else (0, 255, 0) # Set TextspoofColor to red or green
                    cv2.putText(frame, "Spoofing detected", (60, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, .8, (0, 0, 255), 1) # Write "Spoofing detected"
                    cv2.putText(frame, spoofName, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TextspoofColor, 2) # Write "Spoof" or "not spoofed"
                    cv2.rectangle(frame, (left, top), (right, bottom), TextspoofColor, 1) # Draw a rectangle around the face
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness=2)
                    cv2.imwrite(r"Z:\Face_Recognition\Another try\Logs\SpoofDetected" +"\\"+ dt.datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + name + ".jpg", frame)
                    
            
            cv2.putText(frame, f"There are {len(face_locations)} people detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, .8, (0, 255, 255), 1) # Write the number of people detected
            
            cv2.imshow("Frame", frame) # Display the frame
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # Wait for 'q' key to exit
                break
            
        self.cap.release()
        cv2.destroyAllWindows()
        
    def main(self):
        run.put(self.Camera())

if __name__ == "__main__":
    Face_Recognition().main()
