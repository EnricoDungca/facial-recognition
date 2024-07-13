from deepface import DeepFace

"""
img_path: Specifies the path to the image file.
enforce_detection=False: This parameter allows face extraction even if no faces are detected in the image.
anti_spoofing=True: This parameter enables anti-spoofing detection, which means it tries to distinguish between real faces and spoofed (fake) faces.
"""

def anti_spoofing():
    face_objs = DeepFace.extract_faces(
    img_path= r'Z:\Face_Recognition\Another try\Logs\Frame.jpg',
    enforce_detection=False,
    anti_spoofing=True
    )
    
    if face_objs:
        first_face_obj = face_objs[0]  # Assuming there's only one face detected
        is_real = first_face_obj.get('is_real', False)  # Get is_real value, defaulting to False if key not found
        print(f'Is real: {is_real}')
        return is_real
    else:
        print("No face detected in the image.")
        

        

anti_spoofing()
        