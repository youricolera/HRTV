import face_recognition
import os
import pickle

dataset_path = "dataset_faces/youri"

encodings = []

for file in os.listdir(dataset_path):
    img = face_recognition.load_image_file(os.path.join(dataset_path, file))
    faces = face_recognition.face_encodings(img)

    if len(faces) > 0:
        encodings.append(faces[0])

# sauvegarde
with open("youri_encodings.pkl", "wb") as f:
    pickle.dump(encodings, f)

print("Encodings sauvegardés")