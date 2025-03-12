import insightface
import cv2
import numpy as np
import os

# Initialize InsightFace model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

def recognize_faces(image_path, fixed_width=172, fixed_height=216):
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at '{image_path}'.")
        return
    
    faces = model.get(img)
    
    if not faces:
        print("No face detected in the image.")
        return
    
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        x1 = max(0, center_x - fixed_width // 2) + 3
        y1 = max(0, center_y - fixed_height // 2) - 4
        x2 = min(img.shape[1], x1 + fixed_width)
        y2 = min(img.shape[0], y1 + fixed_height)
        
        face_crop = img[y1:y2, x1:x2]
        cv2.imwrite("face_output.jpg", face_crop)
        cv2.imwrite("face_output.png", face_crop)

if __name__ == "__main__":
    recognize_faces("first_page.png")