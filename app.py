import streamlit as st
import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
base_img = cv2.imread("baseimg.png")

def get_face_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None, None
    
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi)
    
    # Create key points based on face and eye detection
    points = []
    # Face corners
    points.extend([(x, y), (x+w, y), (x, y+h), (x+w, y+h)])
    # Face center and edges
    points.extend([(x+w//2, y), (x, y+h//2), (x+w, y+h//2), (x+w//2, y+h)])
    # Face quarter points
    points.extend([(x+w//4, y+h//4), (x+3*w//4, y+h//4), (x+w//4, y+3*h//4), (x+3*w//4, y+3*h//4)])
    
    # Add eye positions if detected
    for (ex, ey, ew, eh) in eyes[:2]:
        points.append((x + ex + ew//2, y + ey + eh//2))
    
    return np.array(points), (x, y, w, h)

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    
    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    if img1_rect.size > 0:
        warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
        img2_rect = cv2.warpAffine(img1_rect, warp_mat, (r2[2], r2[3]))
        
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))
        
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + img2_rect * mask

def face_swap(img1, img2):
    points1, face1 = get_face_landmarks(img1)
    points2, face2 = get_face_landmarks(img2)
    
    if points1 is None or points2 is None:
        return None
    
    # Ensure same number of points
    min_points = min(len(points1), len(points2))
    points1 = points1[:min_points]
    points2 = points2[:min_points]
    
    img1_warped = np.copy(img2)
    
    # Simple triangulation using face bounds
    x2, y2, w2, h2 = face2
    triangles = [
        [(x2, y2), (x2+w2//2, y2), (x2, y2+h2//2)],
        [(x2+w2, y2), (x2+w2//2, y2), (x2+w2, y2+h2//2)],
        [(x2, y2+h2), (x2+w2//2, y2+h2), (x2, y2+h2//2)],
        [(x2+w2, y2+h2), (x2+w2//2, y2+h2), (x2+w2, y2+h2//2)],
        [(x2+w2//2, y2), (x2+w2//2, y2+h2), (x2, y2+h2//2)],
        [(x2+w2//2, y2), (x2+w2//2, y2+h2), (x2+w2, y2+h2//2)]
    ]
    
    x1, y1, w1, h1 = face1
    src_triangles = [
        [(x1, y1), (x1+w1//2, y1), (x1, y1+h1//2)],
        [(x1+w1, y1), (x1+w1//2, y1), (x1+w1, y1+h1//2)],
        [(x1, y1+h1), (x1+w1//2, y1+h1), (x1, y1+h1//2)],
        [(x1+w1, y1+h1), (x1+w1//2, y1+h1), (x1+w1, y1+h1//2)],
        [(x1+w1//2, y1), (x1+w1//2, y1+h1), (x1, y1+h1//2)],
        [(x1+w1//2, y1), (x1+w1//2, y1+h1), (x1+w1, y1+h1//2)]
    ]
    
    for i in range(len(triangles)):
        warp_triangle(img1, img1_warped, src_triangles[i], triangles[i])
    
    # Create smooth mask
    mask = np.zeros(img2.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (x2+w2//2, y2+h2//2), (w2//2, h2//2), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    result = img2 * (1 - mask_3d) + img1_warped * mask_3d
    return result.astype(np.uint8)

st.title("Face Swapper")

uploaded_face = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

if uploaded_face:
    face_img = np.array(Image.open(uploaded_face).convert("RGB"))[:,:,::-1]
    
    result = face_swap(face_img, base_img)
    if result is not None:
        st.image(result[:,:,::-1], caption="Face Swapped Result")
        st.download_button("Download Result", data=cv2.imencode(".png", result)[1].tobytes(), file_name="swapped.png", mime="image/png")
    else:
        st.warning("Could not detect faces in one of the images.")