import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
base_img = cv2.imread("baseimg.png")

def get_landmarks(img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        return np.array([[int(p.x * img.shape[1]), int(p.y * img.shape[0])] for p in landmarks.landmark])

def get_face_mask(landmarks, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(landmarks)
    cv2.fillPoly(mask, [hull], 255)
    return cv2.GaussianBlur(mask, (15, 15), 0)

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    
    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, warp_mat, (r2[2], r2[3]))
    
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))
    
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + img2_rect * mask

def face_swap(img1, img2):
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)
    
    if landmarks1 is None or landmarks2 is None:
        return None
    
    # Use key facial points for triangulation
    face_points = [10, 151, 9, 175, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
    
    points1 = landmarks1[face_points]
    points2 = landmarks2[face_points]
    
    img1_warped = np.copy(img2)
    
    # Delaunay triangulation
    rect = (0, 0, img2.shape[1], img2.shape[0])
    dt = cv2.Subdiv2D(rect)
    
    for p in points2:
        dt.insert((int(p[0]), int(p[1])))
    
    triangles = dt.getTriangleList()
    
    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        # Find indices in points2
        ind1 = np.where((points2 == pt1).all(axis=1))[0]
        ind2 = np.where((points2 == pt2).all(axis=1))[0]
        ind3 = np.where((points2 == pt3).all(axis=1))[0]
        
        if len(ind1) > 0 and len(ind2) > 0 and len(ind3) > 0:
            t1 = [points1[ind1[0]], points1[ind2[0]], points1[ind3[0]]]
            t2 = [pt1, pt2, pt3]
            warp_triangle(img1, img1_warped, t1, t2)
    
    # Create mask and blend
    mask = get_face_mask(landmarks2, img2.shape)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    result = img2 * (1 - mask_3d) + img1_warped * mask_3d
    return result.astype(np.uint8)

st.title("Real Face Swapper")

uploaded_face = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

if uploaded_face:
    face_img = np.array(Image.open(uploaded_face).convert("RGB"))[:,:,::-1]
    
    result = face_swap(face_img, base_img)
    if result is not None:
        st.image(result[:,:,::-1], caption="Face Swapped Result")
        st.download_button("Download Result", data=cv2.imencode(".png", result)[1].tobytes(), file_name="swapped.png", mime="image/png")
    else:
        st.warning("Could not detect faces in one of the images.")