import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Load MediaPipe models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Load base image once
base_img = cv2.imread("baseimg.png")

def get_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    h, w = img.shape[:2]
    return np.array([[int(p.x * w), int(p.y * h)] for p in landmarks.landmark[:68]])

def face_swap(src_face, dst_img):
    landmarks1 = get_landmarks(src_face)
    landmarks2 = get_landmarks(dst_img)
    if landmarks1 is None or landmarks2 is None:
        return None

    hull_index = cv2.convexHull(landmarks2, returnPoints=False)
    hull1 = [landmarks1[i[0]] for i in hull_index]
    hull2 = [landmarks2[i[0]] for i in hull_index]

    mask = np.zeros(dst_img.shape, dtype=dst_img.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull2]))
    center = (r[0] + r[2]//2, r[1] + r[3]//2)

    swapped = cv2.seamlessClone(src_face, dst_img, mask, center, cv2.NORMAL_CLONE)
    return swapped

st.title("Base Image Face Swapper")

uploaded_face = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

if uploaded_face:
    face_img = np.array(Image.open(uploaded_face).convert("RGB"))[:,:,::-1]

    result = face_swap(face_img, base_img)
    if result is not None:
        st.image(result[:,:,::-1], caption="Swapped into Base Image")
        # Optional: download button
        st.download_button("Download Result", data=cv2.imencode(".png", result)[1].tobytes(), file_name="swapped.png", mime="image/png")
    else:
        st.warning("Could not detect faces in one of the images.")