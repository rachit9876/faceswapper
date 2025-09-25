import streamlit as st
import cv2
import numpy as np
import dlib
from PIL import Image

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load base image once
base_img = cv2.imread("baseimg.png")

def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    return np.array([[p.x, p.y] for p in predictor(gray, faces[0]).parts()])

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