import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in landmarks.parts()])

def face_swap(base_img, source_img):
    base_landmarks = get_landmarks(base_img)
    source_landmarks = get_landmarks(source_img)
    
    if base_landmarks is None or source_landmarks is None:
        return None
    
    # Calculate the convex hull of the face
    hull = cv2.convexHull(source_landmarks.astype(np.int32))
    
    # Create a mask
    mask = np.zeros_like(source_img)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    
    # Warp the source face to the base face
    rect = cv2.boundingRect(hull)
    subdiv = cv2.Subdiv2D(rect)
    for p in source_landmarks:
        subdiv.insert((int(p[0]), int(p[1])))
    
    # Triangulate
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32).reshape(-1, 3, 2)
    
    # Warp triangles
    warped_img = np.copy(base_img)
    for tri in triangles:
        # Get corresponding triangles
        base_tri = base_landmarks[tri]
        source_tri = source_landmarks[tri]
        
        # Warp
        warp_mat = cv2.getAffineTransform(source_tri.astype(np.float32), base_tri.astype(np.float32))
        warped = cv2.warpAffine(source_img, warp_mat, (base_img.shape[1], base_img.shape[0]))
        
        # Blend
        mask_tri = np.zeros_like(mask)
        cv2.fillConvexPoly(mask_tri, tri, (255, 255, 255))
        warped_img = cv2.seamlessClone(warped, base_img, mask_tri, (int(base_tri[0][0]), int(base_tri[0][1])), cv2.NORMAL_CLONE)
    
    return warped_img

st.title("Face Swapper")

# Load base image
base_img = cv2.imread("baseimg.png")
if base_img is None:
    st.error("Base image not found!")
    st.stop()

st.image(base_img, caption="Base Image", use_column_width=True)

uploaded_file = st.file_uploader("Upload source image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    source_img = np.array(Image.open(uploaded_file))
    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
    
    result = face_swap(base_img, source_img)
    if result is not None:
        st.image(result, caption="Swapped Image", use_column_width=True)
    else:
        st.error("Face not detected in one or both images!")