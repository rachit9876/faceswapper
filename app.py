import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load base image once
base_img = cv2.imread("baseimg.png")

def get_face_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    return faces[0]

def create_face_mask(face_region, img_shape):
    x, y, w, h = face_region
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # Create elliptical mask for more natural blending
    center = (x + w//2, y + h//2)
    axes = (w//2, h//2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    # Smooth the mask edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask

def face_swap(src_face, dst_img):
    src_region = get_face_region(src_face)
    dst_region = get_face_region(dst_img)
    if src_region is None or dst_region is None:
        return None

    x1, y1, w1, h1 = src_region
    x2, y2, w2, h2 = dst_region
    
    # Extract and resize source face
    src_roi = src_face[y1:y1+h1, x1:x1+w1]
    src_resized = cv2.resize(src_roi, (w2, h2))
    
    # Create mask for seamless cloning
    mask = create_face_mask(dst_region, dst_img.shape)
    
    # Use seamless cloning for natural blending
    center = (x2 + w2//2, y2 + h2//2)
    
    try:
        # Try seamless cloning first
        result = cv2.seamlessClone(src_resized, dst_img, mask[y2:y2+h2, x2:x2+w2], center, cv2.NORMAL_CLONE)
    except:
        # Fallback to manual blending if seamless clone fails
        result = dst_img.copy()
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend the faces
        roi = result[y2:y2+h2, x2:x2+w2]
        mask_roi = mask_3d[y2:y2+h2, x2:x2+w2]
        blended = src_resized * mask_roi + roi * (1 - mask_roi)
        result[y2:y2+h2, x2:x2+w2] = blended.astype(np.uint8)
    
    return result

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