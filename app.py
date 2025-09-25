import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image

# Function to swap faces
def swap_faces(base_image_path, uploaded_image):
    # Load base image
    base_img = cv2.imread(base_image_path)
    base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    
    # Load uploaded image
    uploaded_img = np.array(uploaded_image.convert('RGB'))
    uploaded_img_bgr = cv2.cvtColor(uploaded_img, cv2.COLOR_RGB2BGR)
    
    # Find face locations and landmarks
    base_face_locations = face_recognition.face_locations(base_img_rgb)
    uploaded_face_locations = face_recognition.face_locations(uploaded_img)
    
    if not base_face_locations:
        st.error("No face detected in the base image!")
        return None
    if not uploaded_face_locations:
        st.error("No face detected in the uploaded image!")
        return None
    
    # Get face landmarks (first face only)
    base_landmarks = face_recognition.face_landmarks(base_img_rgb)[0]
    uploaded_landmarks = face_recognition.face_landmarks(uploaded_img)[0]
    
    # Convert landmarks to numpy arrays
    base_points = np.array(base_landmarks.values(), dtype=np.int32)
    uploaded_points = np.array(uploaded_landmarks.values(), dtype=np.int32)
    
    # Calculate convex hull for seamless cloning
    hull_base = cv2.convexHull(base_points)
    hull_uploaded = cv2.convexHull(uploaded_points)
    
    # Find Delaunay triangulation
    rect = cv2.boundingRect(hull_base)
    dt = cv2.Subdiv2D(rect)
    dt.insert(base_points.tolist())
    triangles = dt.getTriangleList()
    
    # Process triangles
    swapped_img = base_img.copy()
    for triangle in triangles:
        # Get triangle points for base and uploaded images
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        
        triangle_base = np.array([pt1, pt2, pt3], dtype=np.int32)
        
        # Find corresponding points in uploaded image
        indices = []
        for point in triangle_base:
            idx = np.where((base_points == point).all(axis=1))[0][0]
            indices.append(idx)
        
        triangle_uploaded = np.array([
            uploaded_points[indices[0]],
            uploaded_points[indices[1]],
            uploaded_points[indices[2]]
        ], dtype=np.int32)
        
        # Warp triangles
        rect_base = cv2.boundingRect(triangle_base)
        rect_uploaded = cv2.boundingRect(triangle_uploaded)
        
        # Offset points
        offset_base = []
        offset_uploaded = []
        for i in range(3):
            offset_base.append((
                triangle_base[i][0] - rect_base[0],
                triangle_base[i][1] - rect_base[1]
            ))
            offset_uploaded.append((
                triangle_uploaded[i][0] - rect_uploaded[0],
                triangle_uploaded[i][1] - rect_uploaded[1]
            ))
        
        # Create masks
        mask = np.zeros((rect_base[3], rect_base[2], 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(offset_base), (1, 1, 1), 16)
        
        # Apply affine transformation
        img_rect = uploaded_img_bgr[
            rect_uploaded[1]:rect_uploaded[1] + rect_uploaded[3],
            rect_uploaded[0]:rect_uploaded[0] + rect_uploaded[2]
        ]
        
        warped = cv2.warpAffine(
            img_rect,
            cv2.getAffineTransform(
                np.float32(offset_uploaded),
                np.float32(offset_base)
            ),
            (rect_base[2], rect_base[3]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # Blend warped triangle with base image
        img_rect_base = swapped_img[
            rect_base[1]:rect_base[1] + rect_base[3],
            rect_base[0]:rect_base[0] + rect_base[2]
        ]
        
        img_rect_base = img_rect_base * (1 - mask) + warped * mask
        swapped_img[
            rect_base[1]:rect_base[1] + rect_base[3],
            rect_base[0]:rect_base[0] + rect_base[2]
        ] = img_rect_base
    
    # Seamless cloning for better blending
    center = tuple(np.mean(hull_base, axis=0).astype(int))
    mask = np.zeros(base_img.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_base, (255, 255, 255))
    
    swapped_img = cv2.seamlessClone(
        swapped_img,
        base_img,
        mask,
        center,
        cv2.NORMAL_CLONE
    )
    
    return cv2.cvtColor(swapped_img, cv2.COLOR_BGR2RGB)

# Streamlit UI
st.title("Face Swapping App")
st.write("Upload your face image to swap with the base image")

# Display base image
base_image_path = "baseimg.png"
try:
    base_img = Image.open(base_image_path)
    st.image(base_img, caption="Base Image", use_column_width=True)
except FileNotFoundError:
    st.error("Base image 'baseimg.png' not found! Please add it to the app directory.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload Your Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    uploaded_img = Image.open(uploaded_file)
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
    
    # Perform face swap
    with st.spinner("Swapping faces..."):
        result = swap_faces(base_image_path, uploaded_img)
    
    if result is not None:
        # Display result
        st.image(result, caption="Face Swapped Result", use_column_width=True)
        
        # Download button
        result_pil = Image.fromarray(result)
        st.download_button(
            label="Download Result",
            data=result_pil.tobytes(),
            file_name="face_swapped.png",
            mime="image/png"
        )