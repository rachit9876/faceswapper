import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib
import urllib.request
import bz2
import os

@st.cache_data
def download_landmark_model():
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    filename = "shape_predictor_68_face_landmarks.dat.bz2"
    
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        st.info("Downloading facial landmark model...")
        urllib.request.urlretrieve(url, filename)
        
        with bz2.BZ2File(filename, 'rb') as f:
            with open("shape_predictor_68_face_landmarks.dat", 'wb') as out:
                out.write(f.read())
        
        os.remove(filename)
        st.success("Model downloaded!")

# Download model if needed
download_landmark_model()

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
base_img = cv2.imread("baseimg.png")

def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in landmarks.parts()])

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    
    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, (r2[2], r2[3]))
    
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))
    
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + img2_rect * mask

def face_swap(img1, img2):
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)
    
    if landmarks1 is None or landmarks2 is None:
        return None
    
    img1_warped = np.copy(img2)
    hull1 = cv2.convexHull(landmarks1)
    hull2 = cv2.convexHull(landmarks2)
    
    # Delaunay triangulation
    rect = (0, 0, img2.shape[1], img2.shape[0])
    dt = cv2.Subdiv2D(rect)
    
    for p in hull2:
        dt.insert((p[0][0], p[0][1]))
    
    triangleList = dt.getTriangleList()
    
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        if cv2.pointPolygonTest(hull2, pt1, False) >= 0 and cv2.pointPolygonTest(hull2, pt2, False) >= 0 and cv2.pointPolygonTest(hull2, pt3, False) >= 0:
            ind1 = np.where((landmarks2 == pt1).all(axis=1))[0]
            ind2 = np.where((landmarks2 == pt2).all(axis=1))[0]
            ind3 = np.where((landmarks2 == pt3).all(axis=1))[0]
            
            if len(ind1) > 0 and len(ind2) > 0 and len(ind3) > 0:
                t1 = [landmarks1[ind1[0]], landmarks1[ind2[0]], landmarks1[ind3[0]]]
                t2 = [pt1, pt2, pt3]
                warp_triangle(img1, img1_warped, t1, t2)
    
    # Seamless cloning
    hull8U = [(hull2[i][0][0], hull2[i][0][1]) for i in range(len(hull2))]
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillPoly(mask, [np.array(hull8U, dtype=np.int32)], (255, 255, 255))
    
    r = cv2.boundingRect(hull2)
    center = (r[0] + int(r[2]/2), r[1] + int(r[3]/2))
    
    return cv2.seamlessClone(np.uint8(img1_warped), img2, mask, center, cv2.NORMAL_CLONE)

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