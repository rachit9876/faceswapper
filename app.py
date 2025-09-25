import streamlit as st
import cv2
import numpy as np
from PIL import Image
from typing import Optional

import mediapipe as mp
from scipy.spatial import Delaunay


# ---------- Landmark detection (MediaPipe FaceMesh) ----------
@st.cache_resource(show_spinner=False)
def get_facemesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def detect_landmarks_bgr(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return 2D pixel coords of 468 FaceMesh landmarks as float32, or None if not found."""
    h, w = img_bgr.shape[:2]
    mesh = get_facemesh()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float32)
    return pts


# ---------- Geometry helpers ----------
def triangulate(points: np.ndarray) -> np.ndarray:
    """Delaunay triangulation returning Nx3 indices of the input points."""
    # Use scipy to get stable indices directly
    tri = Delaunay(points)
    return tri.simplices.astype(np.int32)


def warp_triangle(src_img, dst_img, src_tri, dst_tri):
    # Bounding rects
    r1 = cv2.boundingRect(np.float32(src_tri))
    r2 = cv2.boundingRect(np.float32(dst_tri))

    # Offset triangle points by top-lefts
    src_offset = np.float32([[src_tri[i][0] - r1[0], src_tri[i][1] - r1[1]] for i in range(3)])
    dst_offset = np.float32([[dst_tri[i][0] - r2[0], dst_tri[i][1] - r2[1]] for i in range(3)])

    # Crop patches
    src_patch = src_img[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    if src_patch.size == 0:
        return

    # Affine warp
    M = cv2.getAffineTransform(src_offset, dst_offset)
    warped = cv2.warpAffine(src_patch, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Mask and composite into destination region
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_offset), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)

    dst_region = dst_img[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]
    np.multiply(dst_region, (1.0 - mask), out=dst_region, casting="unsafe")
    dst_region += warped * mask


def piecewise_affine_warp(src_bgr: np.ndarray, dst_bgr: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Warp source face onto destination geometry using piecewise affine triangles."""
    h, w = dst_bgr.shape[:2]
    warped = np.copy(dst_bgr)

    # Triangulate on destination points (more stable in target space)
    tris = triangulate(dst_pts)

    for i, j, k in tris:
        src_tri = np.float32([src_pts[i], src_pts[j], src_pts[k]])
        dst_tri = np.float32([dst_pts[i], dst_pts[j], dst_pts[k]])
        warp_triangle(src_bgr, warped, src_tri, dst_tri)

    return warped


def smooth_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    k = max(1, ksize | 1)  # ensure odd
    return cv2.GaussianBlur(mask, (k, k), 0)


def face_swap(src_bgr: np.ndarray, dst_bgr: np.ndarray, feather: int = 25, clone_mode: str = "normal") -> Optional[np.ndarray]:
    """Return swapped image (src face -> dst image) or None if face not found."""
    src_pts = detect_landmarks_bgr(src_bgr)
    dst_pts = detect_landmarks_bgr(dst_bgr)
    if src_pts is None or dst_pts is None:
        return None

    # Warp source appearance to destination geometry
    warped_src = piecewise_affine_warp(src_bgr, dst_bgr, src_pts, dst_pts)

    # Build mask from destination convex hull of all points
    hull = cv2.convexHull(dst_pts.astype(np.int32))
    mask = np.zeros(dst_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if feather > 0:
        mask = smooth_mask(mask, feather)

    # Seamless clone
    center = tuple(np.mean(hull.reshape(-1, 2), axis=0).astype(int))
    mode = cv2.NORMAL_CLONE if clone_mode == "normal" else cv2.MIXED_CLONE
    try:
        out = cv2.seamlessClone(warped_src, dst_bgr, mask, center, mode)
    except cv2.error:
        # Fallback linear blend
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        out = (dst_bgr.astype(np.float32) * (1 - mask3) + warped_src.astype(np.float32) * mask3).astype(np.uint8)
    return out


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Face Swapper (MediaPipe)", page_icon="🧪")
st.title("Face Swapper")
st.caption("MediaPipe FaceMesh + piecewise affine warp + seamless clone")

col1, col2 = st.columns(2)
with col1:
    up_src = st.file_uploader("Source face (the face to paste)", type=["jpg", "jpeg", "png"], key="src")
with col2:
    up_dst = st.file_uploader("Target image (where to paste)", type=["jpg", "jpeg", "png"], key="dst")

feather = st.slider("Feather (mask blur radius)", 0, 51, 25, step=2)
clone_mode = st.selectbox("Blend mode", ["normal", "mixed"], index=0, help="Try 'mixed' if colors look off")

# Load default destination if none uploaded
default_dst = cv2.imread("baseimg.png")
if default_dst is None:
    st.warning("Missing default destination image 'baseimg.png' in the project folder.")

btn = st.button("Swap face", type="primary")

def load_bgr(file) -> Optional[np.ndarray]:
    if file is None:
        return None
    try:
        arr = np.array(Image.open(file).convert("RGB"))[:, :, ::-1]
        return arr
    except Exception:
        return None

src_bgr = load_bgr(up_src)
dst_bgr = load_bgr(up_dst) if up_dst else (default_dst.copy() if default_dst is not None else None)

if btn:
    if src_bgr is None or dst_bgr is None:
        st.error("Please provide both a source face and a target image (or ensure baseimg.png exists).")
    else:
        with st.spinner("Detecting landmarks and swapping..."):
            result = face_swap(src_bgr, dst_bgr, feather=feather, clone_mode=clone_mode)
        if result is None:
            st.warning("Could not detect a face in one of the images. Try a clearer, frontal face.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB), caption="Source face", use_container_width=True)
            with c2:
                st.image(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB), caption="Target image", use_container_width=True)
            with c3:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Swapped result", use_container_width=True)

            st.download_button(
                "Download result",
                data=cv2.imencode(".png", result)[1].tobytes(),
                file_name="swapped.png",
                mime="image/png",
            )