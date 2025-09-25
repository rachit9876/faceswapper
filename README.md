# Streamlit Face Swapper

This app swaps the face from a source image onto a target image using:
- MediaPipe FaceMesh (468 landmarks)
- Piecewise affine warping (Delaunay triangulation)
- OpenCV seamless cloning for natural blending

## Requirements
See `requirements.txt`. Windows wheels are available for all packages.

## Run
1. Install dependencies (in your Python environment):
```
pip install -r requirements.txt
```
2. Start the app:
```
streamlit run app.py
```
3. In the UI, upload:
   - Source face: the face to transplant
   - Target image: where to paste (optional; defaults to `baseimg.png`)

Adjust the feather radius and blend mode if edges or colors look off.

## Tips
- Use clear, frontal faces with good lighting.
- If a face is not detected, try cropping closer to the face or using higher-resolution images.
- "Mixed" clone mode can help when skin tones differ significantly.
