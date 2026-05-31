import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from skimage.feature import graycomatrix, graycoprops

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Image Analysis Dashboard",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Analysis Dashboard")
st.write("Upload an image to perform OCR, image processing, pixel analysis, and texture analysis.")

# -------------------------------
# OCR Path (Windows)
# -------------------------------
# Uncomment if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # -------------------------------
    # OCR Section
    # -------------------------------
    st.header("🔍 OCR Text Extraction")

    extracted_text = pytesseract.image_to_string(image)

    st.text_area(
        "Extracted Text",
        extracted_text,
        height=200
    )

    # -------------------------------
    # Image Processing
    # -------------------------------
    st.header("🛠 Image Processing")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Grayscale")
        st.image(gray, use_container_width=True)

    with col2:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        st.subheader("Blurred")
        st.image(blurred, use_container_width=True)

    with col3:
        edges = cv2.Canny(gray, 100, 200)
        st.subheader("Edge Detection")
        st.image(edges, use_container_width=True)

    # -------------------------------
    # Pixel Analysis
    # -------------------------------
    st.header("📊 Pixel Analysis")

    mean_pixel = np.mean(gray)
    max_pixel = np.max(gray)
    min_pixel = np.min(gray)
    std_pixel = np.std(gray)

    stats = pd.DataFrame({
        "Metric": [
            "Mean",
            "Maximum",
            "Minimum",
            "Standard Deviation"
        ],
        "Value": [
            round(mean_pixel, 2),
            max_pixel,
            min_pixel,
            round(std_pixel, 2)
        ]
    })

    st.dataframe(stats, use_container_width=True)

    # Histogram
    st.subheader("Pixel Intensity Histogram")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(
        gray.ravel(),
        bins=256,
        range=(0,256)
    )

    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram")

    st.pyplot(fig)

    # -------------------------------
    # Texture Analysis
    # -------------------------------
    st.header("🧠 Texture Analysis (GLCM)")

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]

    texture_df = pd.DataFrame({
        "Feature": [
            "Contrast",
            "Energy",
            "Homogeneity",
            "Correlation"
        ],
        "Value": [
            round(contrast,4),
            round(energy,4),
            round(homogeneity,4),
            round(correlation,4)
        ]
    })

    st.dataframe(texture_df, use_container_width=True)

    # -------------------------------
    # Image Information
    # -------------------------------
    st.header("📌 Image Information")

    st.write(f"Width : {image.size[0]} pixels")
    st.write(f"Height : {image.size[1]} pixels")
    st.write(f"Mode : {image.mode}")

    st.success("Analysis Completed Successfully!")