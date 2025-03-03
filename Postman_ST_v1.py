import streamlit as st
import requests
import os
import json
import pandas as pd
import time
import cv2
import numpy as np

# ✅ API Endpoint
API_URL = "https://tower-ai.onrender.com/process-image"

# ✅ Folders
RESULTS_FOLDER = "API_results"
ANNOTATED_FOLDER = "API_annotated"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# ✅ CSV File Path
CSV_FILE = os.path.join(RESULTS_FOLDER, "results.csv")

# ✅ Ensure CSV file exists with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        f.write("Image Name,Extracted Text,Total Antennas,Class Name,Bounding Box (x1,y1,x2,y2),Confidence Score\n")

def save_results_to_csv(image_name, extracted_text, antenna_detections):
    """Save extracted text & antenna detections separately in CSV."""
    with open(CSV_FILE, "a", newline="") as f:
        if extracted_text:
            f.write(f"{image_name},{extracted_text},,,, \n")
        else:
            f.write(f"{image_name},No text found,,,, \n")

        if antenna_detections:
            for detection in antenna_detections:
                class_name = detection["class_name"]
                bbox = detection["bbox"]
                confidence = f"{detection['confidence']:.2f}"
                f.write(f"{image_name},,{len(antenna_detections)},{class_name},{bbox},{confidence}\n")
        else:
            f.write(f"{image_name},,0,No detection,N/A,N/A\n")

def draw_bounding_boxes(image_path, antenna_detections):
    """Draw bounding boxes with improved visibility on the image."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width, _ = image.shape
    box_thickness = max(3, width // 300)  # Thicker bounding box
    font_scale = max(0.6, width / 1200)  # Scale font based on image size
    text_thickness = max(2, width // 600)  # Bold text

    for detection in antenna_detections:
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # Green box

        # Draw thicker bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

        # Label text
        label = f"{class_name} ({confidence*100:.1f}%)"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

        # Background rectangle for text
        text_x, text_y = x1, max(y1 - 10, text_height + 10)
        cv2.rectangle(image, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), (0, 0, 0), -1)
        
        # Bold, white text
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image

def process_image(file):
    """Process a single image file via API & save results."""
    st.sidebar.write(f"🔄 Processing {file.name}...")

    try:
        # Save uploaded file temporarily
        file_path = os.path.join(RESULTS_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Send Image to FastAPI
        files = {"file": open(file_path, "rb")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            # ✅ Extracted text & detections
            extracted_text = result.get("text_detections", {}).get("all_text", "").strip()
            antenna_detections = result.get("antenna_detections", [])

            # ✅ Save to CSV
            save_results_to_csv(file.name, extracted_text, antenna_detections)
            st.sidebar.success(f"✅ Processed: {file.name}")

            # ✅ Display Extracted Text
            st.subheader("📝 Extracted Text")
            st.write(extracted_text if extracted_text else "No text found")

            if antenna_detections:
                # ✅ Display Antenna Detections
                st.subheader("📡 Antenna Detections")
                df = pd.DataFrame(antenna_detections)
                st.dataframe(df)

                # ✅ Draw and Display Annotated Image
                annotated_img = draw_bounding_boxes(file_path, antenna_detections)
                if annotated_img is not None:
                    annotated_path = os.path.join(ANNOTATED_FOLDER, file.name)
                    cv2.imwrite(annotated_path, annotated_img)

                    # ✅ Show Original & Annotated Images Side by Side (Only if detections are present)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(file_path, caption="📷 Original Image", use_container_width=True)
                    with col2:
                        st.image(annotated_img, caption="🖍 Annotated Image", channels="BGR", use_container_width=True)

                    # ✅ Allow Download of Annotated Image
                    with open(annotated_path, "rb") as f:
                        st.download_button("📥 Download Annotated Image", f, file_name=f"annotated_{file.name}")

        else:
            st.sidebar.error(f"❌ API Error: {response.status_code}")

    except Exception as e:
        st.sidebar.error(f"⚠️ Error processing image: {e}")

# ✅ Streamlit UI
st.title("📡 Tower - Antenna: AI Detection")
st.write("Upload images for **text extraction & antenna detection**")

# **Sidebar for File Browsing**
st.sidebar.header("📂 File Upload")

# **Single Image Upload in Sidebar**
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    process_image(uploaded_file)

# **Batch Processing in Sidebar**
uploaded_files = st.sidebar.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        process_image(file)
        time.sleep(1)

# **CSV Download**
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "rb") as f:
        st.sidebar.download_button("📥 Download CSV Results", f, file_name="results.csv")
