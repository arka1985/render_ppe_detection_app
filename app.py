import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import torch
import time

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="PPE Detection | Dr. Arkaprabha Sau",
    page_icon="🦺",
    layout="wide"
)

# Beautiful CSS
st.markdown("""
<style>
body {
    background-color: #f6f7fb;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #0056b3;
    margin-bottom: -10px;
}

.subtitle {
    font-size: 20px;
    text-align: center;
    color: #007bff;
    margin-bottom: 25px;
}

.card {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg, #ffffff, #e9f0ff);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='title'>🦺 AI-Powered PPE Detection</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>R&D: Dr. Arkaprabha Sau, MBBS, MD (Gold Medalist), DPH, Dip. Geriatric Medicine, CCEBDM, PhD (CSE)</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CPU-ONLY MODEL LOAD
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    device = "cpu"   # force CPU
    model = YOLO("best.pt")
    model.to(device)

    # ❌ No model.half() on CPU
    # ❌ No FP16 image conversion

    return model, device

model, device = load_model()

classNames = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'Machinery', 'Vehicle'
]

# ---------------------------------------------------------
# UPLOAD UI CARD
# ---------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded = st.file_uploader("📤 Upload a Video File", type=["mp4", "avi", "mov"])
start_btn = st.button("🚀 Start Detection", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

frame_placeholder = st.empty()
stats_placeholder = st.empty()

# ---------------------------------------------------------
# VIDEO PROCESSING (CPU OPTIMIZED)
# ---------------------------------------------------------
if uploaded and start_btn:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Error reading video.")
        st.stop()

    # Reduce resolution for CPU speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    st.success("✅ Video Loaded Successfully")

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # CPU → keep normal uint8 frame
        img_input = img

        # YOLO CPU inference
        results = model(
            img_input,
            conf=0.5,
            verbose=False,
            device="cpu"
        )

        total_workers = 0
        helmet_ok = 0
        jacket_ok = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls >= len(classNames):
                    continue

                c_name = classNames[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if c_name == "Person":
                    total_workers += 1
                elif c_name == "Hardhat":
                    helmet_ok += 1
                elif c_name == "Safety Vest":
                    jacket_ok += 1

                # Colors
                if c_name in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                    color = (0, 0, 255)
                elif c_name in ["Hardhat", "Safety Vest", "Mask"]:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                # Thinner box (CPU friendly)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

                # Smaller text box
                label = f"{c_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(img, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # Beautiful dashboard
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (330, 170), (30, 30, 30), -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        cv2.putText(img, f"Total Workers: {total_workers}", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(img, f"Helmet Worn: {helmet_ok}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(img, f"Jacket Worn: {jacket_ok}", (25, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

        frame_placeholder.image(img[:, :, ::-1], width="stretch")

    cap.release()
    st.success("🎉 Detection Completed Successfully!")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>
Developed by <b>Dr. Arkaprabha Sau</b><br>
MBBS, MD (Gold Medalist), DPH, Dip. Geriatric Medicine, CCEBDM,<br>
PhD (Computer Science & Engineering)
</p>
""", unsafe_allow_html=True)
