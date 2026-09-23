import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model_utils import load_model, predict_image

st.set_page_config(page_title="Animal Predictor", page_icon="🐾")
st.title("Animal Predictor")
st.write("Upload an animal photo to classify it with the saved local model.")


@st.cache_resource
def get_model():
    return load_model(PROJECT_ROOT / "models")


upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded photo", use_container_width=True)
    try:
        model, classes = get_model()
        result = predict_image(model, classes, image)
        confidence = result["confidence"]
        st.subheader(f"Prediction: {result['animal']}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.write("Scores for all classes:")
        st.bar_chart(result["all_scores"])
        if confidence < 0.60:
            st.warning("Confidence is low; treat this prediction with caution.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")