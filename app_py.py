# ✅ MUST BE FIRST
import streamlit as st
st.set_page_config(page_title="DR Classifier", layout="wide")

# ✅ OTHER IMPORTS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ---------- CONFIG ----------
MODEL_PATH = "EfficientNetB3_best.keras"
IMAGE_SIZE = (300, 300)
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_efficientnetb0():
    return load_model(MODEL_PATH)

model = load_efficientnetb0()

# ---------- RETINA HEURISTIC CHECK ----------
def is_likely_retinal_image(pil_img):
    img = pil_img.resize(IMAGE_SIZE)
    img_np = np.array(img)
    gray = np.mean(img_np, axis=2)
    dark_pixels = np.sum(gray < 30)
    total_pixels = gray.size
    dark_ratio = dark_pixels / total_pixels
    red_channel_mean = np.mean(img_np[..., 0])
    return dark_ratio > 0.25 and red_channel_mean > 50

# ---------- PREPROCESSING ----------
def preprocess_img_for_model(pil_img):
    img_resized = pil_img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)
    return img_expanded, img_array
    
# ---------- GRADCAM ++ ----------
def generate_gradcam(model, img_array, last_conv_layer="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        inputs = tf.cast(np.expand_dims(preprocess_input(img_array), axis=0), tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs, weights.numpy())

    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)

    # Resize and apply colormap
    cam = cv2.resize(cam, IMAGE_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = np.uint8(0.5 * heatmap + 0.5 * img_array)
    return overlay

# ---------- LIME ----------
def generate_lime_explanation(model, img_array):
    def predict_fn(images):
        images = preprocess_input(images)
        return model.predict(images)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    lime_img, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    return mark_boundaries(lime_img, mask)

# ---------- STREAMLIT UI ----------
st.title("👁️ Diabetic Retinopathy Classifier")
st.write("Upload a **retinal fundus image** to detect DR severity using EfficientNetB3 and XAI.")

uploaded_file = st.file_uploader("Upload a Retinal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=800)

    # ---------- RETINA CHECK ----------
    is_retina = is_likely_retinal_image(image)
    if not is_retina:
        st.warning("⚠️ This may not be a valid retinal image. Results may be unreliable.")

    # ---------- RUN PREDICTION ----------
    img_expanded, original_array = preprocess_img_for_model(image)
    preds = model.predict(img_expanded)[0]
    predicted_class = np.argmax(preds)
    confidence = preds[predicted_class]

    # ---------- BLOCK IF SUSPICIOUS ----------
    if not is_retina and confidence < 0.5:
        st.error("❌ This image is not confidently recognized as a retinal fundus photo. "
                 "Please upload a valid, high-quality retina image.")
        st.stop()

    # ---------- DISPLAY PREDICTION ----------
    st.subheader("Prediction")
    st.write(f"**Class:** {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")
    if confidence < 0.5:
        st.warning("⚠️ The model is not confident in this prediction. "
                   "Please ensure the image is a valid and clear retina photo.")

    # ---------- GRAD-CAM++
    st.subheader("Grad-CAM++ Explanation")
    try:
        gradcam_overlay = generate_gradcam(model, original_array)
        st.image(gradcam_overlay, caption="Grad-CAM++", width=800)
    except Exception as e:
        st.error(f"Grad-CAM++ failed: {e}")

    # ---------- LIME
    st.subheader("LIME Explanation")
    try:
        lime_img = generate_lime_explanation(model, original_array)
        st.image(lime_img, caption="LIME Explanation", width=800)
    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
