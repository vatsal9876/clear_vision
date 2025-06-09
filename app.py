import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

from vae_model import VAEModel  # ensure it uses @register_keras_serializable
from corruption_module import corrupt_image
from utils import compute_psnr, compute_ssim, compute_lpips

# === Constants ===
MODEL_PATH = "models/total_vae_model.keras"  # update path if different

# === Helper Functions ===
@st.cache_resource
def get_model():
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"VAEModel": VAEModel})

def preprocess_image(image_np):
    """Resize, normalize, and batch the image for model input."""
    image = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image = tf.image.resize(image, [64, 64])
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)  # Shape: (1, 64, 64, 3)
    return image

# === Streamlit UI ===
st.set_page_config(page_title="VAE Image Restoration", layout="centered")
st.title("🧼 Image Restoration with VAE")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display original
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Original Image", use_container_width=True)

    # Corrupt and show corrupted image
    corrupted_np = corrupt_image(image_np)
    st.image(corrupted_np, caption="Corrupted Image", use_container_width=True)

    # Preprocess and predict
    corrupted_input = preprocess_image(corrupted_np)
    vae = get_model()
    restored = vae.predict(corrupted_input)[0]  # remove batch dimension

    # Postprocess output
    restored_image = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    st.image(restored_image, caption="Restored Image", use_container_width=True)

    # Evaluation Metrics
    psnr = compute_psnr(image_np, restored_image)
    ssim = compute_ssim(image_np, restored_image)
    lpips = compute_lpips(image_np, restored_image)

    st.subheader("📊 Image Quality Metrics")
    st.write(f"**PSNR:** {psnr:.2f}")
    st.write(f"**SSIM:** {ssim:.4f}")
    st.write(f"**LPIPS:** {lpips:.4f}")

    # Download button
    st.download_button(
        label="📥 Download Restored Image",
        data=Image.fromarray(restored_image).tobytes(),
        file_name="restored.png",
        mime="image/png"
    )
else:
    st.info("Please upload an image to begin.")
