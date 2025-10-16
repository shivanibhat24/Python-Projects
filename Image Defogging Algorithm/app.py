import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="Image Defogging Algorithm", layout="wide")

class DefoggingAlgorithm:
    def __init__(self, window_size=15, w=0.95):
        self.window_size = window_size
        self.w = w
    
    def dark_channel_prior(self, image):
        """Compute dark channel prior of the image"""
        if len(image.shape) == 3:
            min_channel = np.min(image, axis=2)
        else:
            min_channel = image
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.window_size, self.window_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def estimate_atmospheric_light(self, image, dark_channel, percentile=0.1):
        """Estimate atmospheric light from dark channel"""
        h, w = dark_channel.shape
        flat_dark = dark_channel.flatten()
        num_pixels = int(h * w * percentile)
        
        indices = np.argsort(flat_dark)[-num_pixels:]
        flat_image = image.reshape(-1, image.shape[2])
        
        atmospheric_light = np.zeros(3)
        for c in range(3):
            atmospheric_light[c] = np.max(flat_image[indices, c])
        
        return atmospheric_light
    
    def estimate_transmission(self, image, atmospheric_light):
        """Estimate transmission map"""
        normalized = image.astype(np.float32) / atmospheric_light
        dark_channel = self.dark_channel_prior(normalized)
        transmission = 1 - self.w * dark_channel
        transmission = np.clip(transmission, 0.1, 1.0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.window_size, self.window_size))
        transmission = cv2.morphologyEx(transmission, cv2.MORPH_OPEN, kernel)
        transmission = cv2.GaussianBlur(transmission, (5, 5), 0)
        
        return transmission
    
    def dark_channel_defogging(self, image):
        """Dark channel prior defogging algorithm"""
        image_float = image.astype(np.float32)
        
        dark_channel = self.dark_channel_prior(image_float)
        atmospheric_light = self.estimate_atmospheric_light(image_float, dark_channel)
        transmission = self.estimate_transmission(image_float, atmospheric_light)
        
        dehazed = np.zeros_like(image_float)
        for c in range(3):
            dehazed[:, :, c] = (image_float[:, :, c] - atmospheric_light[c] * (1 - transmission)) / (transmission + 1e-8)
        
        dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)
        return dehazed
    
    def single_scale_retinex(self, image, sigma=50):
        """Single scale Retinex algorithm"""
        image_float = image.astype(np.float32)
        
        retinex = np.zeros_like(image_float)
        for c in range(3):
            channel = image_float[:, :, c]
            gaussian = gaussian_filter(channel, sigma=sigma)
            retinex[:, :, c] = np.log(channel + 1) - np.log(gaussian + 1)
        
        retinex = np.clip(retinex, -5, 5)
        retinex = ((retinex + 5) / 10 * 255).astype(np.uint8)
        return retinex
    
    def proposed_algorithm(self, image):
        """Combined algorithm: Retinex first, then dark channel defogging"""
        # Step 1: Apply Retinex for color recovery
        retinex_result = self.single_scale_retinex(image, sigma=40)
        
        # Step 2: Apply dark channel defogging
        dehazed = self.dark_channel_defogging(retinex_result)
        
        return dehazed, retinex_result
    
    def homomorphic_filtering(self, image):
        """Homomorphic filtering algorithm"""
        image_float = image.astype(np.float32) + 1
        
        filtered = np.zeros_like(image_float)
        for c in range(3):
            channel = image_float[:, :, c]
            log_img = np.log(channel)
            gaussian = gaussian_filter(log_img, sigma=30)
            filtered[:, :, c] = log_img - gaussian
        
        result = np.exp(filtered) - 1
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def global_histogram_equalization(self, image):
        """Global histogram equalization algorithm"""
        result = image.copy()
        for c in range(3):
            result[:, :, c] = cv2.equalizeHist(image[:, :, c])
        return result

def calculate_metrics(original, processed):
    """Calculate information entropy, average gradient, MSE, and PSNR"""
    # Information entropy
    hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Average gradient
    gx = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
    avg_gradient = np.mean(np.sqrt(gx**2 + gy**2))
    
    # MSE
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32))**2)
    
    # PSNR
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    
    return entropy, avg_gradient, mse, psnr

st.title("🌫️ Image Defogging Algorithm")
st.markdown("Based on research by Zhiqi Cheng and Shuhua Liu - Shenyang Ligong University")

with st.sidebar:
    st.header("⚙️ Configuration")
    window_size = st.slider("Window Size for Dark Channel", 5, 25, 15, step=2)
    w = st.slider("Weight Parameter (w)", 0.8, 0.99, 0.95, step=0.01)
    percentile = st.slider("Percentile for Atmospheric Light", 0.01, 0.5, 0.1, step=0.01)

uploaded_file = st.file_uploader("Upload a foggy image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    defogging = DefoggingAlgorithm(window_size=window_size, w=w)
    
    # Processing
    with st.spinner("Processing image..."):
        dark_channel_result = defogging.dark_channel_defogging(image_cv)
        retinex_result = defogging.single_scale_retinex(image_cv)
        homomorphic_result = defogging.homomorphic_filtering(image_cv)
        histogram_result = defogging.global_histogram_equalization(image_cv)
        proposed_result, retinex_intermediate = defogging.proposed_algorithm(image_cv)
    
    st.success("Processing complete!")
    
    # Display results
    st.subheader("Visual Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("**Dark Channel Prior**")
        st.image(cv2.cvtColor(dark_channel_result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col3:
        st.markdown("**Retinex Algorithm**")
        st.image(cv2.cvtColor(retinex_result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("**Homomorphic Filtering**")
        st.image(cv2.cvtColor(homomorphic_result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col5:
        st.markdown("**Histogram Equalization**")
        st.image(cv2.cvtColor(histogram_result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col6:
        st.markdown("**Proposed Algorithm (Dark Channel + Retinex)**")
        st.image(cv2.cvtColor(proposed_result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Metrics comparison
    st.subheader("📊 Quantitative Analysis")
    
    gray_original = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    metrics_data = {
        "Algorithm": ["Dark Channel Prior", "Retinex", "Homomorphic Filtering", "Histogram Equalization", "Proposed (Ours)"],
        "Entropy": [],
        "Avg Gradient": [],
        "MSE": [],
        "PSNR": []
    }
    
    for result, name in [
        (dark_channel_result, "Dark Channel Prior"),
        (retinex_result, "Retinex"),
        (homomorphic_result, "Homomorphic Filtering"),
        (histogram_result, "Histogram Equalization"),
        (proposed_result, "Proposed (Ours)")
    ]:
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        entropy, avg_grad, mse, psnr = calculate_metrics(gray_original, gray_result)
        metrics_data["Entropy"].append(entropy)
        metrics_data["Avg Gradient"].append(avg_grad)
        metrics_data["MSE"].append(mse)
        metrics_data["PSNR"].append(psnr)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].bar(metrics_data["Algorithm"], metrics_data["Entropy"])
        axes[0, 0].set_title("Information Entropy (Higher is Better)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(metrics_data["Algorithm"], metrics_data["Avg Gradient"])
        axes[0, 1].set_title("Average Gradient (Higher is Better)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(metrics_data["Algorithm"], metrics_data["MSE"])
        axes[1, 0].set_title("Mean Square Error (Lower is Better)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(metrics_data["Algorithm"], metrics_data["PSNR"])
        axes[1, 1].set_title("Peak Signal-to-Noise Ratio (Higher is Better)")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.dataframe({
            "Algorithm": metrics_data["Algorithm"],
            "Entropy": [f"{x:.2f}" for x in metrics_data["Entropy"]],
            "Avg Gradient": [f"{x:.2f}" for x in metrics_data["Avg Gradient"]],
            "MSE": [f"{x:.2f}" for x in metrics_data["MSE"]],
            "PSNR": [f"{x:.2f}" for x in metrics_data["PSNR"]]
        }, use_container_width=True)
    
    # Download options
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        result_img = Image.fromarray(cv2.cvtColor(proposed_result, cv2.COLOR_BGR2RGB))
        st.download_button(
            "Download Proposed Result",
            value=result_img.tobytes(),
            file_name="proposed_defogged.png",
            mime="image/png"
        )
    
    with col2:
        dark_img = Image.fromarray(cv2.cvtColor(dark_channel_result, cv2.COLOR_BGR2RGB))
        st.download_button(
            "Download Dark Channel Result",
            value=dark_img.tobytes(),
            file_name="dark_channel_defogged.png",
            mime="image/png"
        )
    
    with col3:
        retinex_img = Image.fromarray(cv2.cvtColor(retinex_result, cv2.COLOR_BGR2RGB))
        st.download_button(
            "Download Retinex Result",
            value=retinex_img.tobytes(),
            file_name="retinex_enhanced.png",
            mime="image/png"
        )
