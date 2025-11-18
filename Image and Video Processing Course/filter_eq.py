import streamlit as st
import numpy as np
from PIL import Image
import io
import tempfile
import os
import math

# Video processing with custom filters
try:
    import cv2
except ImportError:
    st.error("The 'opencv-python' library is not installed. Please install it by running `pip install opencv-python` in your terminal.")
    st.stop()

def apply_grayscale_filter(img_array):
    """
    Convert to grayscale using luminance formula: 0.299*R + 0.587*G + 0.114*B
    This mimics human eye sensitivity to different colors.
    """
    if len(img_array.shape) == 3:
        # Luminance formula weights
        gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        return np.stack([gray, gray, gray], axis=2).astype(np.uint8)
    return img_array

def apply_brightness_filter(img_array, factor):
    """
    Brightness adjustment using linear scaling: new_pixel = old_pixel * factor
    Clamps values to valid range [0, 255]
    """
    brightened = img_array.astype(np.float32) * factor
    return np.clip(brightened, 0, 255).astype(np.uint8)

def apply_contrast_filter(img_array, factor):
    """
    Contrast adjustment using formula: new_pixel = ((old_pixel - 128) * factor) + 128
    This pivots around middle gray (128) to enhance or reduce contrast
    """
    contrasted = ((img_array.astype(np.float32) - 128) * factor) + 128
    return np.clip(contrasted, 0, 255).astype(np.uint8)

def apply_gamma_correction(img_array, gamma):
    """
    Gamma correction using power law: new_pixel = 255 * ((old_pixel/255) ^ (1/gamma))
    Gamma < 1 brightens, Gamma > 1 darkens, emphasizing different tonal ranges
    """
    normalized = img_array.astype(np.float32) / 255.0
    corrected = np.power(normalized, 1/gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)

def apply_sepia_filter(img_array):
    """
    Sepia tone using transformation matrix:
    R = 0.393*R + 0.769*G + 0.189*B
    G = 0.349*R + 0.686*G + 0.168*B
    B = 0.272*R + 0.534*G + 0.131*B
    """
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    sepia = np.dot(img_array, sepia_matrix.T)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_negative_filter(img_array):
    """
    Negative/Invert filter: new_pixel = 255 - old_pixel
    """
    return 255 - img_array

def apply_threshold_filter(img_array, threshold=128):
    """
    Binary threshold: pixels above threshold become white (255), below become black (0)
    """
    gray = apply_grayscale_filter(img_array)
    binary = np.where(gray[:, :, 0] > threshold, 255, 0)
    return np.stack([binary, binary, binary], axis=2).astype(np.uint8)

def apply_blur_filter(img_array, kernel_size=5):
    """
    Simple box blur using averaging kernel
    Each pixel becomes the average of its neighbors in a square kernel
    """
    if kernel_size <= 1:
        return img_array
    
    height, width = img_array.shape[:2]
    channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
    
    # Pad the image to handle edges
    pad = kernel_size // 2
    if channels == 1:
        padded = np.pad(img_array, ((pad, pad), (pad, pad)), mode='edge')
    else:
        padded = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    blurred = np.zeros_like(img_array, dtype=np.float32)
    
    # Apply averaging kernel
    for i in range(height):
        for j in range(width):
            if channels == 1:
                window = padded[i:i+kernel_size, j:j+kernel_size]
            else:
                window = padded[i:i+kernel_size, j:j+kernel_size, :]
            blurred[i, j] = np.mean(window, axis=(0, 1))
    
    return np.clip(blurred, 0, 255).astype(np.uint8)

def apply_sharpen_filter(img_array):
    """
    Sharpening using unsharp mask kernel:
    [ 0, -1,  0]
    [-1,  5, -1]  
    [ 0, -1,  0]
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return apply_convolution(img_array, kernel)

def apply_edge_detection(img_array):
    """
    Edge detection using Sobel operator
    Combines horizontal and vertical edge detection
    """
    # Convert to grayscale first
    gray = apply_grayscale_filter(img_array)[:, :, 0]
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply convolution
    edge_x = apply_convolution_2d(gray, sobel_x)
    edge_y = apply_convolution_2d(gray, sobel_y)
    
    # Combine edges
    edges = np.sqrt(edge_x**2 + edge_y**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return np.stack([edges, edges, edges], axis=2)

def apply_convolution(img_array, kernel):
    """
    Apply 2D convolution with given kernel
    """
    if len(img_array.shape) == 3:
        result = np.zeros_like(img_array, dtype=np.float32)
        for channel in range(img_array.shape[2]):
            result[:, :, channel] = apply_convolution_2d(img_array[:, :, channel], kernel)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = apply_convolution_2d(img_array, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

def apply_convolution_2d(img_2d, kernel):
    """
    2D convolution for single channel
    """
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    
    # Pad image
    padded = np.pad(img_2d, pad, mode='edge')
    result = np.zeros_like(img_2d, dtype=np.float32)
    
    height, width = img_2d.shape
    
    for i in range(height):
        for j in range(width):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(window * kernel)
    
    return result

def bilinear_interpolation(img_array, new_width, new_height):
    """
    Custom bilinear interpolation for resizing
    """
    old_height, old_width = img_array.shape[:2]
    channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
    
    # Scale factors
    x_scale = old_width / new_width
    y_scale = old_height / new_height
    
    if channels == 1:
        new_img = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            # Map to original coordinates
            x = j * x_scale
            y = i * y_scale
            
            # Get integer coordinates
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, old_width - 1), min(y1 + 1, old_height - 1)
            
            # Get fractional parts
            dx, dy = x - x1, y - y1
            
            if channels == 1:
                # Bilinear interpolation
                top = img_array[y1, x1] * (1 - dx) + img_array[y1, x2] * dx
                bottom = img_array[y2, x1] * (1 - dx) + img_array[y2, x2] * dx
                new_img[i, j] = top * (1 - dy) + bottom * dy
            else:
                for c in range(channels):
                    top = img_array[y1, x1, c] * (1 - dx) + img_array[y1, x2, c] * dx
                    bottom = img_array[y2, x1, c] * (1 - dx) + img_array[y2, x2, c] * dx
                    new_img[i, j, c] = top * (1 - dy) + bottom * dy
    
    return new_img.astype(np.uint8)

def apply_rotation(img_array, angle_degrees):
    """
    Custom rotation using rotation matrix and nearest neighbor interpolation
    """
    if angle_degrees == 0:
        return img_array
    
    height, width = img_array.shape[:2]
    channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
    
    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    # Calculate new dimensions
    new_width = int(abs(width * cos_a) + abs(height * sin_a))
    new_height = int(abs(width * sin_a) + abs(height * cos_a))
    
    # Centers
    cx_old, cy_old = width // 2, height // 2
    cx_new, cy_new = new_width // 2, new_height // 2
    
    if channels == 1:
        rotated = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        rotated = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            # Translate to origin
            x_new = j - cx_new
            y_new = i - cy_new
            
            # Reverse rotation
            x_old = int(x_new * cos_a + y_new * sin_a + cx_old)
            y_old = int(-x_new * sin_a + y_new * cos_a + cy_old)
            
            # Check bounds
            if 0 <= x_old < width and 0 <= y_old < height:
                rotated[i, j] = img_array[y_old, x_old]
    
    return rotated

def process_image(img, options):
    """
    Apply custom filters to an image based on options
    """
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Apply filters in order
    if options.get("negative", False):
        img_array = apply_negative_filter(img_array)
    
    if options.get("sepia", False):
        img_array = apply_sepia_filter(img_array)
    
    if options.get("grayscale", False):
        img_array = apply_grayscale_filter(img_array)
    
    if options.get("threshold", False):
        img_array = apply_threshold_filter(img_array, options.get("threshold_value", 128))
    
    if options.get("brightness", 1.0) != 1.0:
        img_array = apply_brightness_filter(img_array, options["brightness"])
    
    if options.get("contrast", 1.0) != 1.0:
        img_array = apply_contrast_filter(img_array, options["contrast"])
    
    if options.get("gamma", 1.0) != 1.0:
        img_array = apply_gamma_correction(img_array, options["gamma"])
    
    if options.get("blur", 0) > 0:
        img_array = apply_blur_filter(img_array, options["blur"])
    
    if options.get("sharpen", False):
        img_array = apply_sharpen_filter(img_array)
    
    if options.get("edge_detection", False):
        img_array = apply_edge_detection(img_array)
    
    if options.get("resize"):
        new_width, new_height = options["resize"]
        img_array = bilinear_interpolation(img_array, new_width, new_height)
    
    if options.get("flip_h", False):
        img_array = np.fliplr(img_array)
    
    if options.get("flip_v", False):
        img_array = np.flipud(img_array)
    
    if options.get("rotate", 0) != 0:
        img_array = apply_rotation(img_array, options["rotate"])
    
    # Convert back to PIL Image
    return Image.fromarray(img_array)

def process_video(video_file, options):
    """
    Apply custom filters to video frames
    """
    try:
        # Create temporary file for uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise IOError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine output dimensions
        if options.get("resize"):
            new_width, new_height = options["resize"]
        else:
            new_width, new_height = original_width, original_height

        # Create output video
        output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_temp_file, fourcc, fps, (new_width, new_height))

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply custom filters
            pil_image = Image.fromarray(frame_rgb)
            processed_pil = process_image(pil_image, options)
            
            # Convert back to BGR
            processed_frame = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
            
            # Ensure correct dimensions
            if processed_frame.shape[:2] != (new_height, new_width):
                processed_frame = cv2.resize(processed_frame, (new_width, new_height))
            
            out.write(processed_frame)

        cap.release()
        out.release()
        os.unlink(temp_path)
        return output_temp_file

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # MUST be the first Streamlit command
    st.set_page_config(page_title="Custom Filter Media Editor", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎨 Custom Filter Media Editor")
    st.markdown("Advanced image and video editing using custom mathematical filter equations")
    
    # Sidebar
    st.sidebar.header("📁 Media Upload")
    media_type = st.sidebar.radio("Select Media Type:", ("Image", "Video"))
    uploaded_file = st.sidebar.file_uploader(
        f"Upload a {media_type.lower()}", 
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
    )
    
    if uploaded_file is not None:
        # Initialize session state
        if 'resized_width' not in st.session_state:
            st.session_state.resized_width = None
        if 'resized_height' not in st.session_state:
            st.session_state.resized_height = None
        
        # Filter controls
        st.header("🔧 Custom Filters")
        
        # Basic filters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Color Effects")
            grayscale = st.checkbox("Grayscale (Luminance Formula)")
            sepia = st.checkbox("Sepia Tone")
            negative = st.checkbox("Negative/Invert")
            threshold = st.checkbox("Binary Threshold")
            if threshold:
                threshold_value = st.slider("Threshold Value", 0, 255, 128)
            else:
                threshold_value = 128
        
        with col2:
            st.subheader("Tone Adjustments")
            brightness = st.slider("Brightness (Linear Scale)", 0.1, 3.0, 1.0)
            contrast = st.slider("Contrast (Pivot at 128)", 0.1, 3.0, 1.0)
            gamma = st.slider("Gamma Correction", 0.1, 3.0, 1.0)
        
        # Advanced filters
        st.subheader("Spatial Filters")
        col3, col4 = st.columns(2)
        
        with col3:
            blur_strength = st.slider("Box Blur Kernel Size", 0, 15, 0)
            sharpen = st.checkbox("Unsharp Mask Sharpening")
        
        with col4:
            edge_detection = st.checkbox("Sobel Edge Detection")
        
        # Geometric transformations
        st.subheader("Geometric Transformations")
        
        with st.expander("Resize (Bilinear Interpolation)", expanded=False):
            with st.form("resize_form"):
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    new_width = st.number_input("Width", value=st.session_state.resized_width or 500, min_value=1)
                with col_r2:
                    new_height = st.number_input("Height", value=st.session_state.resized_height or 500, min_value=1)
                
                if st.form_submit_button("Apply Resize"):
                    st.session_state.resized_width = new_width
                    st.session_state.resized_height = new_height
                    st.experimental_rerun()
        
        col5, col6 = st.columns(2)
        with col5:
            flip_h = st.checkbox("Flip Horizontal")
            flip_v = st.checkbox("Flip Vertical")
        
        with col6:
            rotate_angle = st.slider("Rotation Angle (°)", 0.0, 360.0, 0.0, 1.0)
        
        # Build options dictionary
        options = {
            "grayscale": grayscale,
            "sepia": sepia,
            "negative": negative,
            "threshold": threshold,
            "threshold_value": threshold_value,
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma,
            "blur": blur_strength,
            "sharpen": sharpen,
            "edge_detection": edge_detection,
            "resize": (st.session_state.resized_width, st.session_state.resized_height) if st.session_state.resized_width else None,
            "flip_h": flip_h,
            "flip_v": flip_v,
            "rotate": rotate_angle
        }
        
        # Process and display
        if media_type == "Image":
            image_bytes = uploaded_file.read()
            original_image = Image.open(io.BytesIO(image_bytes))
            
            # Set default dimensions
            if st.session_state.resized_width is None:
                st.session_state.resized_width = original_image.width
                st.session_state.resized_height = original_image.height
            
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.subheader("Original Image")
                st.image(original_image, use_column_width=True)
            
            with col_img2:
                st.subheader("Processed Image")
                processed_image = process_image(original_image.copy(), options)
                st.image(processed_image, use_column_width=True)
            
            # Download button
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Processed Image",
                data=buf.getvalue(),
                file_name="custom_filtered_image.png",
                mime="image/png"
            )
        
        elif media_type == "Video":
            st.subheader("Original Video")
            st.video(uploaded_file)
            
            st.subheader("Processed Video")
            with st.spinner("Applying custom filters to video... This may take a moment."):
                output_path = process_video(uploaded_file, options)
            
            if output_path:
                st.video(output_path)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=file,
                        file_name="custom_filtered_video.mp4",
                        mime="video/mp4"
                    )
                os.unlink(output_path)
    
    else:
        st.info(f"👆 Please upload a {media_type.lower()} file to get started with custom filtering!")
        
        # Show filter information
        st.markdown("---")
        st.subheader("🧮 Custom Filter Equations")
        
        with st.expander("View Mathematical Formulas", expanded=False):
            st.markdown("""
            **Grayscale (Luminance):**  
            `Gray = 0.299×R + 0.587×G + 0.114×B`
            
            **Brightness:**  
            `New_Pixel = Old_Pixel × Factor`
            
            **Contrast:**  
            `New_Pixel = ((Old_Pixel - 128) × Factor) + 128`
            
            **Gamma Correction:**  
            `New_Pixel = 255 × ((Old_Pixel/255)^(1/Gamma))`
            
            **Sepia Transformation Matrix:**
            ```
            R = 0.393×R + 0.769×G + 0.189×B
            G = 0.349×R + 0.686×G + 0.168×B  
            B = 0.272×R + 0.534×G + 0.131×B
            ```
            
            **Negative/Invert:**  
            `New_Pixel = 255 - Old_Pixel`
            
            **Box Blur:**  
            Each pixel = average of surrounding pixels in kernel
            
            **Sobel Edge Detection:**  
            Uses convolution with Sobel operators for horizontal/vertical edges
            """)

if __name__ == "__main__":
    main()