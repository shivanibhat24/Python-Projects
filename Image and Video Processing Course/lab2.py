import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
import io

# Set page config
st.set_page_config(
    page_title="Advanced Image Processing Suite",
    page_icon="🖼️",
    layout="wide"
)

class ImageProcessor:
    """Advanced Image Processing Suite with multiple enhancement and filtering techniques"""
    
    @staticmethod
    def contrast_stretching(image, r1=0, s1=0, r2=255, s2=255):
        """Apply contrast stretching to enhance image contrast"""
        # Ensure input parameters are in correct order
        if r1 >= r2:
            r2 = r1 + 1
        if s1 >= s2:
            s2 = s1 + 1
            
        # Create lookup table for contrast stretching
        lut = np.zeros(256, dtype=np.uint8)
        
        # First segment: 0 to r1
        if r1 > 0:
            lut[0:r1+1] = (s1 / r1) * np.arange(0, r1+1)
        
        # Second segment: r1 to r2
        if r2 > r1:
            slope = (s2 - s1) / (r2 - r1)
            lut[r1:r2+1] = s1 + slope * (np.arange(r1, r2+1) - r1)
        
        # Third segment: r2 to 255
        if r2 < 255:
            slope = (255 - s2) / (255 - r2)
            lut[r2:256] = s2 + slope * (np.arange(r2, 256) - r2)
        
        # Apply lookup table
        return cv2.LUT(image, lut)
    
    @staticmethod
    def log_transformation(image, c=1):
        """Apply logarithmic transformation for dynamic range compression"""
        # Convert to float to avoid overflow
        image_float = image.astype(np.float32)
        # Apply log transformation: s = c * log(1 + r)
        transformed = c * np.log(1 + image_float)
        # Normalize to 0-255 range
        transformed = np.uint8(255 * transformed / np.max(transformed))
        return transformed
    
    @staticmethod
    def power_law_transformation(image, gamma=1.0, c=1):
        """Apply power-law (gamma) transformation for gamma correction"""
        # Normalize image to 0-1 range
        normalized = image.astype(np.float32) / 255.0
        # Apply power law transformation: s = c * r^gamma
        transformed = c * np.power(normalized, gamma)
        # Clip values and convert back to uint8
        transformed = np.clip(transformed, 0, 1)
        return np.uint8(transformed * 255)
    
    @staticmethod
    def bit_plane_slicing(image, bit_plane=7, reconstruct=True):
        """Extract specific bit plane and optionally reconstruct higher order planes"""
        if reconstruct:
            # Reconstruct image using bit planes from 'bit_plane' to 7
            reconstructed = np.zeros_like(image)
            for i in range(bit_plane, 8):
                bit_img = (image >> i) & 1
                reconstructed += bit_img * (2 ** i)
            return reconstructed
        else:
            # Extract only the specified bit plane
            bit_img = (image >> bit_plane) & 1
            return bit_img * 255
    
    @staticmethod
    def histogram_equalization(image):
        """Apply histogram equalization to improve contrast"""
        if len(image.shape) == 3:
            # For color images, apply to each channel
            img_eq = np.zeros_like(image)
            for i in range(3):
                img_eq[:,:,i] = cv2.equalizeHist(image[:,:,i])
            return img_eq
        else:
            # For grayscale images
            return cv2.equalizeHist(image)
    
    @staticmethod
    def intensity_level_slicing(image, low_thresh=100, high_thresh=200, highlight_value=255, preserve_bg=True):
        """Apply intensity level slicing to highlight specific intensity ranges"""
        result = image.copy()
        
        if preserve_bg:
            # Preserve background, highlight range
            mask = (image >= low_thresh) & (image <= high_thresh)
            result[mask] = highlight_value
        else:
            # Binary slicing
            result = np.where((image >= low_thresh) & (image <= high_thresh), 
                            highlight_value, 0).astype(np.uint8)
        
        return result
    
    @staticmethod
    def mean_filter(image, kernel_size=3):
        """Apply mean filter for noise reduction"""
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def gaussian_filter_custom(image, sigma=1.0):
        """Apply Gaussian filter for noise reduction"""
        if len(image.shape) == 3:
            # For color images, apply to each channel
            filtered = np.zeros_like(image)
            for i in range(3):
                filtered[:,:,i] = gaussian_filter(image[:,:,i], sigma=sigma)
            return filtered.astype(np.uint8)
        else:
            return gaussian_filter(image, sigma=sigma).astype(np.uint8)
    
    @staticmethod
    def median_filter_custom(image, kernel_size=3):
        """Apply median filter for salt-and-pepper noise reduction"""
        if len(image.shape) == 3:
            # For color images, apply to each channel
            filtered = np.zeros_like(image)
            for i in range(3):
                filtered[:,:,i] = median_filter(image[:,:,i], size=kernel_size)
            return filtered.astype(np.uint8)
        else:
            return median_filter(image, size=kernel_size).astype(np.uint8)
    
    @staticmethod
    def laplacian_filter(image):
        """Apply Laplacian filter for edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Convert back to uint8
        laplacian = np.uint8(np.absolute(laplacian))
        return laplacian
    
    @staticmethod
    def high_pass_filter(image, kernel_size=3):
        """Apply high pass filter for edge enhancement"""
        # Create high pass kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        
        if len(image.shape) == 3:
            # For color images, convert to grayscale first
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            filtered = cv2.filter2D(gray, -1, kernel)
        else:
            filtered = cv2.filter2D(image, -1, kernel)
        
        # Normalize to 0-255 range
        filtered = np.clip(filtered, 0, 255)
        return filtered.astype(np.uint8)
    
    @staticmethod
    def low_pass_filter(image, cutoff_freq=30):
        """Apply low pass filter using Gaussian blur"""
        # Convert cutoff frequency to sigma for Gaussian blur
        sigma = cutoff_freq / 6.0
        
        if len(image.shape) == 3:
            filtered = np.zeros_like(image)
            for i in range(3):
                filtered[:,:,i] = gaussian_filter(image[:,:,i], sigma=sigma)
            return filtered.astype(np.uint8)
        else:
            return gaussian_filter(image, sigma=sigma).astype(np.uint8)

def load_image(uploaded_file):
    """Load and convert uploaded image to numpy array"""
    if uploaded_file is not None:
        # Convert PIL Image to numpy array
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary
        if image.mode != 'RGB' and image.mode != 'L':
            image = image.convert('RGB')
        return np.array(image)
    return None

def display_images(original, processed, title="Processed Image"):
    """Display original and processed images side by side"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original, use_column_width=True)
    
    with col2:
        st.subheader(title)
        st.image(processed, use_column_width=True)

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title(" IVP Experiment")
    st.markdown("""
    A comprehensive image processing application with enhancement and filtering techniques.
    Upload an image and select processing methods to enhance and analyze your images.
    """)
    
    # Sidebar for image upload
    st.sidebar.header(" Image Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Load image
        image = load_image(uploaded_file)
        
        # Display original image info
        st.sidebar.success(f"Image loaded successfully!")
        st.sidebar.info(f"Image shape: {image.shape}")
        
        # Processing method selection
        st.sidebar.header("🔧 Processing Methods")
        
        processing_method = st.sidebar.selectbox(
            "Select Processing Method",
            [
                "Original",
                "Contrast Stretching", 
                "Logarithmic Transformation",
                "Power Law Transformation",
                "Bit-Level Slicing",
                "Histogram Equalization",
                "Intensity Level Slicing",
                "Mean Filter",
                "Gaussian Filter",
                "Median Filter",
                "Laplacian Filter",
                "High Pass Filter",
                "Low Pass Filter"
            ]
        )
        
        # Initialize processor
        processor = ImageProcessor()
        processed_image = image.copy()
        
        # Apply selected processing method with parameters
        if processing_method == "Original":
            processed_image = image
            
        elif processing_method == "Contrast Stretching":
            st.sidebar.subheader("Contrast Stretching Parameters")
            r1 = st.sidebar.slider("Input Low (r1)", 0, 255, 50)
            s1 = st.sidebar.slider("Output Low (s1)", 0, 255, 0)
            r2 = st.sidebar.slider("Input High (r2)", 0, 255, 200)
            s2 = st.sidebar.slider("Output High (s2)", 0, 255, 255)
            
            if len(image.shape) == 3:
                processed_image = np.zeros_like(image)
                for i in range(3):
                    processed_image[:,:,i] = processor.contrast_stretching(image[:,:,i], r1, s1, r2, s2)
            else:
                processed_image = processor.contrast_stretching(image, r1, s1, r2, s2)
                
        elif processing_method == "Logarithmic Transformation":
            st.sidebar.subheader("Log Transformation Parameters")
            c_value = st.sidebar.slider("Scaling constant (c)", 0.1, 5.0, 1.0, 0.1)
            processed_image = processor.log_transformation(image, c_value)
            
        elif processing_method == "Power Law Transformation":
            st.sidebar.subheader("Power Law Parameters")
            gamma = st.sidebar.slider("Gamma (γ)", 0.1, 3.0, 1.0, 0.1)
            c_value = st.sidebar.slider("Scaling constant (c)", 0.1, 2.0, 1.0, 0.1)
            processed_image = processor.power_law_transformation(image, gamma, c_value)
            
        elif processing_method == "Bit-Level Slicing":
            st.sidebar.subheader("Bit-Level Slicing Parameters")
            bit_plane = st.sidebar.slider("Bit Plane", 0, 7, 7)
            reconstruct = st.sidebar.checkbox("Reconstruct Higher Order Planes", True)
            
            if len(image.shape) == 3:
                # Convert to grayscale for bit-plane slicing
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                processed_image = processor.bit_plane_slicing(gray_image, bit_plane, reconstruct)
            else:
                processed_image = processor.bit_plane_slicing(image, bit_plane, reconstruct)
                
        elif processing_method == "Histogram Equalization":
            processed_image = processor.histogram_equalization(image)
            
        elif processing_method == "Intensity Level Slicing":
            st.sidebar.subheader("Intensity Level Slicing Parameters")
            low_thresh = st.sidebar.slider("Low Threshold", 0, 255, 100)
            high_thresh = st.sidebar.slider("High Threshold", 0, 255, 200)
            highlight_value = st.sidebar.slider("Highlight Value", 0, 255, 255)
            preserve_bg = st.sidebar.checkbox("Preserve Background", True)
            
            if len(image.shape) == 3:
                # Convert to grayscale for intensity slicing
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                processed_image = processor.intensity_level_slicing(
                    gray_image, low_thresh, high_thresh, highlight_value, preserve_bg
                )
            else:
                processed_image = processor.intensity_level_slicing(
                    image, low_thresh, high_thresh, highlight_value, preserve_bg
                )
                
        elif processing_method == "Mean Filter":
            st.sidebar.subheader("Mean Filter Parameters")
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
            processed_image = processor.mean_filter(image, kernel_size)
            
        elif processing_method == "Gaussian Filter":
            st.sidebar.subheader("Gaussian Filter Parameters")
            sigma = st.sidebar.slider("Sigma (σ)", 0.1, 5.0, 1.0, 0.1)
            processed_image = processor.gaussian_filter_custom(image, sigma)
            
        elif processing_method == "Median Filter":
            st.sidebar.subheader("Median Filter Parameters")
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
            processed_image = processor.median_filter_custom(image, kernel_size)
            
        elif processing_method == "Laplacian Filter":
            processed_image = processor.laplacian_filter(image)
            
        elif processing_method == "High Pass Filter":
            st.sidebar.subheader("High Pass Filter Parameters")
            kernel_size = st.sidebar.slider("Kernel Size", 3, 9, 3, 2)
            processed_image = processor.high_pass_filter(image, kernel_size)
            
        elif processing_method == "Low Pass Filter":
            st.sidebar.subheader("Low Pass Filter Parameters")
            cutoff_freq = st.sidebar.slider("Cutoff Frequency", 10, 100, 30)
            processed_image = processor.low_pass_filter(image, cutoff_freq)
        
        # Display results
        display_images(image, processed_image, f"{processing_method}")
        
        # Display histogram comparison for relevant methods
        if processing_method in ["Histogram Equalization", "Contrast Stretching", 
                               "Logarithmic Transformation", "Power Law Transformation"]:
            st.subheader(" Histogram Comparison")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Original histogram
            if len(image.shape) == 3:
                for i, color in enumerate(['red', 'green', 'blue']):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    ax1.plot(hist, color=color, alpha=0.7)
            else:
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                ax1.plot(hist, color='gray')
            
            ax1.set_title('Original Image Histogram')
            ax1.set_xlabel('Pixel Intensity')
            ax1.set_ylabel('Frequency')
            
            # Processed histogram
            if len(processed_image.shape) == 3:
                for i, color in enumerate(['red', 'green', 'blue']):
                    hist = cv2.calcHist([processed_image], [i], None, [256], [0, 256])
                    ax2.plot(hist, color=color, alpha=0.7)
            else:
                hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
                ax2.plot(hist, color='gray')
            
            ax2.set_title('Processed Image Histogram')
            ax2.set_xlabel('Pixel Intensity')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Download processed image
        st.subheader(" Download Processed Image")
        
        # Convert processed image to PIL Image for download
        if len(processed_image.shape) == 2:
            pil_image = Image.fromarray(processed_image, mode='L')
        else:
            pil_image = Image.fromarray(processed_image, mode='RGB')
        
        # Create download buffer
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label=" Download Processed Image",
            data=buf,
            file_name=f"processed_{processing_method.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
        
    else:
        # Display welcome message
        st.info(" Please upload an image file to get started!")
        
        # Display feature overview
        st.subheader(" Features Available")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Enhancement Techniques:**
            - Contrast Stretching
            - Logarithmic Transformation
            - Power Law (Gamma) Correction
            - Histogram Equalization
            """)
        
        with col2:
            st.markdown("""
            **Analysis Tools:**
            - Bit-Level Slicing
            - Intensity Level Slicing
            - Histogram Visualization
            """)
        
        with col3:
            st.markdown("""
            **Filtering Methods:**
            - Mean Filter (Noise Reduction)
            - Gaussian Filter (Smoothing)
            - Median Filter (Salt & Pepper)
            - Laplacian Filter (Edge Detection)
            - High/Low Pass Filters
            """)

if __name__ == "__main__":
    main()