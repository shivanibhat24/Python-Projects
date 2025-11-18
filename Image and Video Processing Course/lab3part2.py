import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from scipy import ndimage
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Advanced Image Processing App",
    page_icon="🖼️",
    layout="wide"
)

class ImageProcessor:
    """Class containing all image processing filters"""
    
    @staticmethod
    def contrast_stretching(image):
        """Apply contrast stretching to enhance image contrast"""
        # Convert to grayscale if colored
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Find min and max values
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # Check for division by zero
        if max_val - min_val == 0:
            return image  # Return original if no contrast
        
        # Apply contrast stretching formula
        stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # If original was colored, apply to all channels
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val - min_val == 0:
                    result[:, :, i] = channel
                else:
                    result[:, :, i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            return result
        else:
            return stretched
    
    @staticmethod
    def logarithmic_transformation(image, c=1):
        """Apply logarithmic transformation"""
        # Normalize to 0-1 range
        image_norm = image.astype(np.float64) / 255.0
        # Apply log transformation: s = c * log(1 + r)
        log_transformed = c * np.log(1 + image_norm)
        # Normalize back to 0-255
        log_transformed = (log_transformed / np.max(log_transformed) * 255).astype(np.uint8)
        return log_transformed
    
    @staticmethod
    def power_law_transformation(image, gamma=1.5, c=1):
        """Apply power law (gamma) transformation"""
        # Normalize to 0-1 range
        image_norm = image.astype(np.float64) / 255.0
        # Apply power law: s = c * r^gamma
        power_transformed = c * np.power(image_norm, gamma)
        # Normalize back to 0-255
        power_transformed = (power_transformed * 255).astype(np.uint8)
        return power_transformed
    
    @staticmethod
    def bit_level_slicing(image, bit_level=7):
        """Apply bit-level slicing"""
        # Create a mask for the specified bit level
        bit_mask = 1 << bit_level
        # Apply the mask
        sliced = np.bitwise_and(image, bit_mask)
        # Scale to full range for visibility
        sliced = (sliced > 0).astype(np.uint8) * 255
        return sliced
    
    @staticmethod
    def histogram_equalization(image):
        """Apply histogram equalization"""
        if len(image.shape) == 3:
            # For color images, apply to each channel separately
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return result
        else:
            # For grayscale images
            return cv2.equalizeHist(image)
    
    @staticmethod
    def intensity_level_slicing(image, low_thresh=50, high_thresh=200):
        """Apply intensity level slicing"""
        result = image.copy()
        # Highlight pixels in the specified intensity range
        mask = (image >= low_thresh) & (image <= high_thresh)
        result[mask] = 255
        result[~mask] = 0
        return result
    
    @staticmethod
    def mean_filter(image, kernel_size=5):
        """Apply mean (box) filter for noise reduction"""
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        if len(image.shape) == 3:
            return cv2.filter2D(image, -1, kernel)
        else:
            return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def gaussian_filter_custom(image, kernel_size=5, sigma=1.0):
        """Apply Gaussian filter for noise reduction"""
        if len(image.shape) == 3:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        else:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def median_filter_custom(image, kernel_size=5):
        """Apply median filter for salt and pepper noise reduction"""
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = median_filter(image[:, :, i], size=kernel_size)
            return result.astype(np.uint8)
        else:
            return median_filter(image, size=kernel_size).astype(np.uint8)
    
    @staticmethod
    def laplacian_filter(image):
        """Apply Laplacian filter for edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        return laplacian
    
    @staticmethod
    def high_pass_filter(image):
        """Apply high pass filter"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # High pass kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        
        filtered = cv2.filter2D(gray, -1, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    @staticmethod
    def low_pass_filter(image):
        """Apply low pass filter"""
        # Low pass is essentially a Gaussian blur
        return ImageProcessor.gaussian_filter_custom(image, kernel_size=5, sigma=1.5)
    
    @staticmethod
    def sobel_filter(image):
        """Apply Sobel edge detection filter"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Sobel X and Y
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine both gradients
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
        
        return sobel_combined
    
    @staticmethod
    def prewitt_filter(image):
        """Apply Prewitt edge detection filter"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        
        kernel_y = np.array([[-1, -1, -1],
                            [ 0,  0,  0],
                            [ 1,  1,  1]])
        
        # Apply filters
        prewitt_x = cv2.filter2D(gray, -1, kernel_x)
        prewitt_y = cv2.filter2D(gray, -1, kernel_y)
        
        # Combine both gradients
        prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)
        prewitt_combined = np.uint8(prewitt_combined / prewitt_combined.max() * 255)
        
        return prewitt_combined
    
    @staticmethod
    def robert_filter(image):
        """Apply Robert edge detection filter"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Robert kernels
        kernel_x = np.array([[1, 0],
                            [0, -1]])
        
        kernel_y = np.array([[0, 1],
                            [-1, 0]])
        
        # Apply filters
        robert_x = cv2.filter2D(gray, -1, kernel_x)
        robert_y = cv2.filter2D(gray, -1, kernel_y)
        
        # Combine both gradients
        robert_combined = np.sqrt(robert_x**2 + robert_y**2)
        robert_combined = np.uint8(robert_combined / robert_combined.max() * 255)
        
        return robert_combined
    
    @staticmethod
    def binary_thresholding(image, threshold=128):
        """Apply binary thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary thresholding
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def adaptive_thresholding(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, 
                            threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
        """Apply adaptive thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, max_value, adaptive_method, 
                                       threshold_type, block_size, C)
        return adaptive
    
    @staticmethod
    def otsu_thresholding(image):
        """Apply Otsu's automatic thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Otsu's thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu
    
    @staticmethod
    def multi_level_thresholding(image, thresholds=[85, 170]):
        """Apply multi-level thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Create output image
        result = np.zeros_like(gray)
        
        # Sort thresholds
        thresholds = sorted(thresholds)
        
        # Apply multiple thresholds
        for i, thresh in enumerate(thresholds):
            if i == 0:
                result[gray <= thresh] = int(255 / (len(thresholds) + 1))
            else:
                mask = (gray > thresholds[i-1]) & (gray <= thresh)
                result[mask] = int(255 * (i + 1) / (len(thresholds) + 1))
        
        # Handle pixels above highest threshold
        result[gray > thresholds[-1]] = 255
        
        return result
    
    @staticmethod
    def uniform_quantization(image, levels=8):
        """Apply uniform quantization to reduce intensity levels"""
        # Calculate quantization step
        step = 256 // levels
        
        # Apply quantization
        quantized = (image // step) * step
        
        # Ensure values are in valid range
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        return quantized
    
    @staticmethod
    def non_uniform_quantization(image, levels=8):
        """Apply non-uniform quantization based on histogram"""
        if len(image.shape) == 3:
            # For color images, apply to each channel
            result = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                hist, bins = np.histogram(channel, bins=256, range=[0, 256])
                cumsum = np.cumsum(hist)
                
                # Create quantization levels based on cumulative histogram
                total_pixels = cumsum[-1]
                thresholds = []
                for j in range(1, levels):
                    threshold_pixel = j * total_pixels // levels
                    threshold_idx = np.argmax(cumsum >= threshold_pixel)
                    thresholds.append(threshold_idx)
                
                # Apply quantization
                quantized = np.zeros_like(channel)
                for k, thresh in enumerate(thresholds):
                    if k == 0:
                        quantized[channel <= thresh] = int(255 * k / (levels - 1))
                    else:
                        mask = (channel > thresholds[k-1]) & (channel <= thresh)
                        quantized[mask] = int(255 * k / (levels - 1))
                
                # Handle pixels above highest threshold
                quantized[channel > thresholds[-1]] = 255
                result[:, :, i] = quantized
            
            return result
        else:
            # For grayscale images
            hist, bins = np.histogram(image, bins=256, range=[0, 256])
            cumsum = np.cumsum(hist)
            
            # Create quantization levels based on cumulative histogram
            total_pixels = cumsum[-1]
            thresholds = []
            for j in range(1, levels):
                threshold_pixel = j * total_pixels // levels
                threshold_idx = np.argmax(cumsum >= threshold_pixel)
                thresholds.append(threshold_idx)
            
            # Apply quantization
            quantized = np.zeros_like(image)
            for k, thresh in enumerate(thresholds):
                if k == 0:
                    quantized[image <= thresh] = int(255 * k / (levels - 1))
                else:
                    mask = (image > thresholds[k-1]) & (image <= thresh)
                    quantized[mask] = int(255 * k / (levels - 1))
            
            # Handle pixels above highest threshold
            quantized[image > thresholds[-1]] = 255
            
            return quantized
    
    @staticmethod
    def downsampling(image, factor=2):
        """Apply spatial downsampling"""
        if factor <= 1:
            return image
        
        # Calculate new dimensions
        if len(image.shape) == 3:
            new_height = image.shape[0] // factor
            new_width = image.shape[1] // factor
            # Downsample by taking every nth pixel
            downsampled = image[::factor, ::factor, :]
        else:
            new_height = image.shape[0] // factor
            new_width = image.shape[1] // factor
            # Downsample by taking every nth pixel
            downsampled = image[::factor, ::factor]
        
        return downsampled
    
    @staticmethod
    def upsampling_nearest(image, factor=2):
        """Apply nearest neighbor upsampling"""
        if factor <= 1:
            return image
        
        # Use OpenCV resize with nearest neighbor interpolation
        if len(image.shape) == 3:
            new_width = image.shape[1] * factor
            new_height = image.shape[0] * factor
        else:
            new_width = image.shape[1] * factor
            new_height = image.shape[0] * factor
        
        upsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return upsampled
    
    @staticmethod
    def upsampling_bilinear(image, factor=2):
        """Apply bilinear upsampling"""
        if factor <= 1:
            return image
        
        # Use OpenCV resize with bilinear interpolation
        if len(image.shape) == 3:
            new_width = image.shape[1] * factor
            new_height = image.shape[0] * factor
        else:
            new_width = image.shape[1] * factor
            new_height = image.shape[0] * factor
        
        upsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return upsampled
    
    @staticmethod
    def subsampling_with_aliasing(image, factor=4):
        """Apply subsampling without anti-aliasing to demonstrate aliasing effects"""
        if factor <= 1:
            return image
        
        # Direct subsampling without low-pass filtering
        if len(image.shape) == 3:
            subsampled = image[::factor, ::factor, :]
        else:
            subsampled = image[::factor, ::factor]
        
        return subsampled
    
    @staticmethod
    def subsampling_with_antialiasing(image, factor=4):
        """Apply subsampling with anti-aliasing (low-pass filter first)"""
        if factor <= 1:
            return image
        
        # Apply Gaussian filter to reduce high frequencies
        sigma = factor / 2.0
        if len(image.shape) == 3:
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            subsampled = blurred[::factor, ::factor, :]
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            subsampled = blurred[::factor, ::factor]
        
        return subsampled

def apply_filters(image, filter_list):
    """Apply a list of filters to an image in sequence"""
    processor = ImageProcessor()
    result = image.copy()
    
    filter_methods = {
        "Contrast Stretching": processor.contrast_stretching,
        "Logarithmic Transformation": processor.logarithmic_transformation,
        "Power Law Transformation": processor.power_law_transformation,
        "Bit-Level Slicing": processor.bit_level_slicing,
        "Histogram Equalization": processor.histogram_equalization,
        "Intensity Level Slicing": processor.intensity_level_slicing,
        "Mean Filter": processor.mean_filter,
        "Gaussian Filter": processor.gaussian_filter_custom,
        "Median Filter": processor.median_filter_custom,
        "Laplacian Filter": processor.laplacian_filter,
        "High Pass Filter": processor.high_pass_filter,
        "Low Pass Filter": processor.low_pass_filter,
        "Sobel Filter": processor.sobel_filter,
        "Prewitt Filter": processor.prewitt_filter,
        "Robert Filter": processor.robert_filter,
        "Binary Thresholding": processor.binary_thresholding,
        "Adaptive Thresholding": processor.adaptive_thresholding,
        "Otsu Thresholding": processor.otsu_thresholding,
        "Multi-Level Thresholding": processor.multi_level_thresholding,
        "Uniform Quantization": processor.uniform_quantization,
        "Non-Uniform Quantization": processor.non_uniform_quantization,
        "Downsampling": processor.downsampling,
        "Upsampling (Nearest)": processor.upsampling_nearest,
        "Upsampling (Bilinear)": processor.upsampling_bilinear,
        "Subsampling (with Aliasing)": processor.subsampling_with_aliasing,
        "Subsampling (Anti-Aliased)": processor.subsampling_with_antialiasing
    }
    
    for filter_name in filter_list:
        if filter_name in filter_methods:
            try:
                result = filter_methods[filter_name](result)
                # Ensure result is in valid range
                result = np.clip(result, 0, 255).astype(np.uint8)
            except Exception as e:
                st.error(f"Error applying {filter_name}: {str(e)}")
                break
    
    return result

def main():
    st.title("🖼️ Advanced Image Processing App")
    st.markdown("Upload an image and apply multiple enhancement filters to improve its quality!")
    
    # Display Streamlit version for debugging
    st.sidebar.write(f"Streamlit version: {st.__version__}")
    
    # Sidebar for filter selection
    st.sidebar.header("Filter Selection")
    
    available_filters = [
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
        "Low Pass Filter",
        "Sobel Filter",
        "Prewitt Filter",
        "Robert Filter",
        "Binary Thresholding",
        "Adaptive Thresholding",
        "Otsu Thresholding",
        "Multi-Level Thresholding",
        "Uniform Quantization",
        "Non-Uniform Quantization",
        "Downsampling",
        "Upsampling (Nearest)",
        "Upsampling (Bilinear)",
        "Subsampling (with Aliasing)",
        "Subsampling (Anti-Aliased)"
    ]
    
    # Multi-select for filters
    selected_filters = st.sidebar.multiselect(
        "Choose filters to apply (in order):",
        available_filters
    )
    
    # Number of filters selected
    st.sidebar.info(f"Number of filters selected: {len(selected_filters)}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Ensure image is in correct format
            if len(image_array.shape) == 4:  # RGBA
                image_array = image_array[:, :, :3]  # Convert to RGB
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Original Image", width=None)
                st.info(f"Image dimensions: {image_array.shape}")
        
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return
        
        # Process button
        if st.button("🎯 Apply Filters", disabled=len(selected_filters)==0):
            if len(selected_filters) > 0:
                with st.spinner("Processing image..."):
                    try:
                        # Apply selected filters
                        processed_image = apply_filters(image_array, selected_filters)
                        
                        with col2:
                            st.subheader("Processed Image")
                            st.image(processed_image, caption=f"After applying {len(selected_filters)} filter(s)", width=None)
                            
                            # Show applied filters
                            st.success("✅ Filters applied successfully!")
                            st.write("**Applied filters in order:**")
                            for i, filter_name in enumerate(selected_filters, 1):
                                st.write(f"{i}. {filter_name}")
                        
                        # Download button
                        if processed_image is not None:
                            # Convert to PIL Image for download
                            if len(processed_image.shape) == 2:
                                # Grayscale
                                pil_image = Image.fromarray(processed_image, mode='L')
                            else:
                                # Color
                                pil_image = Image.fromarray(processed_image, mode='RGB')
                            
                            # Save to bytes
                            img_buffer = BytesIO()
                            pil_image.save(img_buffer, format='PNG')
                            img_bytes = img_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Processed Image",
                                data=img_bytes,
                                file_name=f"processed_image_{len(selected_filters)}_filters.png",
                                mime="image/png"
                            )
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
            else:
                st.warning("Please select at least one filter to apply.")
    
    # Information section
    with st.expander("ℹ️ About the Filters"):
        st.markdown("""
        **Enhancement Filters:**
        - **Contrast Stretching**: Improves contrast by stretching intensity values
        - **Logarithmic Transformation**: Enhances dark regions of an image
        - **Power Law Transformation**: Gamma correction for brightness adjustment
        - **Histogram Equalization**: Improves contrast through histogram redistribution
        - **Intensity Level Slicing**: Highlights specific intensity ranges
        - **Bit-Level Slicing**: Highlights specific bit planes
        
        **Noise Reduction Filters:**
        - **Mean Filter**: Reduces noise by averaging neighboring pixels
        - **Gaussian Filter**: Applies Gaussian blur for smooth noise reduction
        - **Median Filter**: Effective for salt-and-pepper noise removal
        - **Low Pass Filter**: Removes high-frequency noise
        
        **Edge Detection Filters:**
        - **Laplacian Filter**: Detects edges using second-order derivatives
        - **High Pass Filter**: Enhances edges and fine details
        - **Sobel Filter**: Gradient-based edge detection
        - **Prewitt Filter**: Edge detection using Prewitt operator
        - **Robert Filter**: Simple edge detection using Robert cross-gradient
        
        **Thresholding Filters:**
        - **Binary Thresholding**: Converts image to binary using fixed threshold
        - **Adaptive Thresholding**: Uses local statistics for thresholding
        - **Otsu Thresholding**: Automatic threshold selection using Otsu's method
        - **Multi-Level Thresholding**: Creates multiple intensity levels
        
        **Quantization Filters:**
        - **Uniform Quantization**: Reduces intensity levels uniformly
        - **Non-Uniform Quantization**: Reduces levels based on histogram distribution
        
        **Sampling Filters:**
        - **Downsampling**: Reduces spatial resolution by factor of 2
        - **Upsampling (Nearest)**: Increases resolution using nearest neighbor
        - **Upsampling (Bilinear)**: Increases resolution using bilinear interpolation
        - **Subsampling (with Aliasing)**: Demonstrates aliasing effects
        - **Subsampling (Anti-Aliased)**: Proper subsampling with anti-aliasing filter
        """)
        
    # Parameter Controls
    with st.expander("🎛️ Advanced Parameters"):
        st.markdown("""
        **Note**: These are default parameters used by the filters. For custom control, 
        you would need to modify the code to accept user inputs.
        
        **Current Default Parameters:**
        - Binary Thresholding: threshold = 128
        - Adaptive Thresholding: block_size = 11, C = 2  
        - Multi-Level Thresholding: thresholds = [85, 170]
        - Uniform Quantization: levels = 8
        - Non-Uniform Quantization: levels = 8
        - Downsampling/Upsampling: factor = 2
        - Subsampling: factor = 4
        """)
    
    # Usage instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Upload an Image**: Click on the file uploader and select your image
        2. **Select Filters**: Choose one or more filters from the sidebar
        3. **Apply Filters**: Click the "Apply Filters" button to process your image
        4. **Download Result**: Download your processed image using the download button
        
        **Tips:**
        - Filters are applied in the order you select them
        - You can select multiple filters for enhanced processing
        - Different combinations of filters can produce various artistic effects
        - Edge detection filters work best on grayscale images
        """)

if __name__ == "__main__":
    main()