import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
import io

def sobel_filter(image):
    """Apply Sobel edge detection filter"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Sobel kernels
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    return sobel_combined

def prewitt_filter(image):
    """Apply Prewitt edge detection filter"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    # Apply convolution
    edges_x = cv2.filter2D(gray, cv2.CV_64F, prewitt_x)
    edges_y = cv2.filter2D(gray, cv2.CV_64F, prewitt_y)
    
    # Combine gradients
    prewitt_combined = np.sqrt(edges_x**2 + edges_y**2)
    prewitt_combined = np.uint8(np.clip(prewitt_combined, 0, 255))
    
    return prewitt_combined

def robert_filter(image):
    """Apply Robert edge detection filter"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Robert kernels
    robert_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    robert_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # Apply convolution
    edges_x = cv2.filter2D(gray, cv2.CV_64F, robert_x)
    edges_y = cv2.filter2D(gray, cv2.CV_64F, robert_y)
    
    # Combine gradients
    robert_combined = np.sqrt(edges_x**2 + edges_y**2)
    robert_combined = np.uint8(np.clip(robert_combined, 0, 255))
    
    return robert_combined

def calculate_color_intensity(color):
    """Calculate color intensity (brightness) using luminance formula"""
    # Using standard luminance formula: 0.299*R + 0.587*G + 0.114*B
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def sort_colors_by_intensity(colors, labels, image_shape):
    """Sort colors by intensity (brightness) from darkest to brightest"""
    # Calculate intensity for each color
    intensities = [calculate_color_intensity(color) for color in colors]
    
    # Create list of tuples (intensity, color, original_index)
    color_data = list(zip(intensities, colors, range(len(colors))))
    
    # Sort by intensity (darkest to brightest)
    color_data.sort(key=lambda x: x[0])
    
    # Extract sorted colors and create mapping
    sorted_colors = np.array([item[1] for item in color_data])
    old_to_new_mapping = {item[2]: i for i, item in enumerate(color_data)}
    
    # Remap labels to match new sorting
    new_labels = np.array([old_to_new_mapping[label] for label in labels])
    
    return sorted_colors, new_labels

def extract_dominant_colors(image, n_colors=8):
    """Extract dominant colors from image using K-means clustering and sort by intensity"""
    # Reshape image to be a list of pixels
    data = image.reshape((-1, 3))
    
    # Apply K-means clustering with error handling
    try:
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(data)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Sort colors by intensity
        sorted_colors, sorted_labels = sort_colors_by_intensity(colors, labels, image.shape)
        
        return sorted_colors, sorted_labels
    except Exception as e:
        st.error(f"Error in color extraction: {e}")
        # Return default colors if clustering fails (sorted by intensity)
        default_colors = np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]])  # Black to white
        default_labels = np.zeros(data.shape[0])
        return default_colors, default_labels

def create_color_mask(image, target_color, tolerance=30):
    """Create a mask for pixels similar to the target color"""
    # Calculate color distance
    color_diff = np.sqrt(np.sum((image - target_color) ** 2, axis=2))
    
    # Create mask where color difference is less than tolerance
    mask = color_diff < tolerance
    
    return mask

def apply_color_filter(image, target_color, tolerance=30):
    """Apply color filter to show only parts with target color"""
    mask = create_color_mask(image, target_color, tolerance)
    
    # Create filtered image
    filtered_image = image.copy()
    filtered_image[~mask] = [255, 255, 255]  # Set non-matching pixels to white
    
    return filtered_image

def rgb_to_hex(rgb):
    """Convert RGB to hex color code"""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def main():
    st.set_page_config(page_title="Lab 3 Part 1", layout="wide")
    
    st.title("Lab3 Part 1")
    st.markdown("Upload an image to apply edge detection filters and extract colors!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Load and display original image
        try:
            image = Image.open(uploaded_file)
            image_array = np.array(image.convert('RGB'))
            
            st.subheader("Original Image")
            st.image(image, caption="Original Image", use_column_width=True)
        
            # Create tabs for different operations
            tab1, tab2 = st.tabs(["🔍 Edge Detection Filters", "🎨 Color Extraction"])
            
            with tab1:
                st.subheader("Edge Detection Filters")
                
                try:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Sobel Filter**")
                        sobel_result = sobel_filter(image_array)
                        st.image(sobel_result, caption="Sobel Edge Detection", use_column_width=True)
                    
                    with col2:
                        st.markdown("**Prewitt Filter**")
                        prewitt_result = prewitt_filter(image_array)
                        st.image(prewitt_result, caption="Prewitt Edge Detection", use_column_width=True)
                    
                    with col3:
                        st.markdown("**Robert Filter**")
                        robert_result = robert_filter(image_array)
                        st.image(robert_result, caption="Robert Edge Detection", use_column_width=True)
                        
                except Exception as e:
                    st.error(f"Error applying edge detection filters: {e}")
            
            with tab2:
                st.subheader("Color Extraction and Filtering")
                
                try:
                    # Color extraction parameters
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        n_colors = st.slider("Number of colors to extract:", 3, 15, 8)
                        tolerance = st.slider("Color matching tolerance:", 10, 100, 30)
                    
                    with col2:
                        # Extract dominant colors
                        with st.spinner("Extracting colors..."):
                            colors, labels = extract_dominant_colors(image_array, n_colors)
                        
                        st.markdown("**Extracted Colors (sorted by intensity - darkest to brightest):**")
                        
                        # Create color palette
                        palette_cols = st.columns(min(8, len(colors)))
                        
                        color_buttons = []
                        for i, color in enumerate(colors):
                            col_idx = i % 8
                            with palette_cols[col_idx]:
                                hex_color = rgb_to_hex(color)
                                intensity = calculate_color_intensity(color)
                                
                                # Create a colored button using HTML/CSS
                                button_html = f"""
                                <div style="
                                    width: 70px; 
                                    height: 70px; 
                                    background-color: {hex_color}; 
                                    border: 2px solid #333; 
                                    border-radius: 8px; 
                                    margin: 5px auto;
                                    cursor: pointer;
                                    display: flex;
                                    flex-direction: column;
                                    align-items: center;
                                    justify-content: center;
                                    color: {'white' if intensity < 127 else 'black'};
                                    font-size: 9px;
                                    text-align: center;
                                    padding: 2px;
                                ">
                                    <div>RGB<br>({color[0]}, {color[1]}, {color[2]})</div>
                                    <div style="font-size: 8px; margin-top: 2px;">
                                        Intensity: {intensity:.0f}
                                    </div>
                                </div>
                                """
                                st.markdown(button_html, unsafe_allow_html=True)
                                
                                # Create actual button for interaction
                                if st.button(f"Filter Color {i+1}", key=f"color_{i}", use_container_width=True):
                                    st.session_state.selected_color = color
                                    st.session_state.selected_color_index = i
                        
                        # Show filtered image if color is selected
                        if 'selected_color' in st.session_state:
                            selected_color = st.session_state.selected_color
                            color_index = st.session_state.selected_color_index
                            intensity = calculate_color_intensity(selected_color)
                            
                            st.markdown(f"**Filtered Image - Color {color_index + 1}:**")
                            st.markdown(f"Selected Color: RGB({selected_color[0]}, {selected_color[1]}, {selected_color[2]}) | Intensity: {intensity:.0f}")
                            
                            # Apply color filter
                            filtered_image = apply_color_filter(image_array, selected_color, tolerance)
                            
                            # Display filtered image
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(image_array, caption="Original Image", use_column_width=True)
                            with col2:
                                st.image(filtered_image, caption=f"Filtered for Color {color_index + 1}", use_column_width=True)
                            
                            # Show statistics
                            mask = create_color_mask(image_array, selected_color, tolerance)
                            coverage_percentage = (np.sum(mask) / mask.size) * 100
                            
                            st.info(f"This color covers approximately {coverage_percentage:.2f}% of the image")
                        
                        # Reset button
                        if st.button("🔄 Reset Color Selection"):
                            if 'selected_color' in st.session_state:
                                del st.session_state.selected_color
                            if 'selected_color_index' in st.session_state:
                                del st.session_state.selected_color_index
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error in color extraction: {e}")
        
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.info("Please try uploading a different image file.")

if __name__ == "__main__":
    main()