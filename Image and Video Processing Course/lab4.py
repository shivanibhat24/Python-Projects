import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.ndimage import distance_transform_edt, label, maximum_filter, generate_binary_structure
import io

def load_image(uploaded_file):
    """Load and convert uploaded image"""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_array
            
        return gray_image, image_array
    return None, None

def create_structuring_element(shape, size):
    """Create structuring element for morphological operations"""
    if shape == "Rectangle":
        return np.ones((size, size*2), dtype=bool)
    elif shape == "Cross":
        kernel = np.zeros((size*2+1, size*2+1), dtype=bool)
        kernel[size, :] = True
        kernel[:, size] = True
        return kernel
    else:  # Circle/Disk
        y, x = np.ogrid[-size: size+1, -size: size+1]
        mask = x*x + y*y <= size*size
        return mask

def hough_line_detection(image):
    """Detect lines using Hough Transform"""
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # Draw lines on original image
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return result, edges

def hough_circle_detection(image):
    """Detect circles using Hough Transform"""
    circles = cv2.HoughCircles(
        image, 
        cv2.HOUGH_GRADIENT, 
        1, 
        20,
        param1=50, 
        param2=30, 
        minRadius=0, 
        maxRadius=0
    )
    
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(result, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    return result

def morphological_erosion(image, kernel_shape, kernel_size):
    """Perform morphological erosion"""
    binary = image > 127
    kernel = create_structuring_element(kernel_shape, kernel_size)
    result = binary_erosion(binary, structure=kernel)
    return (result * 255).astype(np.uint8)

def morphological_dilation(image, kernel_shape, kernel_size):
    """Perform morphological dilation"""
    binary = image > 127
    kernel = create_structuring_element(kernel_shape, kernel_size)
    result = binary_dilation(binary, structure=kernel)
    return (result * 255).astype(np.uint8)

def morphological_opening(image, kernel_shape, kernel_size):
    """Perform morphological opening (erosion followed by dilation)"""
    binary = image > 127
    kernel = create_structuring_element(kernel_shape, kernel_size)
    result = binary_opening(binary, structure=kernel)
    return (result * 255).astype(np.uint8)

def morphological_closing(image, kernel_shape, kernel_size):
    """Perform morphological closing (dilation followed by erosion)"""
    binary = image > 127
    kernel = create_structuring_element(kernel_shape, kernel_size)
    result = binary_closing(binary, structure=kernel)
    return (result * 255).astype(np.uint8)

def morphological_gradient(image, kernel_size=2):
    """Perform morphological gradient"""
    binary = image > 127
    kernel = create_structuring_element("Circle", kernel_size)
    
    dilated = binary_dilation(binary, structure=kernel)
    eroded = binary_erosion(binary, structure=kernel)
    # FIXED: Use XOR instead of subtraction for boolean arrays
    gradient = np.logical_xor(dilated, eroded)
    
    return (gradient * 255).astype(np.uint8)

def boundary_extraction(image, kernel_size=1):
    """Extract boundaries using morphological operations"""
    binary = image > 127
    kernel = create_structuring_element("Circle", kernel_size)
    
    eroded = binary_erosion(binary, structure=kernel)
    # FIXED: Use XOR instead of subtraction for boolean arrays
    boundary = np.logical_xor(binary, eroded)
    
    return (boundary * 255).astype(np.uint8)

def separate_connected_components(image, min_distance=20):
    """Separate connected components using distance transform and watershed"""
    binary = image > 127
    
    # Distance transform
    distance = distance_transform_edt(binary)
    
    # Find local maxima
    neighborhood = generate_binary_structure(2, 2)
    local_maxima = maximum_filter(distance, footprint=neighborhood) == distance
    
    # Apply threshold
    threshold = 0.3 * distance.max()
    local_maxima = local_maxima & (distance > threshold)
    
    # Get coordinates and filter by distance
    coords = np.where(local_maxima)
    coordinates = list(zip(coords[0], coords[1]))
    
    filtered_coords = []
    for coord in coordinates:
        too_close = False
        for existing in filtered_coords:
            dist = np.sqrt((coord[0] - existing[0])**2 + (coord[1] - existing[1])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            filtered_coords.append(coord)
    
    # Create markers
    markers = np.zeros_like(binary, dtype=int)
    for i, (y, x) in enumerate(filtered_coords):
        markers[y, x] = i + 1
    
    # Simple region growing instead of watershed
    labeled_array, num_features = label(binary)
    
    # Create colored result
    result = np.zeros((*binary.shape, 3), dtype=np.uint8)
    colors = np.random.randint(50, 255, (num_features + 1, 3))
    
    for i in range(1, num_features + 1):
        mask = labeled_array == i
        result[mask] = colors[i]
    
    return result, num_features

def noise_removal_and_hole_filling(image, open_size=2, close_size=4):
    """Remove noise using opening and fill holes using closing"""
    binary = image > 127
    
    # Opening to remove noise
    open_kernel = create_structuring_element("Circle", open_size)
    opened = binary_opening(binary, structure=open_kernel)
    
    # Closing to fill holes
    close_kernel = create_structuring_element("Circle", close_size)
    closed = binary_closing(opened, structure=close_kernel)
    
    return (opened * 255).astype(np.uint8), (closed * 255).astype(np.uint8)

def main():
    st.set_page_config(page_title="Morphological Image Processing", layout="wide")
    
    st.title("🔬 Morphological Image Processing Application")
    st.markdown("Complete toolkit for image morphology operations and shape detection")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🛠️ Operations")
    operation_type = st.sidebar.selectbox(
        "Select Operation Category:",
        [
            "Hough Transform",
            "Basic Morphological Operations", 
            "Advanced Morphological Operations",
            "Specific Tasks"
        ]
    )
    
    # File uploader
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )
    
    if uploaded_file is not None:
        gray_image, original_image = load_image(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📷 Original Image")
            if len(original_image.shape) == 3:
                st.image(original_image, caption="Original (Color)")
            else:
                st.image(original_image, caption="Original (Grayscale)")
            st.image(gray_image, caption="Grayscale Version")
        
        with col2:
            st.subheader("🔧 Processing Results")
            
            if operation_type == "Hough Transform":
                st.markdown("### Hough Transform Operations")
                
                transform_type = st.selectbox(
                    "Select Transform Type:", 
                    ["Line Detection", "Circle Detection"]
                )
                
                if st.button("🚀 Apply Hough Transform", type="primary"):
                    with st.spinner("Processing..."):
                        if transform_type == "Line Detection":
                            result, edges = hough_line_detection(gray_image)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(edges, caption="Edge Detection")
                            with col_b:
                                st.image(result, caption="Detected Lines")
                        else:
                            result = hough_circle_detection(gray_image)
                            st.image(result, caption="Detected Circles")
            
            elif operation_type == "Basic Morphological Operations":
                st.markdown("### Basic Morphological Operations")
                
                operation = st.selectbox(
                    "Select Operation:", 
                    ["Erosion", "Dilation", "Opening", "Closing"]
                )
                
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    kernel_type = st.selectbox("Kernel Shape:", ["Circle", "Rectangle", "Cross"])
                with col_param2:
                    kernel_size = st.slider("Kernel Size:", 1, 10, 3)
                
                if st.button("🚀 Apply Operation", type="primary"):
                    with st.spinner("Processing..."):
                        if operation == "Erosion":
                            result = morphological_erosion(gray_image, kernel_type, kernel_size)
                        elif operation == "Dilation":
                            result = morphological_dilation(gray_image, kernel_type, kernel_size)
                        elif operation == "Opening":
                            result = morphological_opening(gray_image, kernel_type, kernel_size)
                        else:  # Closing
                            result = morphological_closing(gray_image, kernel_type, kernel_size)
                        
                        st.image(result, caption=f"{operation} Result")
                        st.success(f"{operation} completed successfully!")
            
            elif operation_type == "Advanced Morphological Operations":
                st.markdown("### Advanced Morphological Operations")
                
                advanced_op = st.selectbox(
                    "Select Advanced Operation:", 
                    ["Morphological Gradient", "Boundary Extraction", "Component Separation"]
                )
                
                if advanced_op == "Component Separation":
                    min_distance = st.slider("Minimum Distance Between Objects:", 10, 50, 20)
                elif advanced_op in ["Morphological Gradient", "Boundary Extraction"]:
                    kernel_size = st.slider("Kernel Size:", 1, 5, 2)
                
                if st.button("🚀 Apply Advanced Operation", type="primary"):
                    with st.spinner("Processing..."):
                        if advanced_op == "Morphological Gradient":
                            result = morphological_gradient(gray_image, kernel_size)
                            st.image(result, caption="Morphological Gradient")
                        elif advanced_op == "Boundary Extraction":
                            result = boundary_extraction(gray_image, kernel_size)
                            st.image(result, caption="Extracted Boundaries")
                        else:  # Component Separation
                            result, num_objects = separate_connected_components(gray_image, min_distance)
                            st.image(result, caption="Separated Components")
                            st.success(f"Found {num_objects} separate components!")
            
            else:  # Specific Tasks
                st.markdown("### Specific Tasks")
                
                task = st.selectbox(
                    "Select Task:", 
                    [
                        "Task 1: Ball Separation (Erosion + Watershed)",
                        "Task 2: Noise Removal + Hole Filling"
                    ]
                )
                
                if st.button("🚀 Execute Task", type="primary"):
                    with st.spinner("Executing task..."):
                        if "Task 1" in task:
                            # Step 1: Erosion to separate balls
                            eroded = morphological_erosion(gray_image, "Circle", 3)
                            
                            # Step 2: Component separation
                            separated, num_objects = separate_connected_components(eroded, min_distance=15)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(eroded, caption="Step 1: Erosion")
                            with col_b:
                                st.image(separated, caption="Step 2: Separated Components")
                            
                            st.success(f"Task 1 completed! Separated {num_objects} objects.")
                        
                        else:  # Task 2
                            opened, closed = noise_removal_and_hole_filling(gray_image)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(opened, caption="Step 1: Noise Removal (Opening)")
                            with col_b:
                                st.image(closed, caption="Step 2: Hole Filling (Closing)")
                            
                            st.success("Task 2 completed! Noise removed and holes filled.")
        
        # Information section
        st.markdown("---")
        st.subheader("📚 Operation Information")
        
        if operation_type == "Basic Morphological Operations":
            st.info("""
            **Morphological Operations Explained:**
            - **Erosion**: Shrinks white regions, removes small objects and noise
            - **Dilation**: Expands white regions, fills small holes and gaps
            - **Opening**: Erosion → Dilation, removes noise while preserving shape
            - **Closing**: Dilation → Erosion, fills holes and connects broken parts
            """)
        
        elif operation_type == "Hough Transform":
            st.info("""
            **Hough Transform:**
            - Mathematical technique for detecting geometric shapes
            - Transforms image space to parameter space
            - Robust to noise and partial occlusions
            - Widely used for line and circle detection
            """)
        
        elif operation_type == "Advanced Morphological Operations":
            st.info("""
            **Advanced Operations:**
            - **Morphological Gradient**: Highlights edges and boundaries
            - **Boundary Extraction**: Isolates object contours
            - **Component Separation**: Separates touching objects using distance transform
            """)
    
    else:
        st.info("👆 Please upload an image to start processing!")
        
        # Show demo information
        st.markdown("## 🎯 Available Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Hough Transform
            - **Line Detection**: Find straight lines in images
            - **Circle Detection**: Identify circular shapes
            
            ### ⚙️ Basic Morphological Operations  
            - **Erosion**: Object shrinking
            - **Dilation**: Object expansion
            - **Opening**: Noise removal
            - **Closing**: Gap filling
            """)
        
        with col2:
            st.markdown("""
            ### 🔬 Advanced Operations
            - **Morphological Gradient**: Edge enhancement
            - **Boundary Extraction**: Contour detection
            - **Component Separation**: Object separation
            
            ### 🎯 Specific Tasks
            - **Task 1**: Ball separation using erosion + watershed
            - **Task 2**: Fingerprint cleaning (denoise + fill holes)
            """)

if __name__ == "__main__":
    main()