import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import base64

# Professional theme configurations
THEMES = {
    # Professional Dark Themes
    "Corporate Slate": {
        "primary": "#3b82f6", 
        "secondary": "#60a5fa", 
        "background": "#0f172a", 
        "text": "#e2e8f0", 
        "card": "#1e293b", 
        "accent": "#2563eb"
    },
    "Executive Navy": {
        "primary": "#0ea5e9", 
        "secondary": "#38bdf8", 
        "background": "#020617", 
        "text": "#f1f5f9", 
        "card": "#0c1629", 
        "accent": "#0284c7"
    },
    "Modern Charcoal": {
        "primary": "#06b6d4", 
        "secondary": "#22d3ee", 
        "background": "#18181b", 
        "text": "#f4f4f5", 
        "card": "#27272a", 
        "accent": "#0891b2"
    },
    "Professional Indigo": {
        "primary": "#6366f1", 
        "secondary": "#818cf8", 
        "background": "#1e1b4b", 
        "text": "#e0e7ff", 
        "card": "#312e81", 
        "accent": "#4f46e5"
    },
    "Business Teal": {
        "primary": "#14b8a6", 
        "secondary": "#2dd4bf", 
        "background": "#042f2e", 
        "text": "#ccfbf1", 
        "card": "#134e4a", 
        "accent": "#0d9488"
    },
    "Elegant Emerald": {
        "primary": "#10b981", 
        "secondary": "#34d399", 
        "background": "#022c22", 
        "text": "#d1fae5", 
        "card": "#064e3b", 
        "accent": "#059669"
    },
    "Refined Purple": {
        "primary": "#a855f7", 
        "secondary": "#c084fc", 
        "background": "#2e1065", 
        "text": "#f3e8ff", 
        "card": "#4c1d95", 
        "accent": "#9333ea"
    },
    "Sophisticated Rose": {
        "primary": "#f43f5e", 
        "secondary": "#fb7185", 
        "background": "#4c0519", 
        "text": "#ffe4e6", 
        "card": "#881337", 
        "accent": "#e11d48"
    },
    "Warm Copper": {
        "primary": "#ea580c", 
        "secondary": "#fb923c", 
        "background": "#1c0a00", 
        "text": "#ffedd5", 
        "card": "#431407", 
        "accent": "#c2410c"
    },
    "Golden Executive": {
        "primary": "#eab308", 
        "secondary": "#facc15", 
        "background": "#1c1917", 
        "text": "#fef9c3", 
        "card": "#292524", 
        "accent": "#ca8a04"
    },
    "Arctic Professional": {
        "primary": "#06b6d4", 
        "secondary": "#67e8f9", 
        "background": "#083344", 
        "text": "#cffafe", 
        "card": "#164e63", 
        "accent": "#0891b2"
    },
    "Steel Blue": {
        "primary": "#3b82f6", 
        "secondary": "#93c5fd", 
        "background": "#172554", 
        "text": "#dbeafe", 
        "card": "#1e3a8a", 
        "accent": "#2563eb"
    },
    "Deep Ocean": {
        "primary": "#0284c7", 
        "secondary": "#0ea5e9", 
        "background": "#082f49", 
        "text": "#e0f2fe", 
        "card": "#0c4a6e", 
        "accent": "#0369a1"
    },
    "Midnight Blue": {
        "primary": "#1e40af", 
        "secondary": "#3b82f6", 
        "background": "#1e1b4b", 
        "text": "#dbeafe", 
        "card": "#312e81", 
        "accent": "#1d4ed8"
    },
    "Forest Green": {
        "primary": "#16a34a", 
        "secondary": "#22c55e", 
        "background": "#14532d", 
        "text": "#dcfce7", 
        "card": "#166534", 
        "accent": "#15803d"
    }
}

def apply_theme(theme_name):
    """Apply selected theme styling to the application"""
    theme = THEMES[theme_name]
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {theme['background']} 0%, {theme['card']} 100%);
            color: {theme['text']};
        }}
        .stApp header {{
            background-color: transparent !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['primary']} !important;
            text-shadow: 0 0 10px {theme['primary']}40;
            font-weight: 700 !important;
        }}
        p, span, div, label {{
            color: {theme['text']} !important;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            color: {theme['background']};
            border: 2px solid {theme['primary']};
            font-weight: 600;
            padding: 0.5rem 2rem;
            border-radius: 10px;
            transition: all 0.3s;
            box-shadow: 0 0 20px {theme['primary']}60;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 30px {theme['primary']}90;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme['card']};
            border-radius: 10px;
            padding: 0.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {theme['text']} !important;
            background-color: transparent;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']}) !important;
            color: {theme['background']} !important;
            box-shadow: 0 0 15px {theme['primary']}70;
        }}
        .stMetric {{
            background-color: {theme['card']};
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid {theme['primary']}50;
            box-shadow: 0 0 15px {theme['primary']}30;
        }}
        .stMetric label {{
            color: {theme['secondary']} !important;
            font-weight: 600;
        }}
        .stMetric [data-testid="stMetricValue"] {{
            color: {theme['primary']} !important;
            font-size: 2rem !important;
        }}
        .stAlert {{
            background-color: {theme['card']} !important;
            color: {theme['text']} !important;
            border-left: 4px solid {theme['primary']};
        }}
        .stSelectbox, .stSlider {{
            color: {theme['text']} !important;
        }}
        .stSelectbox > div > div {{
            background-color: {theme['card']} !important;
            color: {theme['text']} !important;
            border: 1px solid {theme['primary']}50;
        }}
        [data-baseweb="select"] {{
            background-color: {theme['card']} !important;
        }}
        .stMarkdown {{
            color: {theme['text']} !important;
        }}
        .stMarkdown a {{
            color: {theme['secondary']} !important;
        }}
        hr {{
            border-color: {theme['primary']}50 !important;
        }}
        .uploadedFile {{
            background-color: {theme['card']} !important;
            border: 1px solid {theme['primary']}50;
        }}
        .stSpinner > div {{
            border-top-color: {theme['primary']} !important;
        }}
        .stRadio > div {{
            background-color: {theme['card']};
            padding: 1rem;
            border-radius: 8px;
        }}
        .stExpander {{
            background-color: {theme['card']};
            border: 1px solid {theme['primary']}30;
            border-radius: 8px;
        }}
    </style>
    """, unsafe_allow_html=True)

def save_image_formats(image, quality=90):
    """Save image in multiple formats and return size information"""
    results = {}
    
    # PNG (lossless)
    png_buffer = io.BytesIO()
    image.save(png_buffer, format='PNG')
    results['PNG'] = {
        'buffer': png_buffer,
        'size': len(png_buffer.getvalue()),
        'data': png_buffer.getvalue()
    }
    
    # JPEG (lossy)
    jpeg_buffer = io.BytesIO()
    image.save(jpeg_buffer, format='JPEG', quality=quality)
    results[f'JPEG_Q{quality}'] = {
        'buffer': jpeg_buffer,
        'size': len(jpeg_buffer.getvalue()),
        'data': jpeg_buffer.getvalue()
    }
    
    # WebP
    webp_buffer = io.BytesIO()
    image.save(webp_buffer, format='WEBP', quality=quality)
    results[f'WEBP_Q{quality}'] = {
        'buffer': webp_buffer,
        'size': len(webp_buffer.getvalue()),
        'data': webp_buffer.getvalue()
    }
    
    return results

def verify_lossless_reconstruction(original_image, compressed_data):
    """Verify if compression maintains bit-exact reconstruction"""
    reconstructed = Image.open(io.BytesIO(compressed_data))
    
    original_array = np.array(original_image)
    reconstructed_array = np.array(reconstructed)
    
    dimensions_match = original_array.shape == reconstructed_array.shape
    
    if dimensions_match:
        bit_exact = np.array_equal(original_array, reconstructed_array)
        max_diff = np.max(np.abs(original_array.astype(int) - reconstructed_array.astype(int)))
        mse = np.mean((original_array.astype(float) - reconstructed_array.astype(float)) ** 2)
    else:
        bit_exact = False
        max_diff = None
        mse = None
    
    return {
        'dimensions_match': dimensions_match,
        'bit_exact': bit_exact,
        'max_difference': max_diff,
        'mse': mse,
        'original_shape': original_array.shape,
        'reconstructed_shape': reconstructed_array.shape
    }

def detect_roi(image_cv):
    """Detect regions of interest (faces) in image"""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def roi_based_compression(image, roi_quality=90, bg_quality=40):
    """Apply ROI-based compression with different quality levels"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = detect_roi(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    if len(faces) == 0:
        st.warning("No faces detected. Using center region as ROI.")
        h, w = img_cv.shape[:2]
        faces = [(w//4, h//4, w//2, h//2)]
    
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    roi_buffer = io.BytesIO()
    bg_buffer = io.BytesIO()
    
    roi_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    roi_img.save(roi_buffer, format='JPEG', quality=roi_quality)
    
    img_cv_float = img_cv.astype(float)
    
    low_quality_buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).save(
        low_quality_buffer, format='JPEG', quality=bg_quality
    )
    low_quality_img = Image.open(low_quality_buffer)
    low_quality_cv = cv2.cvtColor(np.array(low_quality_img), cv2.COLOR_RGB2BGR)
    
    high_quality_buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).save(
        high_quality_buffer, format='JPEG', quality=roi_quality
    )
    high_quality_img = Image.open(high_quality_buffer)
    high_quality_cv = cv2.cvtColor(np.array(high_quality_img), cv2.COLOR_RGB2BGR)
    
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    result = (high_quality_cv * mask_3ch + low_quality_cv * (1 - mask_3ch)).astype(np.uint8)
    
    result_buffer = io.BytesIO()
    result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    result_img.save(result_buffer, format='JPEG', quality=70)
    
    return result_img, faces, len(result_buffer.getvalue())

def display_url_input_interface():
    """Display interface for loading images from URL"""
    st.markdown("""
    ### Load Image from URL
    
    **How to use:**
    1. Right-click on any web image (e.g., Google Images)
    2. Select "Copy Image Address" or "Copy Image Link"
    3. Paste the URL below
    """)
    
    url_input = st.text_input(
        "Image URL:",
        placeholder="https://example.com/image.jpg",
        help="Paste the direct URL to an image"
    )
    
    if url_input and st.button("Load Image from URL"):
        try:
            import urllib.request
            with urllib.request.urlopen(url_input) as url:
                image_data = url.read()
            
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            st.session_state['pasted_image'] = image
            st.success("Image loaded successfully from URL")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.info("Please ensure the URL points directly to an image file")

def main():
    st.title("Image Compression Analysis Tool")
    st.markdown("Professional tool for analyzing lossless and lossy compression techniques")
    
    if 'pasted_image' not in st.session_state:
        st.session_state['pasted_image'] = None
    
    # Sidebar configuration
    st.sidebar.header("Theme Configuration")
    
    theme_categories = {
        "Professional": ["Corporate Slate", "Executive Navy", "Modern Charcoal", "Professional Indigo", "Business Teal"],
        "Sophisticated": ["Elegant Emerald", "Refined Purple", "Sophisticated Rose", "Steel Blue", "Arctic Professional"],
        "Warm": ["Warm Copper", "Golden Executive"],
        "Cool": ["Deep Ocean", "Midnight Blue", "Forest Green"]
    }
    
    selected_category = st.sidebar.radio("Theme Category", list(theme_categories.keys()))
    selected_theme = st.sidebar.selectbox("Select Theme", theme_categories[selected_category])
    apply_theme(selected_theme)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Documentation")
    st.sidebar.info(
        "**Features:**\n\n"
        "Part A: Lossless Compression Analysis (PNG)\n\n"
        "Part B: Lossy Compression Comparison (JPEG, WebP)\n\n"
        "Part C: ROI-based Adaptive Compression"
    )
    
    # Image input section
    st.header("Image Input")
    
    input_method = st.radio(
        "Select input method:",
        ["File Upload", "Image URL"],
        horizontal=True
    )
    
    original_image = None
    
    if input_method == "File Upload":
        st.markdown("**Upload or drag and drop your image file**")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            help="Drag and drop an image file here, or click to browse"
        )
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')
    
    else:
        display_url_input_interface()
        
        if st.session_state.get('pasted_image') is not None:
            st.success("Image loaded from URL")
            original_image = st.session_state['pasted_image']
            
            if st.button("Clear Loaded Image"):
                st.session_state['pasted_image'] = None
                st.rerun()
    
    if original_image is not None:
        st.image(original_image, caption="Original Image", use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs(["Part A: Lossless Analysis", "Part B: Lossy Comparison", "Part C: ROI-based Compression"])
        
        with tab1:
            st.header("Lossless Compression Analysis")
            st.markdown("Analyzing PNG lossless compression using DEFLATE algorithm")
            
            col1, col2 = st.columns(2)
            with col1:
                jpeg_quality_a = st.slider("JPEG Quality for Comparison", 50, 100, 90, 5, key="jpeg_quality_a")
            with col2:
                error_margin = st.slider("Acceptable Error Margin (pixels)", 0, 10, 0, 1, key="error_margin")
            
            if st.button("Run Lossless Analysis"):
                with st.spinner("Processing compression analysis..."):
                    results = save_image_formats(original_image, quality=jpeg_quality_a)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("File Size Metrics")
                        st.metric("PNG (Lossless)", f"{results['PNG']['size']:,} bytes")
                        st.metric(f"JPEG Q{jpeg_quality_a} (Lossy)", f"{results[f'JPEG_Q{jpeg_quality_a}']['size']:,} bytes")
                        
                        compression_ratio = (1 - results[f'JPEG_Q{jpeg_quality_a}']['size'] / results['PNG']['size']) * 100
                        st.metric("Compression Savings", f"{compression_ratio:.1f}%")
                    
                    with col2:
                        st.subheader("Reconstruction Verification")
                        verification = verify_lossless_reconstruction(original_image, results['PNG']['data'])
                        
                        st.markdown("**PNG Reconstruction Process:**")
                        st.markdown("1. Original image encoded using DEFLATE compression")
                        st.markdown("2. Compressed data stored with PNG headers and CRC checksums")
                        st.markdown("3. Decompression reverses DEFLATE algorithm")
                        st.markdown("4. Pixel values reconstructed bit-for-bit")
                        
                        st.markdown("---")
                        st.markdown("**Verification Results:**")
                        st.write(f"Dimensions Match: {verification['dimensions_match']}")
                        st.write(f"Bit-Exact Match: {verification['bit_exact']}")
                        st.write(f"Maximum Pixel Difference: {verification['max_difference']}")
                        st.write(f"Mean Squared Error: {verification['mse']:.6f}")
                    
                    st.subheader("Compression Comparison")
                    
                    jpeg_verification = verify_lossless_reconstruction(original_image, results[f'JPEG_Q{jpeg_quality_a}']['data'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**PNG (Lossless)**")
                        st.write(f"Max Difference: {verification['max_difference']}")
                        st.write(f"MSE: {verification['mse']:.6f}")
                        within_margin = verification['max_difference'] <= error_margin
                        st.write(f"Within Margin: {within_margin}")
                    
                    with col2:
                        st.markdown(f"**JPEG Q{jpeg_quality_a} (Lossy)**")
                        st.write(f"Max Difference: {jpeg_verification['max_difference']}")
                        st.write(f"MSE: {jpeg_verification['mse']:.6f}")
                        within_margin = jpeg_verification['max_difference'] <= error_margin
                        st.write(f"Within Margin: {within_margin}")
                    
                    with col3:
                        st.markdown("**Quality Metrics**")
                        quality_loss = jpeg_verification['mse'] - verification['mse']
                        st.write(f"MSE Difference: {quality_loss:.6f}")
                        psnr = 10 * np.log10(255**2 / jpeg_verification['mse']) if jpeg_verification['mse'] > 0 else float('inf')
                        st.write(f"JPEG PSNR: {psnr:.2f} dB")
                        st.write(f"Size Reduction: {compression_ratio:.1f}%")
                    
                    st.subheader("Visual Comparison")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(original_image, caption="Original Image", use_container_width=True)
                    
                    with col2:
                        reconstructed_png = Image.open(io.BytesIO(results['PNG']['data']))
                        st.image(reconstructed_png, caption="PNG Reconstruction (Bit-Exact)", use_container_width=True)
                    
                    with col3:
                        reconstructed_jpeg = Image.open(io.BytesIO(results[f'JPEG_Q{jpeg_quality_a}']['data']))
                        st.image(reconstructed_jpeg, caption=f"JPEG Q{jpeg_quality_a} (Lossy)", use_container_width=True)
                    
                    st.subheader("Pixel Difference Heatmap")
                    
                    original_array = np.array(original_image)
                    jpeg_array = np.array(Image.open(io.BytesIO(results[f'JPEG_Q{jpeg_quality_a}']['data'])))
                    
                    diff = np.abs(original_array.astype(float) - jpeg_array.astype(float))
                    diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    im = ax.imshow(diff_magnitude, cmap='hot', interpolation='nearest')
                    ax.set_title(f'Pixel Difference Magnitude (Max: {np.max(diff_magnitude):.2f})')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, label='Difference Magnitude')
                    st.pyplot(fig)
                    
                    st.subheader("Analysis Summary")
                    
                    if verification['bit_exact']:
                        st.success("PNG: Perfect reconstruction - 100% bit-exact match with original")
                    else:
                        st.warning("PNG: Unexpected variation detected")
                    
                    if jpeg_verification['max_difference'] <= error_margin:
                        st.success(f"JPEG: All pixel differences within acceptable margin ({error_margin} pixels)")
                    else:
                        st.warning(f"JPEG: Some pixels exceed error margin. Max difference: {jpeg_verification['max_difference']} pixels")
                    
                    st.info(f"""
                    **Key Findings:**
                    - PNG preserves exact pixel values (lossless compression)
                    - JPEG Q{jpeg_quality_a} achieves MSE of {jpeg_verification['mse']:.4f}
                    - File size reduction of {compression_ratio:.1f}% with lossy compression
                    - PSNR: {psnr:.2f} dB (>40 dB indicates excellent quality)
                    """)
        
        with tab2:
            st.header("Lossy Compression Analysis")
            st.markdown("Comparing lossy compression formats at various quality levels")
            
            if st.button("Run Lossy Analysis"):
                with st.spinner("Processing multiple compression levels..."):
                    jpeg_qualities = [90, 70, 50, 30]
                    webp_qualities = [90, 50]
                    
                    st.subheader("JPEG Quality Levels")
                    cols = st.columns(4)
                    
                    jpeg_results = {}
                    for idx, quality in enumerate(jpeg_qualities):
                        results = save_image_formats(original_image, quality=quality)
                        jpeg_results[quality] = results[f'JPEG_Q{quality}']
                        
                        with cols[idx]:
                            img = Image.open(io.BytesIO(results[f'JPEG_Q{quality}']['data']))
                            st.image(img, caption=f"Quality {quality}", use_container_width=True)
                            st.caption(f"Size: {results[f'JPEG_Q{quality}']['size']:,} bytes")
                    
                    st.subheader("WebP Quality Comparison")
                    cols = st.columns(2)
                    
                    for idx, quality in enumerate(webp_qualities):
                        results = save_image_formats(original_image, quality=quality)
                        
                        with cols[idx]:
                            img = Image.open(io.BytesIO(results[f'WEBP_Q{quality}']['data']))
                            st.image(img, caption=f"WebP Quality {quality}", use_container_width=True)
                            st.caption(f"Size: {results[f'WEBP_Q{quality}']['size']:,} bytes")
                    
                    st.subheader("Compression Artifacts")
                    st.markdown("""
                    **Common artifacts at lower quality levels:**
                    - **Blocking**: 8x8 pixel blocks visible in JPEG compression
                    - **Blurring**: Loss of high-frequency details and edge definition
                    - **Color banding**: Reduced color gradients in smooth areas
                    - **Ringing**: Edge artifacts and oscillations near sharp transitions
                    """)
                    
                    st.subheader("File Size Analysis")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    qualities = list(jpeg_results.keys())
                    sizes = [jpeg_results[q]['size'] for q in qualities]
                    ax.plot(qualities, sizes, marker='o', linewidth=2, markersize=8, color='#3b82f6')
                    ax.set_xlabel('JPEG Quality Level')
                    ax.set_ylabel('File Size (bytes)')
                    ax.set_title('Quality vs File Size Trade-off')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        with tab3:
            st.header("ROI-based Adaptive Compression")
            st.markdown("Selective quality preservation for important image regions")
            
            col1, col2 = st.columns(2)
            with col1:
                roi_quality = st.slider("ROI Quality Level", 50, 100, 90, 5)
            with col2:
                bg_quality = st.slider("Background Quality Level", 10, 80, 40, 5)
            
            if st.button("Apply ROI Compression"):
                with st.spinner("Detecting regions of interest and applying compression..."):
                    roi_result, faces, roi_size = roi_based_compression(
                        original_image, roi_quality, bg_quality
                    )
                    
                    standard_buffer = io.BytesIO()
                    original_image.save(standard_buffer, format='JPEG', quality=60)
                    standard_size = len(standard_buffer.getvalue())
                    standard_img = Image.open(standard_buffer)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("ROI-based Compression")
                        
                        roi_display = np.array(original_image.copy())
                        for (x, y, w, h) in faces:
                            cv2.rectangle(roi_display, (x, y), (x+w, y+h), (255, 0, 0), 3)
                        st.image(roi_display, caption="Detected Regions of Interest", use_container_width=True)
                        
                        st.image(roi_result, caption=f"ROI: Q{roi_quality}, Background: Q{bg_quality}", use_container_width=True)
                        st.metric("File Size", f"{roi_size:,} bytes")
                    
                    with col2:
                        st.subheader("Standard JPEG Q60")
                        st.image(standard_img, caption="Uniform Quality 60", use_container_width=True)
                        st.metric("File Size", f"{standard_size:,} bytes")
                    
                    st.subheader("Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Detected ROI Regions", len(faces))
                    with col2:
                        savings = ((standard_size - roi_size) / standard_size) * 100
                        st.metric("Size Difference", f"{savings:+.1f}%")
                    with col3:
                        st.metric("Compression Strategy", f"ROI:{roi_quality}/BG:{bg_quality}")
                    
                    st.info(
                        "ROI-based compression maintains high quality in critical regions "
                        "(faces, text) while applying aggressive compression to background areas. "
                        "This approach optimizes the balance between file size and perceptual quality."
                    )

if __name__ == "__main__":
    main()