import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import io

# Page configuration
st.set_page_config(
    page_title="Media Processing Suite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎬 Media Processing Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Image and Video Processing Tools</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Processing Mode")
    processing_mode = st.radio(
        "Select Mode",
        ["Video Processing", "Image Processing"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if processing_mode == "Video Processing":
        st.markdown("### 🎥 Video Operations")
        video_operation = st.selectbox(
            "Select Operation",
            ["Video Stabilization", "Object Tracking", "Motion Detection", "Background Removal"]
        )
    else:
        st.markdown("### 🖼️ Image Operations")
        image_operation = st.selectbox(
            "Select Operation",
            ["Block Matching", "Edge Detection", "Feature Matching"]
        )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("Advanced computer vision processing using OpenCV. All processing happens locally.")

# Helper function to save video properly
def save_video_with_frames(frames, fps, output_path, size):
    """Save frames to video file with proper codec"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    if not out.isOpened():
        raise Exception("Failed to open video writer")
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

# Main content area
if processing_mode == "Video Processing":
    st.markdown('<div class="section-header">📹 Video Processing</div>', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_video is not None:
        # Save uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        video_path = tfile.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Original Video")
            st.video(video_path)
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            st.info(f"**Resolution:** {width}x{height}\n\n**FPS:** {fps}\n\n**Duration:** {duration:.2f}s\n\n**Frames:** {frame_count}")
        
        with col2:
            st.markdown(f"#### ⚙️ {video_operation}")
            
            # Processing parameters
            max_frames = st.slider("Max frames to process (for speed)", 30, 300, 150, 30)
            
            if video_operation == "Video Stabilization":
                st.markdown('<div class="info-box">🎯 Stabilizes shaky video using optical flow tracking.</div>', unsafe_allow_html=True)
                
                smoothing = st.slider("Smoothing strength", 5, 50, 30, 5)
                
                if st.button("🚀 Apply Stabilization", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        
                        # Read first frame
                        ret, prev_frame = cap.read()
                        if not ret:
                            raise Exception("Cannot read video")
                        
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        transforms = []
                        frames = [prev_frame]
                        
                        frame_idx = 0
                        status_text.text("📊 Analyzing motion...")
                        
                        while frame_idx < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            # Detect features
                            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                                               qualityLevel=0.01, minDistance=30)
                            
                            if prev_pts is not None:
                                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                                
                                if curr_pts is not None and status is not None:
                                    idx = np.where(status == 1)[0]
                                    if len(idx) > 0:
                                        prev_pts = prev_pts[idx]
                                        curr_pts = curr_pts[idx]
                                        
                                        if len(prev_pts) >= 4:
                                            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                                            if m is not None:
                                                dx = m[0, 2]
                                                dy = m[1, 2]
                                                da = np.arctan2(m[1, 0], m[0, 0])
                                            else:
                                                dx = dy = da = 0
                                        else:
                                            dx = dy = da = 0
                                    else:
                                        dx = dy = da = 0
                                else:
                                    dx = dy = da = 0
                            else:
                                dx = dy = da = 0
                            
                            transforms.append([dx, dy, da])
                            frames.append(frame)
                            prev_gray = gray.copy()
                            frame_idx += 1
                            
                            progress_bar.progress(frame_idx / max_frames * 0.5)
                        
                        cap.release()
                        
                        status_text.text("🔧 Applying stabilization...")
                        
                        # Calculate smooth trajectory
                        trajectory = np.cumsum(transforms, axis=0)
                        smoothed_trajectory = np.copy(trajectory)
                        
                        for i in range(3):
                            for j in range(len(trajectory)):
                                start = max(0, j - smoothing)
                                end = min(len(trajectory), j + smoothing + 1)
                                smoothed_trajectory[j, i] = np.mean(trajectory[start:end, i])
                        
                        difference = smoothed_trajectory - trajectory
                        transforms_smooth = transforms + difference
                        
                        # Apply transforms
                        stabilized_frames = []
                        for i, frame in enumerate(frames):
                            if i == 0:
                                stabilized_frames.append(frame)
                            else:
                                dx, dy, da = transforms_smooth[i - 1]
                                
                                m = np.array([[np.cos(da), -np.sin(da), dx],
                                             [np.sin(da), np.cos(da), dy]], dtype=np.float32)
                                
                                stabilized = cv2.warpAffine(frame, m, (width, height),
                                                           borderMode=cv2.BORDER_REFLECT)
                                stabilized_frames.append(stabilized)
                            
                            progress_bar.progress(0.5 + (i / len(frames)) * 0.5)
                        
                        status_text.text("💾 Saving video...")
                        
                        # Save video
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        save_video_with_frames(stabilized_frames, fps, output_path, (width, height))
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        st.markdown('<div class="success-box">✅ Stabilization completed!</div>', unsafe_allow_html=True)
                        
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.video(output_path)
                        st.download_button(
                            label="📥 Download Stabilized Video",
                            data=video_bytes,
                            file_name="stabilized_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                        
                        os.unlink(output_path)
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            
            elif video_operation == "Object Tracking":
                st.markdown('<div class="info-box">🎯 Track moving objects across frames.</div>', unsafe_allow_html=True)
                
                tracker_type = st.selectbox("Tracker Algorithm", ["CSRT", "KCF", "MIL"])
                
                if st.button("🚀 Start Tracking", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        ret, first_frame = cap.read()
                        
                        if not ret:
                            raise Exception("Cannot read video")
                        
                        # Auto-select center region
                        h, w = first_frame.shape[:2]
                        bbox = (w//3, h//3, w//3, h//3)
                        
                        # Initialize tracker
                        if tracker_type == 'CSRT':
                            tracker = cv2.TrackerCSRT_create()
                        elif tracker_type == 'KCF':
                            tracker = cv2.TrackerKCF_create()
                        else:
                            tracker = cv2.TrackerMIL_create()
                        
                        tracker.init(first_frame, bbox)
                        
                        status_text.text("🔍 Tracking object...")
                        
                        tracked_frames = [first_frame]
                        frame_idx = 0
                        
                        while frame_idx < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            success, bbox = tracker.update(frame)
                            
                            if success:
                                x, y, w, h = [int(v) for v in bbox]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(frame, f"{tracker_type} Tracker", (x, y - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "Tracking Lost", (50, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            
                            tracked_frames.append(frame)
                            frame_idx += 1
                            progress_bar.progress(frame_idx / max_frames)
                        
                        cap.release()
                        
                        status_text.text("💾 Saving video...")
                        
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        save_video_with_frames(tracked_frames, fps, output_path, (width, height))
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        st.markdown('<div class="success-box">✅ Tracking completed!</div>', unsafe_allow_html=True)
                        
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.video(output_path)
                        st.download_button(
                            label="📥 Download Tracked Video",
                            data=video_bytes,
                            file_name="tracked_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                        
                        os.unlink(output_path)
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            
            elif video_operation == "Motion Detection":
                st.markdown('<div class="info-box">🏃 Detect and highlight motion in video.</div>', unsafe_allow_html=True)
                
                sensitivity = st.slider("Detection sensitivity", 10, 50, 30, 5)
                
                if st.button("🚀 Detect Motion", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        ret, prev_frame = cap.read()
                        
                        if not ret:
                            raise Exception("Cannot read video")
                        
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
                        
                        status_text.text("🔍 Analyzing motion...")
                        
                        motion_frames = [prev_frame]
                        frame_idx = 0
                        
                        while frame_idx < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            gray = cv2.GaussianBlur(gray, (21, 21), 0)
                            
                            # Calculate difference
                            frame_delta = cv2.absdiff(prev_gray, gray)
                            thresh = cv2.threshold(frame_delta, sensitivity, 255, cv2.THRESH_BINARY)[1]
                            thresh = cv2.dilate(thresh, None, iterations=2)
                            
                            # Find contours
                            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE)
                            
                            motion_detected = False
                            for contour in contours:
                                if cv2.contourArea(contour) > 500:
                                    motion_detected = True
                                    x, y, w, h = cv2.boundingRect(contour)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Add status text
                            status = "MOTION DETECTED" if motion_detected else "No Motion"
                            color = (0, 0, 255) if motion_detected else (0, 255, 0)
                            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, color, 2)
                            
                            motion_frames.append(frame)
                            prev_gray = gray
                            frame_idx += 1
                            progress_bar.progress(frame_idx / max_frames)
                        
                        cap.release()
                        
                        status_text.text("💾 Saving video...")
                        
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        save_video_with_frames(motion_frames, fps, output_path, (width, height))
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        st.markdown('<div class="success-box">✅ Motion detection completed!</div>', unsafe_allow_html=True)
                        
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.video(output_path)
                        st.download_button(
                            label="📥 Download Analyzed Video",
                            data=video_bytes,
                            file_name="motion_detected_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                        
                        os.unlink(output_path)
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            
            elif video_operation == "Background Removal":
                st.markdown('<div class="info-box">🎭 Remove or replace video background.</div>', unsafe_allow_html=True)
                
                bg_method = st.selectbox("Method", ["MOG2", "KNN"])
                replace_bg = st.checkbox("Replace with color")
                
                if replace_bg:
                    bg_color = st.color_picker("Background Color", "#00FF00")
                
                if st.button("🚀 Process Background", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        
                        if bg_method == "MOG2":
                            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                                history=500, varThreshold=16, detectShadows=True)
                        else:
                            bg_subtractor = cv2.createBackgroundSubtractorKNN(
                                history=500, dist2Threshold=400, detectShadows=True)
                        
                        status_text.text("🎭 Removing background...")
                        
                        processed_frames = []
                        frame_idx = 0
                        
                        while frame_idx < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            fg_mask = bg_subtractor.apply(frame)
                            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
                            
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                            
                            fg_mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                            
                            if replace_bg:
                                hex_color = bg_color.lstrip('#')
                                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                bgr = (rgb[2], rgb[1], rgb[0])
                                background = np.full_like(frame, bgr, dtype=np.uint8)
                                result = np.where(fg_mask_3ch == 255, frame, background)
                            else:
                                result = cv2.bitwise_and(frame, fg_mask_3ch)
                            
                            processed_frames.append(result)
                            frame_idx += 1
                            progress_bar.progress(frame_idx / max_frames)
                        
                        cap.release()
                        
                        status_text.text("💾 Saving video...")
                        
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        save_video_with_frames(processed_frames, fps, output_path, (width, height))
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        st.markdown('<div class="success-box">✅ Background removal completed!</div>', unsafe_allow_html=True)
                        
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.video(output_path)
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name="background_removed_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                        
                        os.unlink(output_path)
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
        
        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass

else:  # Image Processing
    st.markdown('<div class="section-header">🖼️ Image Processing</div>', unsafe_allow_html=True)
    
    if image_operation == "Block Matching":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Reference Image")
            ref_image = st.file_uploader("Upload Reference", type=['jpg', 'jpeg', 'png'], key='ref')
        
        with col2:
            st.markdown("#### 📥 Target Image")
            target_image = st.file_uploader("Upload Target", type=['jpg', 'jpeg', 'png'], key='target')
        
        if ref_image and target_image:
            ref_img = Image.open(ref_image)
            target_img = Image.open(target_image)
            
            ref_cv = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
            target_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(ref_img, caption="Reference", use_column_width=True)
            
            with col2:
                st.image(target_img, caption="Target", use_column_width=True)
            
            st.markdown("#### ⚙️ Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                block_size = st.slider("Block Size", 8, 64, 16, 8)
            
            with col2:
                search_area = st.slider("Search Area", 8, 64, 32, 8)
            
            with col3:
                method = st.selectbox("Method", ["SAD", "SSD", "NCC"])
            
            if st.button("🚀 Perform Block Matching", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        ref_gray = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY)
                        target_gray = cv2.cvtColor(target_cv, cv2.COLOR_BGR2GRAY)
                        
                        h = min(ref_gray.shape[0], target_gray.shape[0])
                        w = min(ref_gray.shape[1], target_gray.shape[1])
                        ref_gray = ref_gray[:h, :w]
                        target_gray = target_gray[:h, :w]
                        
                        result_img = target_cv[:h, :w].copy()
                        motion_vectors = []
                        
                        for i in range(0, h - block_size, block_size):
                            for j in range(0, w - block_size, block_size):
                                ref_block = ref_gray[i:i+block_size, j:j+block_size]
                                
                                min_diff = float('inf')
                                best_match = (0, 0)
                                
                                for di in range(-search_area, search_area, 4):
                                    for dj in range(-search_area, search_area, 4):
                                        ti = i + di
                                        tj = j + dj
                                        
                                        if 0 <= ti < h - block_size and 0 <= tj < w - block_size:
                                            target_block = target_gray[ti:ti+block_size, tj:tj+block_size]
                                            
                                            if method == "SAD":
                                                diff = np.sum(np.abs(ref_block.astype(float) - target_block.astype(float)))
                                            elif method == "SSD":
                                                diff = np.sum((ref_block.astype(float) - target_block.astype(float)) ** 2)
                                            else:  # NCC
                                                diff = -np.sum((ref_block - np.mean(ref_block)) * 
                                                             (target_block - np.mean(target_block)))
                                            
                                            if diff < min_diff:
                                                min_diff = diff
                                                best_match = (di, dj)
                                
                                motion_vectors.append(best_match)
                                
                                center_x = j + block_size // 2
                                center_y = i + block_size // 2
                                end_x = center_x + best_match[1]
                                end_y = center_y + best_match[0]
                                
                                cv2.arrowedLine(result_img, (center_x, center_y), (end_x, end_y),
                                              (0, 255, 0), 2, tipLength=0.3)
                                cv2.rectangle(result_img, (j, i), (j+block_size, i+block_size),
                                            (255, 0, 0), 1)
                        
                        st.markdown('<div class="success-box">✅ Block matching completed!</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Motion Vectors")
                            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            st.image(result_rgb, use_column_width=True)
                            
                            result_pil = Image.fromarray(result_rgb)
                            buf = io.BytesIO()
                            result_pil.save(buf, format='PNG')
                            st.download_button(
                                label="📥 Download Result",
                                data=buf.getvalue(),
                                file_name="block_matching_result.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        with col2:
                            st.markdown("#### 📈 Statistics")
                            motion_mags = [np.sqrt(mv[0]**2 + mv[1]**2) for mv in motion_vectors]
                            st.metric("Average Motion", f"{np.mean(motion_mags):.2f} px")
                            st.metric("Max Motion", f"{np.max(motion_mags):.2f} px")
                            st.metric("Blocks Processed", len(motion_vectors))
                    
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    
    elif image_operation == "Edge Detection":
        st.markdown('<div class="info-box">🔍 Detect edges in images using various algorithms.</div>', unsafe_allow_html=True)
        
        uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_img:
            img = Image.open(uploaded_img)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img, caption="Original", use_column_width=True)
            
            with col2:
                method = st.selectbox("Edge Detection Method", ["Canny", "Sobel", "Laplacian"])
                
                if method == "Canny":
                    threshold1 = st.slider("Lower Threshold", 0, 255, 50)
                    threshold2 = st.slider("Upper Threshold", 0, 255, 150)
                
                if st.button("🚀 Detect Edges", use_container_width=True):
                    with st.spinner("Processing..."):
                        try:
                            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                            
                            if method == "Canny":
                                edges = cv2.Canny(gray, threshold1, threshold2)
                            elif method == "Sobel":
                                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                                edges = np.sqrt(sobelx**2 + sobely**2)
                                edges = np.uint8(edges / edges.max() * 255)
                            else:  # Laplacian
                                edges = cv2.Laplacian(gray, cv2.CV_64F)
                                edges = np.uint8(np.absolute(edges))
                            
                            st.image(edges, caption=f"{method} Edges", use_column_width=True)
                            
                            result_pil = Image.fromarray(edges)
                            buf = io.BytesIO()
                            result_pil.save(buf, format='PNG')
                            st.download_button(
                                label="📥 Download Result",
                                data=buf.getvalue(),
                                file_name=f"edges_{method.lower()}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    
    elif image_operation == "Feature Matching":
        st.markdown('<div class="info-box">🔗 Match features between two images using keypoint detection.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            img1_file = st.file_uploader("Upload First Image", type=['jpg', 'jpeg', 'png'], key='feat1')
        
        with col2:
            img2_file = st.file_uploader("Upload Second Image", type=['jpg', 'jpeg', 'png'], key='feat2')
        
        if img1_file and img2_file:
            img1 = Image.open(img1_file)
            img2 = Image.open(img2_file)
            
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img1, caption="Image 1", use_column_width=True)
            
            with col2:
                st.image(img2, caption="Image 2", use_column_width=True)
            
            detector_type = st.selectbox("Feature Detector", ["ORB", "SIFT", "AKAZE"])
            max_matches = st.slider("Max matches to display", 10, 100, 50, 10)
            
            if st.button("🚀 Match Features", use_container_width=True):
                with st.spinner("Matching features..."):
                    try:
                        gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
                        
                        if detector_type == "ORB":
                            detector = cv2.ORB_create(nfeatures=1000)
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        elif detector_type == "SIFT":
                            detector = cv2.SIFT_create()
                            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                        else:  # AKAZE
                            detector = cv2.AKAZE_create()
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        
                        kp1, des1 = detector.detectAndCompute(gray1, None)
                        kp2, des2 = detector.detectAndCompute(gray2, None)
                        
                        if des1 is not None and des2 is not None:
                            matches = bf.match(des1, des2)
                            matches = sorted(matches, key=lambda x: x.distance)
                            matches = matches[:max_matches]
                            
                            result = cv2.drawMatches(img1_cv, kp1, img2_cv, kp2, matches, None,
                                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                            
                            st.markdown('<div class="success-box">✅ Feature matching completed!</div>', unsafe_allow_html=True)
                            
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            st.image(result_rgb, caption=f"Matched Features ({len(matches)} matches)", 
                                   use_column_width=True)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Keypoints (Image 1)", len(kp1))
                            with col2:
                                st.metric("Keypoints (Image 2)", len(kp2))
                            with col3:
                                st.metric("Good Matches", len(matches))
                            
                            result_pil = Image.fromarray(result_rgb)
                            buf = io.BytesIO()
                            result_pil.save(buf, format='PNG')
                            st.download_button(
                                label="📥 Download Result",
                                data=buf.getvalue(),
                                file_name="feature_matching_result.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        else:
                            st.warning("⚠️ No features detected in one or both images.")
                    
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p><strong>Media Processing Suite</strong> | Advanced Computer Vision Tools</p>
    <p>Built with OpenCV • Streamlit • Python</p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
        💡 Tip: All processing happens locally on your machine
    </p>
</div>
""", unsafe_allow_html=True)