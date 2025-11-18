import streamlit as st
from PIL import Image, ImageEnhance
import io
import tempfile
import os
import numpy as np

# To handle video processing, we'll use opencv-python.
# The user will need to install this: `pip install opencv-python`
# We will check if it's available.
try:
    import cv2
except ImportError:
    st.error("The 'opencv-python' library is not installed. Please install it by running `pip install opencv-python` in your terminal.")
    st.stop()


def process_image(img, options):
    """
    Applies editing options to an image using the Pillow library.
    
    Args:
        img (PIL.Image.Image): The input image.
        options (dict): A dictionary of editing options.
    
    Returns:
        PIL.Image.Image: The processed image.
    """
    # Create a mutable copy to avoid modifying the original
    processed_img = img.copy()

    if options.get("grayscale", False):
        processed_img = processed_img.convert("L")

    if options.get("resize"):
        new_width, new_height = options["resize"]
        resize_filter = options.get("resize_filter", Image.Resampling.LANCZOS)
        # Using the selected resampling filter to prevent quality loss during resizing
        processed_img = processed_img.resize((new_width, new_height), resize_filter)

    if options.get("flip_h", False):
        processed_img = processed_img.transpose(Image.FLIP_LEFT_RIGHT)

    if options.get("flip_v", False):
        processed_img = processed_img.transpose(Image.FLIP_TOP_BOTTOM)

    if options.get("rotate"):
        angle = options["rotate"]
        processed_img = processed_img.rotate(angle, expand=True)
    
    if options.get("brightness"):
        enhancer = ImageEnhance.Brightness(processed_img)
        processed_img = enhancer.enhance(options["brightness"])
    
    return processed_img


def process_video(video_file, options):
    """
    Applies editing options to a video using the opencv-python library.
    
    Args:
        video_file (UploadedFile): The uploaded video file object.
        options (dict): A dictionary of editing options.
        
    Returns:
        str: The path to the processed video file, or None if processing fails.
    """
    try:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise IOError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine output dimensions for the writer
        if options.get("resize"):
            new_width, new_height = options["resize"]
        else:
            new_width, new_height = original_width, original_height
        
        # Use a temporary file for the output video
        output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        # Using 'mp4v' codec, which is a common and relatively high-quality choice.
        # For truly lossless video, a different codec and more advanced configuration would be needed.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_temp_file, fourcc, fps, (new_width, new_height))

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert OpenCV frame (BGR) to Pillow Image (RGB)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Apply image processing functions
            processed_pil_image = process_image(pil_image, options)
            
            # Convert processed Pillow Image back to OpenCV frame (BGR)
            processed_frame = cv2.cvtColor(np.array(processed_pil_image), cv2.COLOR_RGB2BGR)

            out.write(processed_frame)

        cap.release()
        out.release()
        os.unlink(temp_path) # Clean up the original temporary file
        return output_temp_file

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        return None


def main():
    """The main function for the Streamlit app."""
    
    # Custom CSS for gradients and dark/light theme
    dark_mode_css = """
    <style>
    body {
        color: #f1f1f1;
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .st-eb, .st-ex, .st-ey {
        background: linear-gradient(145deg, #1f2733, #151a23);
        box-shadow: 5px 5px 10px #0c0f14, -5px -5px 10px #1c212a;
        color: #f1f1f1;
    }
    .stButton>button {
        background: linear-gradient(145deg, #1f2733, #151a23);
        box-shadow: 5px 5px 10px #0c0f14, -5px -5px 10px #1c212a;
        color: #f1f1f1;
        border: none;
        border-radius: 8px;
    }
    </style>
    """
    
    light_mode_css = """
    <style>
    body {
        color: #333333;
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .st-eb, .st-ex, .st-ey {
        background: linear-gradient(145deg, #e6e9f0, #f0f2f6);
        box-shadow: 5px 5px 10px #d8dbe2, -5px -5px 10px #ffffff;
        color: #333333;
    }
    .stButton>button {
        background: linear-gradient(145deg, #e6e9f0, #f0f2f6);
        box-shadow: 5px 5px 10px #d8dbe2, -5px -5px 10px #ffffff;
        color: #333333;
        border: none;
        border-radius: 8px;
    }
    </style>
    """
    
    # Set the page configuration
    st.set_page_config(page_title="Media Editor", layout="wide")

    # Theme selector in the sidebar
    st.sidebar.header("Theme")
    theme_choice = st.sidebar.radio("Select a theme:", ("Dark", "Light"))
    
    if theme_choice == "Dark":
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        st.markdown(light_mode_css, unsafe_allow_html=True)
        
    st.title("Image and Video Editor")
    st.markdown("A simple, professional-style editor for your images and videos.")

    st.sidebar.header("Media and Files")
    media_type = st.sidebar.radio("Select Media Type:", ("Image", "Video"))
    
    uploaded_file = st.sidebar.file_uploader(f"Upload a {media_type.lower()}", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

    if uploaded_file is not None:
        
        # Initialize session state for resize values if they don't exist
        if 'resized_width' not in st.session_state:
            st.session_state.resized_width = None
        if 'resized_height' not in st.session_state:
            st.session_state.resized_height = None
        if 'resize_filter_name' not in st.session_state:
            st.session_state.resize_filter_name = "LANCZOS"
            
        # Layout the editing options in the main content area
        st.header("Editing Tools")
        
        # Grayscale and Brightness (Live edits)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Grayscale")
            grayscale_option = st.checkbox("Convert to Grayscale")
        with col2:
            st.subheader("Brightness")
            brightness_factor = st.slider("Adjust Brightness", 0.1, 3.0, 1.0)
        
        st.markdown("---")

        # Resize (Form-based edit)
        with st.expander("Resize", expanded=False):
            st.write("Change the dimensions of your media. Click 'Apply' to see changes.")
            with st.form("resize_form"):
                col_resize1, col_resize2 = st.columns(2)
                with col_resize1:
                    new_width = st.number_input("Width", value=st.session_state.resized_width if st.session_state.resized_width is not None else 500, min_value=1)
                    resize_filter_name_form = st.selectbox("Resampling Filter", ("LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"), index=("LANCZOS", "BICUBIC", "BILINEAR", "NEAREST").index(st.session_state.resize_filter_name))
                with col_resize2:
                    new_height = st.number_input("Height", value=st.session_state.resized_height if st.session_state.resized_height is not None else 500, min_value=1)
                
                resize_button = st.form_submit_button("Apply Resize")
            
            if resize_button:
                st.session_state.resized_width = new_width
                st.session_state.resized_height = new_height
                st.session_state.resize_filter_name = resize_filter_name_form
                st.rerun()  # Fixed: Changed from st.experimental_rerun() to st.rerun()


        st.markdown("---")

        # Flip (Live edits)
        with st.expander("Flip", expanded=False):
            st.write("Flip your media horizontally or vertically.")
            col_flip1, col_flip2 = st.columns(2)
            with col_flip1:
                flip_h = st.checkbox("Flip Horizontal")
            with col_flip2:
                flip_v = st.checkbox("Flip Vertical")

        st.markdown("---")
        
        # Rotate (Live edits)
        with st.expander("Rotate", expanded=False):
            st.write("Rotate your media to any angle.")
            rotate_angle = st.slider("Rotate Angle ($^\circ$)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)
        
        st.markdown("---")

        # Create the options dictionary from the current widget values
        filter_map = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "BICUBIC": Image.Resampling.BICUBIC,
            "BILINEAR": Image.Resampling.BILINEAR,
            "NEAREST": Image.Resampling.NEAREST
        }
        
        # Build options for process_image
        options = {
            "grayscale": grayscale_option,
            "brightness": brightness_factor,
            "resize": (st.session_state.resized_width, st.session_state.resized_height) if st.session_state.resized_width is not None else None,
            "resize_filter": filter_map[st.session_state.resize_filter_name] if st.session_state.resize_filter_name is not None else None,
            "flip_h": flip_h,
            "flip_v": flip_v,
            "rotate": rotate_angle
        }


        st.header(f"Original {media_type}")
        
        if media_type == "Image":
            image_bytes = uploaded_file.read()
            original_image = Image.open(io.BytesIO(image_bytes))
            
            # If resized_width is not set, set it to the original image dimensions
            if st.session_state.resized_width is None:
                st.session_state.resized_width = original_image.width
                st.session_state.resized_height = original_image.height
                
            st.image(original_image, caption="Original Image", use_column_width=True)

            st.header("Processed Image")
            processed_image = process_image(original_image.copy(), options)
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            
            # Download button for the processed image
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )

        elif media_type == "Video":
            st.video(uploaded_file)
            
            st.header("Processed Video")
            with st.spinner("Processing video... This may take a moment."):
                output_path = process_video(uploaded_file, options)
            
            if output_path:
                st.video(output_path)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
                os.unlink(output_path)
        
    else:
        st.info(f"Please upload a {media_type.lower()} to get started.")

if __name__ == "__main__":
    main()