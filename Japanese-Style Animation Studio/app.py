# app.py
from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import time
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.style_transfer import AnimeStyleTransfer
import threading
import logging
import imageio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Set up the style transfer models
models = {
    'ghibli': AnimeStyleTransfer('models/weights/ghibli_model.pth'),
    'kyoani': AnimeStyleTransfer('models/weights/kyoani_model.pth'),
    'trigger': AnimeStyleTransfer('models/weights/trigger_model.pth'),
    'mappa': AnimeStyleTransfer('models/weights/mappa_model.pth'),
    'ufotable': AnimeStyleTransfer('models/weights/ufotable_model.pth')
}

# In-memory job tracking
jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(file_path)
        
        # Get parameters
        style = request.form.get('style', 'ghibli').lower()
        animated = request.form.get('animated', 'false').lower() == 'true'
        frames = int(request.form.get('frames', '24'))
        transition = request.form.get('transition', 'false').lower() == 'true'
        
        # Style validation
        if style not in models:
            return jsonify({'error': 'Invalid style selected'}), 400
        
        # Set up job tracking
        jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'file_path': file_path,
            'style': style,
            'animated': animated,
            'frames': frames,
            'transition': transition,
            'result_paths': []
        }
        
        # Start processing in background
        threading.Thread(target=process_image, args=(job_id,)).start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing'
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

def process_image(job_id):
    """Process the uploaded image with the selected style(s)"""
    job = jobs[job_id]
    
    try:
        # Load image
        image = Image.open(job['file_path']).convert('RGB')
        
        # Get base filename for results
        base_filename = os.path.basename(job['file_path']).split('_', 1)[1]
        filename_without_ext = os.path.splitext(base_filename)[0]
        
        if job['animated']:
            # Process animation
            result_files = create_animation(
                job_id, 
                image, 
                job['style'], 
                job['frames'], 
                job['transition']
            )
            job['result_paths'] = result_files
        else:
            # Single image processing
            output_path = os.path.join(RESULTS_FOLDER, f"{job_id}_{filename_without_ext}_{job['style']}.png")
            
            # Apply style transfer
            styled_image = models[job['style']].transfer_style(image)
            styled_image.save(output_path)
            
            job['result_paths'] = [output_path]
        
        job['status'] = 'completed'
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        logger.error(f"Error processing job {job_id}: {e}")

def create_animation(job_id, image, style, frames, transition):
    """Create an animation with the specified parameters"""
    job = jobs[job_id]
    result_files = []
    
    # Set up base filename
    base_filename = os.path.basename(job['file_path']).split('_', 1)[1]
    filename_without_ext = os.path.splitext(base_filename)[0]
    
    # Create frames for the animation
    animation_frames = []
    
    if transition:
        # Get all styles for transition
        styles = list(models.keys())
        # If the selected style exists, start with it
        if style in styles:
            # Move the selected style to the front
            styles.remove(style)
            styles.insert(0, style)
    else:
        # Just use the selected style
        styles = [style]
    
    # Generate all frames
    for frame_idx in range(frames):
        # Update progress
        job['progress'] = int((frame_idx / frames) * 100)
        
        if transition:
            # Calculate which style to use based on frame position
            style_idx = int((frame_idx / frames) * len(styles))
            # Ensure we don't go out of bounds
            style_idx = min(style_idx, len(styles) - 1)
            current_style = styles[style_idx]
        else:
            current_style = style
            
        # Apply the style transfer
        styled_frame = models[current_style].transfer_style(image)
        
        # Convert to numpy array for saving with imageio
        frame_array = np.array(styled_frame)
        animation_frames.append(frame_array)
        
        # Save individual frames too
        if frame_idx % 5 == 0 or frame_idx == frames - 1:  # Save every 5th frame and the last one
            frame_path = os.path.join(RESULTS_FOLDER, f"{job_id}_{filename_without_ext}_frame_{frame_idx}.png")
            styled_frame.save(frame_path)
            result_files.append(frame_path)
    
    # Save as GIF
    gif_path = os.path.join(RESULTS_FOLDER, f"{job_id}_{filename_without_ext}_animation.gif")
    imageio.mimsave(gif_path, animation_frames, fps=12)
    result_files.append(gif_path)
    
    # Save as MP4
    mp4_path = os.path.join(RESULTS_FOLDER, f"{job_id}_{filename_without_ext}_animation.mp4")
    imageio.mimsave(mp4_path, animation_frames, fps=24)
    result_files.append(mp4_path)
    
    return result_files

@app.route('/api/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check the status of a processing job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'progress': job['progress']
    }
    
    if job['status'] == 'completed':
        # Get just the filenames, not full paths
        result_files = [os.path.basename(path) for path in job['result_paths']]
        response['result_files'] = result_files
    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    """Serve the processed result files"""
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    """Get a list of public gallery items"""
    # In a real implementation, this would query a database
    # Here we're just returning mock data
    gallery_items = [
        {
            'id': '1',
            'title': 'Mountain Landscape in Ghibli Style',
            'style': 'ghibli',
            'username': 'artlover123',
            'likes': 245,
            'image_url': '/api/gallery/sample1.jpg'
        },
        {
            'id': '2',
            'title': 'City Street in KyoAni Style',
            'style': 'kyoani',
            'username': 'animeartist',
            'likes': 189,
            'image_url': '/api/gallery/sample2.jpg'
        },
        {
            'id': '3',
            'title': 'Beach Sunset in Ufotable Style',
            'style': 'ufotable',
            'username': 'sunsetlover',
            'likes': 302,
            'image_url': '/api/gallery/sample3.jpg'
        },
        {
            'id': '4',
            'title': 'Forest Path in MAPPA Style',
            'style': 'mappa',
            'username': 'naturewalker',
            'likes': 156,
            'image_url': '/api/gallery/sample4.jpg'
        },
        {
            'id': '5',
            'title': 'Urban Cityscape in Trigger Style',
            'style': 'trigger',
            'username': 'cityscaper',
            'likes': 278,
            'image_url': '/api/gallery/sample5.jpg'
        },
        {
            'id': '6',
            'title': 'Portrait in Ghibli Style',
            'style': 'ghibli',
            'username': 'portraitmaster',
            'likes': 412,
            'image_url': '/api/gallery/sample6.jpg'
        }
    ]
    
    return jsonify(gallery_items)

# Clean up old files periodically
def cleanup_old_files():
    """Delete files older than 24 hours"""
    while True:
        now = time.time()
        for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                # If file is older than 24 hours (86400 seconds)
                if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > 86400:
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {e}")
        
        # Wait for 1 hour before checking again
        time.sleep(3600)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
