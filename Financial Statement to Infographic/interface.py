"""
Financial Animation Web Interface
--------------------------------
A simple web interface for using the Financial Statement Animation Engine.
"""

import os
import tempfile
import base64
from flask import Flask, request, render_template, send_file, redirect, url_for
import pandas as pd
from financial_animation_engine import FinancialAnimator

app = Flask(__name__)

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Create a simple HTML template
with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Financial Statement Animation Engine</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            background-color: #f4f4f8;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="file"], 
        input[type="number"],
        select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 100%;
            box-sizing: border-box;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .preview {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
        .preview h2 {
            margin-top: 0;
            color: #2c3e50;
        }
        video {
            width: 100%;
            border-radius: 4px;
            margin-top: 10px;
        }
        .sample-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .sample-button {
            background-color: #27ae60;
        }
        .sample-button:hover {
            background-color: #219653;
        }
        .alert {
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 15px;
            color: white;
        }
        .alert-success {
            background-color: #2ecc71;
        }
        .alert-error {
            background-color: #e74c3c;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Financial Statement Animation Engine</h1>
        
        {% if message %}
        <div class="alert alert-{{ message_type }}">
            {{ message }}
        </div>
        {% endif %}
        
        <div class="sample-buttons">
            <form action="/sample" method="post">
                <input type="hidden" name="sample_type" value="income">
                <button type="submit" class="sample-button">Try Income Statement Sample</button>
            </form>
            <form action="/sample" method="post">
                <input type="hidden" name="sample_type" value="balance">
                <button type="submit" class="sample-button">Try Balance Sheet Sample</button>
            </form>
            <form action="/sample" method="post">
                <input type="hidden" name="sample_type" value="cash_flow">
                <button type="submit" class="sample-button">Try Cash Flow Sample</button>
            </form>
        </div>
        
        <form action="/upload" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="csv_file">Upload Financial CSV File:</label>
                <input type="file" id="csv_file" name="csv_file" accept=".csv" required>
            </div>
            
            <div class="form-group">
                <label for="duration">Animation Duration (seconds):</label>
                <input type="number" id="duration" name="duration" min="5" max="60" value="15">
            </div>
            
            <div class="form-group">
                <label for="fps">Frames Per Second:</label>
                <input type="number" id="fps" name="fps" min="10" max="30" value="24">
            </div>
            
            <button type="submit">Generate Animation</button>
        </form>
        
        {% if video_path %}
        <div class="preview">
            <h2>Generated Animation</h2>
            <video controls>
                <source src="{{ video_path }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <p><a href="{{ video_path }}" download="financial_animation.mp4">Download Animation</a></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
    """)

# Directory for storing uploaded files and generated videos
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'financial_animations')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sample', methods=['POST'])
def process_sample():
    """Process a sample dataset"""
    sample_type = request.form.get('sample_type')
    
    # Generate a unique file name for this sample
    sample_csv_path = os.path.join(UPLOAD_FOLDER, f"sample_{sample_type}_{base64.urlsafe_b64encode(os.urandom(6)).decode()}.csv")
    output_path = os.path.join(UPLOAD_FOLDER, f"animation_{base64.urlsafe_b64encode(os.urandom(6)).decode()}.mp4")
    
    # Create the appropriate sample file
    if sample_type == 'income':
        from financial_animation_engine import create_sample_csv
        create_sample_csv(sample_csv_path)
    elif sample_type == 'balance':
        from financial_animation_engine import create_sample_balance_sheet
        create_sample_balance_sheet(sample_csv_path)
    elif sample_type == 'cash_flow':
        from financial_animation_engine import create_sample_cash_flow
        create_sample_cash_flow(sample_csv_path)
    else:
        return render_template('index.html', message="Invalid sample type", message_type="error")
    
    try:
        # Create animator and generate animation
        animator = FinancialAnimator(csv_path=sample_csv_path)
        statement_type = animator.prepare_data()
        animator.create_animated_infographic(output_path=output_path, fps=24, duration=15)
        
        # Return the template with the video path
        video_url = url_for('get_video', filename=os.path.basename(output_path))
        return render_template('index.html', 
                               video_path=video_url,
                               message=f"Generated animation from sample {statement_type} data",
                               message_type="success")
    
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}", message_type="error")


@app.route('/upload', methods=['POST'])
def upload_file():
    """Process an uploaded CSV file"""
    if 'csv_file' not in request.files:
        return render_template('index.html', message="No file part", message_type="error")
    
    file = request.files['csv_file']
    
    if file.filename == '':
        return render_template('index.html', message="No file selected", message_type="error")
    
    if not file.filename.endswith('.csv'):
        return render_template('index.html', message="File must be a CSV", message_type="error")
    
    # Get other form parameters
    try:
        duration = int(request.form.get('duration', 15))
        fps = int(request.form.get('fps', 24))
    except ValueError:
        return render_template('index.html', message="Invalid duration or fps value", message_type="error")
    
    # Generate unique filenames
    csv_path = os.path.join(UPLOAD_FOLDER, f"upload_{base64.urlsafe_b64encode(os.urandom(6)).decode()}.csv")
    output_path = os.path.join(UPLOAD_FOLDER, f"animation_{base64.urlsafe_b64encode(os.urandom(6)).decode()}.mp4")
    
    # Save the uploaded file
    file.save(csv_path)
    
    try:
        # Create animator and generate animation
        animator = FinancialAnimator(csv_path=csv_path)
        statement_type = animator.prepare_data()
        animator.create_animated_infographic(output_path=output_path, fps=fps, duration=duration)
        
        # Return the template with the video path
        video_url = url_for('get_video', filename=os.path.basename(output_path))
        return render_template('index.html', 
                               video_path=video_url,
                               message=f"Successfully processed {statement_type}",
                               message_type="success")
    
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}", message_type="error")


@app.route('/videos/<filename>')
def get_video(filename):
    """Serve generated videos"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


def cleanup_old_files():
    """Clean up old files to prevent disk space issues"""
    # Get all files in the upload folder
    files = os.listdir(UPLOAD_FOLDER)
    
    # Get current time
    now = time.time()
    
    # Delete files older than 1 hour
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.isfile(file_path):
            # Get file's last modified time
            mtime = os.path.getmtime(file_path)
            # If file is older than 1 hour, delete it
            if now - mtime > 3600:
                try:
                    os.remove(file_path)
                except:
                    pass


if __name__ == '__main__':
    import time
    import threading
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=lambda: (time.sleep(3600), cleanup_old_files()))
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
