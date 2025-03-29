import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import os
import time
import json

class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()
        # Content encoding layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # Normalization layers
        self.norm1 = nn.InstanceNorm2d(64)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(256)
        self.norm4 = nn.InstanceNorm2d(512)
        self.norm5 = nn.InstanceNorm2d(512)
    
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = F.relu(self.norm5(self.conv5(x)))
        return x

class StyleEncoder(nn.Module):
    def __init__(self, style_dim=64, num_styles=5):
        super(StyleEncoder, self).__init__()
        # Style encoding layers with style-bank for different anime styles
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Style bank for different anime studios/styles
        self.style_bank = nn.Parameter(torch.randn(num_styles, style_dim))
        
        # Style mixing and selection
        self.style_fc = nn.Linear(512, style_dim)
        self.style_attention = nn.Linear(style_dim, num_styles)
    
    def forward(self, x, style_id=None):
        # Extract general style features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Global average pooling
        x = x.mean([2, 3])
        
        # Project to style space
        style_features = self.style_fc(x)
        
        if style_id is not None:
            # Use specific style from the bank
            style = self.style_bank[style_id]
        else:
            # Calculate attention weights for style mixing
            attention = F.softmax(self.style_attention(style_features), dim=1)
            # Weighted sum of style vectors
            style = torch.matmul(attention, self.style_bank)
            
        return style

class AnimationGenerator(nn.Module):
    def __init__(self, style_dim=64):
        super(AnimationGenerator, self).__init__()
        # AdaIN-based decoder for style transfer
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        
        # Style modulators
        self.mod1 = StyleModulator(512, style_dim)
        self.mod2 = StyleModulator(256, style_dim)
        self.mod3 = StyleModulator(128, style_dim)
        self.mod4 = StyleModulator(64, style_dim)
        
        # Line detection and enhancement
        self.line_detector = LineDetector()
        
        # Color palette adjustments for each studio
        self.color_adjustment = ColorAdjustment(style_dim)
    
    def forward(self, content, style):
        # Apply style modulation at each layer
        x = self.mod1(content, style)
        x = F.relu(self.upsample1(x))
        
        x = self.mod2(x, style)
        x = F.relu(self.upsample2(x))
        
        x = self.mod3(x, style)
        x = F.relu(self.upsample3(x))
        
        x = self.mod4(x, style)
        x = self.upsample4(x)
        
        # Enhance lines and shapes for anime style
        lines = self.line_detector(content)
        
        # Apply studio-specific color adjustments
        x = self.color_adjustment(x, style)
        
        # Combine with line art
        x = x + lines
        
        return torch.tanh(x)

class StyleModulator(nn.Module):
    def __init__(self, feature_dim, style_dim):
        super(StyleModulator, self).__init__()
        # AdaIN parameters
        self.fc_scale = nn.Linear(style_dim, feature_dim)
        self.fc_bias = nn.Linear(style_dim, feature_dim)
        self.norm = nn.InstanceNorm2d(feature_dim, affine=False)
    
    def forward(self, x, style):
        # Adaptive Instance Normalization
        x = self.norm(x)
        
        # Apply style-based scaling and bias
        scale = self.fc_scale(style).unsqueeze(2).unsqueeze(3)
        bias = self.fc_bias(style).unsqueeze(2).unsqueeze(3)
        
        return x * scale + bias

class LineDetector(nn.Module):
    def __init__(self):
        super(LineDetector, self).__init__()
        # Edge detection filters
        self.conv_h = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_v = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Initialize with Sobel filters
        self.conv_h.weight.data = torch.FloatTensor([[[
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]]]).repeat(1, 512, 1, 1)
        
        self.conv_v.weight.data = torch.FloatTensor([[[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]]]).repeat(1, 512, 1, 1)
        
        # Line refinement
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Detect horizontal and vertical edges
        edges_h = self.conv_h(x)
        edges_v = self.conv_v(x)
        
        # Combine edges
        edges = torch.sqrt(edges_h.pow(2) + edges_v.pow(2) + 1e-6)
        
        # Refine edges for anime-style lines
        lines = self.refine(edges)
        
        # Upsample to match the output resolution
        lines = F.interpolate(lines, scale_factor=8, mode='bilinear')
        
        return lines

class ColorAdjustment(nn.Module):
    def __init__(self, style_dim):
        super(ColorAdjustment, self).__init__()
        # Color adjustment layers
        self.hue_shift = nn.Linear(style_dim, 1)  # Hue shift
        self.saturation = nn.Linear(style_dim, 1)  # Saturation adjustment
        self.contrast = nn.Linear(style_dim, 1)   # Contrast adjustment
        self.brightness = nn.Linear(style_dim, 1)  # Brightness adjustment
        
        # Studio-specific color palettes
        # Ghibli: soft, natural colors
        # KyoAni: vibrant, clear colors
        # Trigger: bold, high contrast
        # MAPPA: rich, detailed
        # Ufotable: dynamic, vivid
        self.studio_presets = nn.Parameter(torch.FloatTensor([
            [0.0, 1.1, 1.0, 1.05],  # Ghibli
            [0.02, 1.2, 1.1, 1.1],  # KyoAni
            [0.05, 1.4, 1.3, 0.95],  # Trigger
            [-0.03, 1.2, 1.2, 0.9],  # MAPPA
            [0.03, 1.3, 1.25, 1.15]  # Ufotable
        ]))
    
    def forward(self, x, style):
        # Generate adjustment values from style
        hue = torch.tanh(self.hue_shift(style)) * 0.1  # Small hue shifts
        sat = torch.sigmoid(self.saturation(style)) * 1.5
        con = torch.sigmoid(self.contrast(style)) * 1.5
        bri = torch.sigmoid(self.brightness(style)) * 1.5
        
        # Simple RGB adjustments (a full HSV conversion would be more accurate)
        # For demonstration purposes, using simple adjustments
        
        # Contrast adjustment
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        x = (x - mean) * con + mean
        
        # Brightness adjustment
        x = x * bri
        
        # Saturation adjustment (approximate)
        luminance = x.mean(dim=1, keepdim=True)
        x = (x - luminance) * sat + luminance
        
        # Clamp values
        x = torch.clamp(x, -1, 1)
        
        return x

class AnimeTransformationModel(nn.Module):
    def __init__(self, style_dim=64, num_styles=5):
        super(AnimeTransformationModel, self).__init__()
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder(style_dim, num_styles)
        self.generator = AnimationGenerator(style_dim)
        
        # Style names for different anime studios
        self.style_names = ["Ghibli", "KyoAni", "Trigger", "MAPPA", "Ufotable"]
        
        # Style characteristics
        self.style_characteristics = {
            "Ghibli": "Soft lighting, natural colors, detailed backgrounds",
            "KyoAni": "Fluid animation, detailed character expressions, clean lines",
            "Trigger": "Dynamic poses, bold colors, stylized animation",
            "MAPPA": "Detailed shading, rich textures, atmospheric lighting",
            "Ufotable": "Dynamic camera, vibrant effects, smooth motion"
        }
    
    def forward(self, img, style_id=None):
        # Extract content features
        content = self.content_encoder(img)
        
        # Get style features
        style = self.style_encoder(img, style_id)
        
        # Generate anime-styled output
        output = self.generator(content, style)
        
        return output
    
    def get_available_styles(self):
        return self.style_names
    
    def get_style_info(self, style_name):
        if style_name in self.style_characteristics:
            return self.style_characteristics[style_name]
        return "Unknown style"

# Image preprocessing
def preprocess_image(image_path, target_size=(256, 256)):
    if isinstance(image_path, str):
        # Load from file path
        img = Image.open(image_path).convert('RGB')
    else:
        # Assume image_path is already a PIL Image
        img = image_path.convert('RGB')
    
    # Resize while maintaining aspect ratio
    img.thumbnail(target_size, Image.LANCZOS)
    
    # Create a new white background image
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    
    # Calculate position to paste the resized image (centered)
    x_offset = (target_size[0] - img.width) // 2
    y_offset = (target_size[1] - img.height) // 2
    
    # Paste resized image onto the white background
    new_img.paste(img, (x_offset, y_offset))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(new_img)

# Example usage function
def transform_to_anime(model, image_path, style_name=None, output_path=None):
    # Load and preprocess image
    img = preprocess_image(image_path)
    img = img.unsqueeze(0)  # Add batch dimension
    
    # Convert style name to ID if provided
    style_id = None
    if style_name:
        if style_name in model.style_names:
            style_id = model.style_names.index(style_name)
    
    # Transform image
    with torch.no_grad():
        output = model(img, style_id)
    
    # Postprocess output
    output = (output * 0.5 + 0.5).clamp(0, 1)
    output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Convert to PIL image
    output_img = (output_img * 255).astype(np.uint8)
    output_img = Image.fromarray(output_img)
    
    # Save if path is provided
    if output_path:
        output_img.save(output_path)
    
    return output_img

# Animation functions
def create_animation_sequence(model, image_path, output_dir=None, frames=24, transition_styles=True, fps=12):
    """Create a sequence of frames transitioning between different anime styles"""
    frames_out = []
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup for style transitions
    num_styles = len(model.style_names)
    
    # Prepare the input image once
    if isinstance(image_path, str):
        input_img = Image.open(image_path).convert('RGB')
    else:
        input_img = image_path
    
    input_tensor = preprocess_image(input_img).unsqueeze(0)
    
    # Extract content features once
    with torch.no_grad():
        content = model.content_encoder(input_tensor)
    
    for i in range(frames):
        print(f"Processing frame {i+1}/{frames}")
        
        if transition_styles:
            # Interpolate between two styles
            style1 = int((i / frames) * num_styles) % num_styles
            style2 = (style1 + 1) % num_styles
            alpha = (i / frames * num_styles) % 1.0
            
            # Get style vectors
            with torch.no_grad():
                style_vec1 = model.style_encoder(input_tensor, style1)
                style_vec2 = model.style_encoder(input_tensor, style2)
                
                # Interpolate style vectors
                interpolated_style = style_vec1 * (1 - alpha) + style_vec2 * alpha
                
                # Generate frame with interpolated style
                output = model.generator(content, interpolated_style)
                
                # Postprocess output
                output = (output * 0.5 + 0.5).clamp(0, 1)
                output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_img = (output_img * 255).astype(np.uint8)
                
                # Convert to PIL image
                output_img = Image.fromarray(output_img)
        else:
            # Use single style
            style_id = i % num_styles
            with torch.no_grad():
                style_vec = model.style_encoder(input_tensor, style_id)
                output = model.generator(content, style_vec)
                
                # Postprocess output
                output = (output * 0.5 + 0.5).clamp(0, 1)
                output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_img = (output_img * 255).astype(np.uint8)
                
                # Convert to PIL image
                output_img = Image.fromarray(output_img)
        
        # Save frame if directory is provided
        if output_dir:
            output_img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
        
        frames_out.append(output_img)
    
    # Create animation if output directory is provided
    if output_dir:
        try:
            # Use the first frame to determine the final animation size
            output_size = frames_out[0].size
            
            # Create an animated GIF
            output_path = os.path.join(output_dir, "animation.gif")
            frames_out[0].save(
                output_path,
                save_all=True,
                append_images=frames_out[1:],
                optimize=False,
                duration=int(1000/fps),
                loop=0
            )
            
            # Try to create a video if OpenCV is available
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(output_dir, "animation.mp4")
                video = cv2.VideoWriter(video_path, fourcc, fps, output_size)
                
                for frame in frames_out:
                    # Convert PIL image to OpenCV format
                    cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    video.write(cv_frame)
                
                video.release()
                print(f"Created video at {video_path}")
            except Exception as e:
                print(f"Error creating video: {e}")
        
        except Exception as e:
            print(f"Error creating animation: {e}")
    
    return frames_out

# Web service integration
class AnimeStyleAPI:
    def __init__(self, model_path=None):
        # Initialize the model
        self.model = AnimeTransformationModel()
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Available styles
        self.styles = self.model.get_available_styles()
        
        # Processing queue
        self.jobs = {}
    
    def process_image(self, image_data, style_name=None, animated=False, frames=24):
        """Process an incoming image request"""
        # Generate a unique job ID
        job_id = str(int(time.time() * 1000))
        
        # Store job information
        self.jobs[job_id] = {
            "status": "processing",
            "created": time.time(),
            "options": {
                "style": style_name,
                "animated": animated,
                "frames": frames
            },
            "results": None
        }
        
        try:
            # Create a temporary directory for this job
            job_dir = f"temp/job_{job_id}"
            os.makedirs(job_dir, exist_ok=True)
            
            # Save input image
            input_path = os.path.join(job_dir, "input.jpg")
            with open(input_path, "wb") as f:
                f.write(image_data)
            
            # Process based on options
            if animated:
                # Create animation sequence
                output_dir = os.path.join(job_dir, "frames")
                os.makedirs(output_dir, exist_ok=True)
                
                frames = create_animation_sequence(
                    self.model,
                    input_path,
                    output_dir=output_dir,
                    frames=frames,
                    transition_styles=(style_name is None)
                )
                
                # Update job with results
                self.jobs[job_id]["results"] = {
                    "type": "animation",
                    "gif_path": os.path.join(output_dir, "animation.gif"),
                    "mp4_path": os.path.join(output_dir, "animation.mp4"),
                    "frame_count": len(frames)
                }
            else:
                # Process single image
                output_path = os.path.join(job_dir, "output.jpg")
                transform_to_anime(self.model, input_path, style_name, output_path)
                
                # Update job with results
                self.jobs[job_id]["results"] = {
                    "type": "image",
                    "path": output_path
                }
            
            # Mark job as complete
            self.jobs[job_id]["status"] = "complete"
            
            return job_id
            
        except Exception as e:
            # Handle errors
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            print(f"Error processing job {job_id}: {e}")
            return job_id
    
    def get_job_status(self, job_id):
        """Check the status of a processing job"""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        
        return self.jobs[job_id]
    
    def get_available_styles(self):
        """Return the list of available animation styles"""
        return {
            "styles": self.styles,
            "descriptions": {style: self.model.get_style_info(style) for style in self.styles}
        }

# Example API usage
def api_example():
    # Initialize the API
    api = AnimeStyleAPI()
    
    # Get available styles
    styles = api.get_available_styles()
    print(f"Available styles: {styles}")
    
    # Process an image
    with open("sample.jpg", "rb") as f:
        image_data = f.read()
    
    # Submit a job for processing
    job_id = api.process_image(image_data, style_name="Ghibli", animated=True, frames=12)
    print(f"Submitted job: {job_id}")
    
    # Check job status
    while True:
        status = api.get_job_status(job_id)
        print(f"Job status: {status['status']}")
        
        if status['status'] in ['complete', 'failed']:
            break
        
        time.sleep(1)
    
    # Get results
    if status['status'] == 'complete':
        results = status['results']
        print(f"Job completed successfully: {results}")
    else:
        print(f"Job failed: {status.get('error', 'Unknown error')}")

# Cherry blossom animation helper for UI
def create_cherry_blossom_animation(canvas_width, canvas_height, num_blossoms=50):
    """Generate parameters for cherry blossom animation"""
    blossoms = []
    
    for i in range(num_blossoms):
        blossom = {
            "x": np.random.rand() * canvas_width,
            "y": np.random.rand() * canvas_height,
            "size": np.random.rand() * 10 + 8,
            "rotation": np.random.rand() * 360,
            "fall_duration": np.random.rand() * 10 + 10,
            "sway_duration": np.random.rand() * 6 + 4,
            "delay": np.random.rand() * 10
        }
        blossoms.append(blossom)
    
    return blossoms

# Interactive visualization using JavaScript (for web integration)
def generate_visualization_data(model, image_path, num_styles=5):
    """Generate data for interactive visualization"""
    # Process image with different styles
    results = []
    
    for i, style in enumerate(model.style_names[:num_styles]):
        # Process image with this style
        output = transform_to_anime(model, image_path, style)
        
        # Convert to base64 for web display
        import io
        import base64
        
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Add to results
        results.append({
            "style": style,
            "description": model.get_style_info(style),
            "image": f"data:image/png;base64,{img_str}"
        })
    
    return results

# Save and load model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path):
    model = AnimeTransformationModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model
