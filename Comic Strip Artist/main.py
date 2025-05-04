import os
import argparse
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import time
from pathlib import Path


class ComicStripGenerator:
    def __init__(self, output_dir="comic_output", use_gpu=True):
        """Initialize the Comic Strip Generator with required models"""
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initializing models on {self.device}...")
        
        # Initialize BLIP for image captioning
        print("Loading BLIP for image understanding...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        
        # Initialize Stable Diffusion pipeline
        print("Loading Stable Diffusion pipeline...")
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Use the DPM-Solver++ scheduler for faster inference
        self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipeline.scheduler.config
        )
        
        # Initialize Mistral model for text understanding and generation
        print("Loading Mistral for text generation...")
        self.mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.mistral_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", 
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        print("All models loaded successfully!")

    def generate_comic_script(self, prompt):
        """Generate a comic script from a prompt using Mistral"""
        system_prompt = """You are an expert comic writer. Create a 3-panel comic script based on the user's prompt.
        For each panel, provide:
        1. A detailed visual description for image generation
        2. Any dialogue or caption text
        
        Format your response as a JSON object with this structure:
        {
            "title": "Comic Title",
            "panels": [
                {
                    "description": "Detailed visual description for image generation",
                    "dialogue": "Any dialogue or caption text"
                },
                ...
            ]
        }
        
        Keep descriptions concise but detailed enough for good image generation.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Format messages for Mistral
        formatted_prompt = self.mistral_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.mistral_tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        outputs = self.mistral_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = self.mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the JSON part from the response
        try:
            # Find the JSON object in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON found in the response")
                
            json_str = response[start_idx:end_idx+1]
            script = json.loads(json_str)
            return script
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Mistral response: {e}")
            print("Generating fallback script...")
            # Fallback script with basic structure
            return {
                "title": "Generated Comic",
                "panels": [
                    {"description": f"{prompt} - panel 1", "dialogue": ""},
                    {"description": f"{prompt} - panel 2", "dialogue": ""},
                    {"description": f"{prompt} - panel 3", "dialogue": ""}
                ]
            }

    def generate_panel_image(self, description, panel_number, negative_prompt="blurry, deformed, ugly, bad anatomy"):
        """Generate an image for a comic panel using Stable Diffusion"""
        prompt = f"comic panel showing {description}, comic book style, clear lines, vibrant colors"
        
        # Generate the image
        image = self.sd_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=512,
        ).images[0]
        
        # Save the raw panel
        panel_path = os.path.join(self.output_dir, f"panel_{panel_number}.png")
        image.save(panel_path)
        
        return image, panel_path

    def custom_text_wrap(self, text, max_chars_per_line):
        """Custom text wrapping function that doesn't rely on textwrap"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word would exceed the line length
            if current_length + len(word) + (1 if current_length > 0 else 0) > max_chars_per_line:
                # Save the current line and start a new one
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                # Add the word to the current line
                if current_length > 0:
                    current_length += 1  # For the space
                current_line.append(word)
                current_length += len(word)
        
        # Add the last line if it's not empty
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)

    def add_dialogue_to_panel(self, image, dialogue_text):
        """Add dialogue text to a panel image using custom text wrapping"""
        if not dialogue_text.strip():
            return image
            
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Create a copy to draw on
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Set up text parameters
        width, height = img_with_text.size
        max_chars_per_line = 30
        font_size = 20
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Try another common font
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                # Fallback to default font
                font = ImageFont.load_default()
        
        # Wrap text using our custom wrapper
        wrapped_text = self.custom_text_wrap(dialogue_text, max_chars_per_line)
        
        # Calculate text position (top of image with padding)
        padding = 10
        
        # Measure text dimensions
        # Get the dimensions of the wrapped text
        lines = wrapped_text.split('\n')
        max_line_width = 0
        total_height = 0
        
        for line in lines:
            # For PIL >= 9.2.0
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
            except AttributeError:
                # For older PIL versions
                line_width, line_height = draw.textsize(line, font=font)
            
            max_line_width = max(max_line_width, line_width)
            total_height += line_height
        
        text_width = max_line_width
        text_height = total_height
        
        # Create semi-transparent text bubble
        bubble_padding = 10
        bubble_coords = [
            padding - bubble_padding, 
            padding - bubble_padding,
            padding + text_width + bubble_padding,
            padding + text_height + bubble_padding
        ]
        
        # Draw white bubble with black outline
        draw.rectangle(bubble_coords, fill=(255, 255, 255, 220), outline=(0, 0, 0))
        
        # Draw text line by line
        y_offset = padding
        for line in lines:
            draw.text((padding, y_offset), line, fill=(0, 0, 0), font=font)
            # Move to next line
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_height = bbox[3] - bbox[1]
            except AttributeError:
                # For older PIL versions
                _, line_height = draw.textsize(line, font=font)
            
            y_offset += line_height
        
        return img_with_text

    def analyze_image_with_blip(self, image):
        """Analyze an image using BLIP to get a caption"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        outputs = self.blip_model.generate(
            **inputs,
            max_length=30,
            num_beams=5,
            temperature=1.0
        )
        
        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def create_comic_strip(self, panel_images):
        """Combine multiple panel images into a comic strip"""
        # Calculate dimensions for the comic strip
        panel_width, panel_height = panel_images[0].size
        strip_width = panel_width * len(panel_images)
        strip_height = panel_height
        
        # Create a new image with white background
        comic_strip = Image.new('RGB', (strip_width, strip_height), color='white')
        
        # Paste each panel into the comic strip
        for i, panel in enumerate(panel_images):
            comic_strip.paste(panel, (i * panel_width, 0))
        
        # Save the comic strip
        strip_path = os.path.join(self.output_dir, "comic_strip.png")
        comic_strip.save(strip_path)
        
        return comic_strip, strip_path

    def generate_full_comic(self, prompt, progress=None):
        """Generate a full comic strip from a text prompt with optional progress updates"""
        print(f"Generating comic from prompt: '{prompt}'")
        
        if progress:
            progress(0.05, "Generating comic script...")
        
        # Generate the comic script
        script = self.generate_comic_script(prompt)
        
        # Save the script
        script_path = os.path.join(self.output_dir, "comic_script.json")
        with open(script_path, "w") as f:
            json.dump(script, f, indent=2)
            
        print(f"Generated script with title: '{script['title']}'")
        
        if progress:
            progress(0.2, f"Script generated: '{script['title']}'")
        
        # Process each panel
        final_panels = []
        panel_paths = []
        
        for i, panel in enumerate(script["panels"]):
            panel_progress = 0.2 + (i * 0.2)  # Distribute 60% of progress across panels
            print(f"Generating panel {i+1}...")
            
            if progress:
                progress(panel_progress, f"Generating panel {i+1} image...")
            
            # Generate the raw image
            raw_image, panel_path = self.generate_panel_image(panel["description"], i+1)
            
            if progress:
                progress(panel_progress + 0.05, f"Analyzing panel {i+1}...")
            
            # Analyze the image with BLIP
            caption = self.analyze_image_with_blip(raw_image)
            print(f"BLIP caption: {caption}")
            
            if progress:
                progress(panel_progress + 0.1, f"Adding dialogue to panel {i+1}...")
            
            # Add dialogue
            final_image = self.add_dialogue_to_panel(raw_image, panel["dialogue"])
            
            # Save the final panel with dialogue
            final_path = os.path.join(self.output_dir, f"panel_{i+1}_with_dialogue.png")
            final_image.save(final_path)
            
            final_panels.append(final_image)
            panel_paths.append(final_path)
        
        if progress:
            progress(0.9, "Creating final comic strip...")
        
        # Create the comic strip
        comic_strip, strip_path = self.create_comic_strip(final_panels)
        
        # Create results object
        results = {
            "title": script["title"],
            "prompt": prompt,
            "script": script,
            "panel_paths": panel_paths,
            "comic_strip_path": strip_path,
            "comic_strip": comic_strip,
            "panels": final_panels
        }
        
        if progress:
            progress(1.0, "Comic generation complete!")
        
        return results


# Create a Gradio UI
def create_ui(generator):
    """Create a Gradio UI for the comic generator"""
    
    # Create a session state to store the current output dir
    class SessionState:
        def __init__(self):
            self.output_dir = f"comic_output_{int(time.time())}"
            self.generator = generator
            self.generator.output_dir = self.output_dir
            
    session = SessionState()
    
    def generate_comic(prompt, progress=gr.Progress()):
        """Generate a comic with progress updates"""
        # Create a new output directory for each generation
        session.output_dir = f"comic_output_{int(time.time())}"
        os.makedirs(session.output_dir, exist_ok=True)
        session.generator.output_dir = session.output_dir
        
        # Generate the comic
        results = session.generator.generate_full_comic(prompt, progress)
        
        # Return the results for display
        return (
            results["title"],
            results["comic_strip"],
            results["panels"][0] if len(results["panels"]) > 0 else None,
            results["panels"][1] if len(results["panels"]) > 1 else None,
            results["panels"][2] if len(results["panels"]) > 2 else None,
            Path(results["comic_strip_path"]).absolute().as_uri(),
            json.dumps(results["script"], indent=2)
        )
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as app:
        gr.Markdown(
            """
            # 🎭 AI Comic Strip Generator
            
            Create beautiful comic strips from your ideas using AI. Just type a prompt and watch your story come to life!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Your Comic Idea",
                    placeholder="Enter a prompt for your comic strip (e.g., 'A cat and dog debate philosophy')",
                    lines=3
                )
                generate_button = gr.Button("Generate Comic Strip", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### Tips for Great Comics:
                    - Be specific about characters and settings
                    - Include emotions or conflicts
                    - Think in terms of a beginning, middle, and end
                    """
                )
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Comic Strip"):
                title_output = gr.Textbox(label="Comic Title")
                comic_output = gr.Image(label="Generated Comic Strip", type="pil")
                download_link = gr.HTML(label="Download Link")
                
            with gr.TabItem("Individual Panels"):
                with gr.Row():
                    panel1_output = gr.Image(label="Panel 1", type="pil")
                    panel2_output = gr.Image(label="Panel 2", type="pil")
                    panel3_output = gr.Image(label="Panel 3", type="pil")
            
            with gr.TabItem("Script Details"):
                script_output = gr.JSON(label="Comic Script")
        
        # Set up the generation flow
        generate_button.click(
            fn=generate_comic,
            inputs=[prompt_input],
            outputs=[title_output, comic_output, panel1_output, panel2_output, panel3_output, download_link, script_output]
        )
        
        gr.Markdown(
            """
            ### About
            This comic generator uses advanced AI models to create custom comic strips:
            - Mistral 7B for scriptwriting
            - Stable Diffusion for image generation
            - BLIP for image understanding
            
            Created with ❤️ by Shivani Bhat
            """
        )
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Generate comic strips from text prompts")
    parser.add_argument("--prompt", type=str, help="Text prompt to generate comic from")
    parser.add_argument("--output_dir", type=str, default="comic_output", help="Directory to save output files")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--ui", action="store_true", help="Launch web UI", default=False)
    parser.add_argument("--port", type=int, default=7860, help="Port for the web UI")
    parser.add_argument("--share", action="store_true", help="Create a public link for the web UI")
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = ComicStripGenerator(output_dir=args.output_dir, use_gpu=not args.cpu)
    
    # Check if we should launch the UI
    if args.ui or not args.prompt:
        print("Launching web UI...")
        app = create_ui(generator)
        app.launch(server_port=args.port, share=args.share)
    else:
        # Generate the comic via command line
        results = generator.generate_full_comic(args.prompt)
        
        print("\nComic generation complete!")
        print(f"Title: {results['title']}")
        print(f"Comic strip saved to: {results['comic_strip_path']}")
        print(f"Individual panels saved to: {', '.join(results['panel_paths'])}")


if __name__ == "__main__":
    main()
