import os
import argparse
import json
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
import textwrap
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    def add_dialogue_to_panel(self, image, dialogue_text):
        """Add dialogue text to a panel image"""
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
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Wrap text to fit the image
        wrapped_text = textwrap.fill(dialogue_text, width=max_chars_per_line)
        
        # Calculate text position (top of image with padding)
        padding = 10
        text_width, text_height = draw.textbbox((0, 0), wrapped_text, font=font)[2:]
        
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
        
        # Draw text
        draw.text((padding, padding), wrapped_text, fill=(0, 0, 0), font=font)
        
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

    def generate_full_comic(self, prompt):
        """Generate a full comic strip from a text prompt"""
        print(f"Generating comic from prompt: '{prompt}'")
        
        # Generate the comic script
        script = self.generate_comic_script(prompt)
        
        # Save the script
        script_path = os.path.join(self.output_dir, "comic_script.json")
        with open(script_path, "w") as f:
            json.dump(script, f, indent=2)
            
        print(f"Generated script with title: '{script['title']}'")
        
        # Process each panel
        final_panels = []
        panel_paths = []
        
        for i, panel in enumerate(script["panels"]):
            print(f"Generating panel {i+1}...")
            
            # Generate the raw image
            raw_image, panel_path = self.generate_panel_image(panel["description"], i+1)
            
            # Analyze the image with BLIP
            caption = self.analyze_image_with_blip(raw_image)
            print(f"BLIP caption: {caption}")
            
            # Add dialogue
            final_image = self.add_dialogue_to_panel(raw_image, panel["dialogue"])
            
            # Save the final panel with dialogue
            final_path = os.path.join(self.output_dir, f"panel_{i+1}_with_dialogue.png")
            final_image.save(final_path)
            
            final_panels.append(final_image)
            panel_paths.append(final_path)
        
        # Create the comic strip
        comic_strip, strip_path = self.create_comic_strip(final_panels)
        
        # Create results object
        results = {
            "title": script["title"],
            "prompt": prompt,
            "script": script,
            "panel_paths": panel_paths,
            "comic_strip_path": strip_path
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate comic strips from text prompts")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate comic from")
    parser.add_argument("--output_dir", type=str, default="comic_output", help="Directory to save output files")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = ComicStripGenerator(output_dir=args.output_dir, use_gpu=not args.cpu)
    
    # Generate the comic
    results = generator.generate_full_comic(args.prompt)
    
    print("\nComic generation complete!")
    print(f"Title: {results['title']}")
    print(f"Comic strip saved to: {results['comic_strip_path']}")
    print(f"Individual panels saved to: {', '.join(results['panel_paths'])}")


if __name__ == "__main__":
    main()
