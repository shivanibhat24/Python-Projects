import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
import argparse
import datetime
import random
import json

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("No GPU found, using CPU")

class CamoGAN:
    def __init__(self, img_size=128, channels=3, latent_dim=100):
        self.img_size = img_size
        self.channels = channels
        self.img_shape = (img_size, img_size, channels)
        self.latent_dim = latent_dim
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        # Build networks
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Load pretrained model if available
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        self.checkpoint_prefix = os.path.join(self.model_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.optimizer,
            discriminator_optimizer=self.optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        
        # Try to restore the latest checkpoint
        self.checkpoint_status = self.checkpoint.restore(
            tf.train.latest_checkpoint(self.model_dir))

    def _build_generator(self):
        """Build the generator network"""
        model = tf.keras.Sequential()
        
        # Foundation layer
        model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Reshape((16, 16, 256)))
        
        # Upsampling layers
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        # Output layer
        model.add(layers.Conv2DTranspose(self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        
        return model

    def _build_discriminator(self):
        """Build the discriminator network"""
        model = tf.keras.Sequential()
        
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.img_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        
        return model
    
    def generate_images(self, batch_size=1):
        """Generate camouflage pattern images"""
        # Random noise as input
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        # Generate images
        generated_images = self.generator(noise, training=False)
        
        # Rescale images to [0, 255]
        generated_images = (generated_images * 0.5 + 0.5) * 255
        generated_images = tf.cast(generated_images, tf.uint8)
        
        return generated_images
    
    @tf.function
    def train_step(self, real_images):
        """Single training step"""
        batch_size = real_images.shape[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images
            generated_images = self.generator(noise, training=True)
            
            # Discriminator outputs
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            # Losses
            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def _generator_loss(self, fake_output):
        """Calculate generator loss"""
        return tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(tf.ones_like(fake_output), fake_output))

    def _discriminator_loss(self, real_output, fake_output):
        """Calculate discriminator loss"""
        real_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def train(self, dataset, epochs=50, save_interval=10):
        """Train the GAN model"""
        for epoch in range(epochs):
            start = time.time()
            gen_loss_list = []
            disc_loss_list = []
            
            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch)
                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)
            
            # Calculate average losses
            avg_gen_loss = sum(gen_loss_list) / len(gen_loss_list)
            avg_disc_loss = sum(disc_loss_list) / len(disc_loss_list)
            
            print(f'Epoch {epoch+1}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {time.time()-start:.2f} sec')
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                
        # Save final model
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print(f"Model saved to {self.checkpoint_prefix}")

    def apply_environmental_adjustments(self, images, environment='woodland'):
        """Apply environment-specific adjustments to generated patterns"""
        adjusted_images = []
        
        # Define color mappings for different environments
        env_colors = {
            'woodland': [(34, 83, 31), (86, 65, 27), (48, 54, 24), (59, 93, 56)],
            'desert': [(193, 154, 107), (237, 201, 175), (217, 179, 130), (200, 167, 128)],
            'urban': [(100, 100, 100), (70, 70, 70), (130, 130, 130), (50, 50, 50)],
            'snow': [(220, 220, 220), (200, 200, 200), (240, 240, 240), (180, 180, 180)],
            'naval': [(32, 56, 100), (60, 90, 120), (20, 40, 80), (80, 105, 140)]
        }
        
        # Get appropriate colors
        colors = env_colors.get(environment, env_colors['woodland'])
        
        for img in images:
            # Convert image to numpy for OpenCV processing
            img_np = img.numpy()
            
            # Create color-adjusted version
            adjusted = np.zeros_like(img_np)
            
            # Create a mask for different color regions
            for i in range(4):
                lower_threshold = 63 * i
                upper_threshold = 63 * (i + 1)
                
                if i == 3:
                    mask = (img_np[:,:,0] >= lower_threshold)
                else:
                    mask = (img_np[:,:,0] >= lower_threshold) & (img_np[:,:,0] < upper_threshold)
                
                # Apply environment-specific color
                color = colors[i]
                adjusted[mask] = color
            
            # Apply some randomness for more natural look
            noise = np.random.randint(-15, 16, adjusted.shape)
            adjusted = np.clip(adjusted + noise, 0, 255).astype(np.uint8)
            
            adjusted_images.append(adjusted)
        
        return adjusted_images

    def optimize_for_sensor(self, images, sensor_type='visual'):
        """Optimize patterns for different sensor types"""
        optimized_images = []
        
        for img in images:
            # Convert to numpy for processing
            img_np = img.numpy() if tf.is_tensor(img) else img
            
            if sensor_type == 'infrared' or sensor_type == 'ir':
                # For IR sensors, adjust thermal signature patterns
                # Add texture that breaks up heat signatures
                texture = np.random.randint(-30, 31, img_np.shape, dtype=np.int16)
                
                # Create specific thermal-breaking patterns
                h, w, _ = img_np.shape
                for i in range(0, h, 16):
                    for j in range(0, w, 16):
                        if random.random() > 0.5:
                            block_size = random.randint(8, 24)
                            if i + block_size <= h and j + block_size <= w:
                                texture[i:i+block_size, j:j+block_size] += random.randint(-50, 51)
                
                # Apply texture adjustments
                optimized = np.clip(img_np + texture, 0, 255).astype(np.uint8)
                
            elif sensor_type == 'radar':
                # For radar, create angular patterns to scatter radar waves
                optimized = img_np.copy()
                h, w, _ = img_np.shape
                
                # Add angular patterns
                for _ in range(50):
                    thickness = random.randint(1, 3)
                    angle = random.randint(0, 180)
                    length = random.randint(w//8, w//4)
                    
                    # Calculate line endpoints
                    x1, y1 = random.randint(0, w), random.randint(0, h)
                    
                    # Convert angle to radians
                    rad = np.radians(angle)
                    x2 = int(x1 + length * np.cos(rad))
                    y2 = int(y1 + length * np.sin(rad))
                    
                    # Draw line
                    cv2.line(optimized, (x1, y1), (x2, y2), 
                             (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
                             thickness)
                
            else:  # Default: visual
                # For visual sensors, enhance contrast and apply edge detection
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Create a 3-channel edge image
                edges_colored = np.zeros_like(img_np)
                
                # Apply edge overlay
                for c in range(3):
                    edges_colored[:,:,c] = img_np[:,:,c] * 0.85 + edges.astype(np.float32) * 0.15
                
                optimized = np.clip(edges_colored, 0, 255).astype(np.uint8)
            
            optimized_images.append(optimized)
        
        return optimized_images


class CamouflageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camouflage Pattern Generator")
        self.root.geometry("900x700")
        self.root.minsize(900, 700)
        
        # Set theme colors
        self.bg_color = "#2c3e50"
        self.accent_color = "#27ae60"
        self.text_color = "#ecf0f1"
        self.button_color = "#3498db"
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
        
        # Create GAN model
        self.model = CamoGAN()
        
        # Set up UI
        self._create_widgets()
        
        # Images storage
        self.current_images = []
        self.current_displayed_image = None
        self.current_image_index = 0
        
    def _create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel (controls)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Title
        title_label = ttk.Label(control_frame, text="Camouflage Generator", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)
        
        # Generation settings frame
        settings_frame = ttk.LabelFrame(control_frame, text="Generation Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Environment selection
        env_frame = ttk.Frame(settings_frame)
        env_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(env_frame, text="Environment:").pack(side=tk.LEFT, padx=5)
        
        self.environment_var = tk.StringVar(value="woodland")
        environments = ["woodland", "desert", "urban", "snow", "naval"]
        env_dropdown = ttk.Combobox(env_frame, textvariable=self.environment_var, values=environments, state="readonly", width=15)
        env_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Sensor type selection
        sensor_frame = ttk.Frame(settings_frame)
        sensor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sensor_frame, text="Sensor Type:").pack(side=tk.LEFT, padx=5)
        
        self.sensor_var = tk.StringVar(value="visual")
        sensors = ["visual", "infrared", "radar"]
        sensor_dropdown = ttk.Combobox(sensor_frame, textvariable=self.sensor_var, values=sensors, state="readonly", width=15)
        sensor_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Resolution selection
        res_frame = ttk.Frame(settings_frame)
        res_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(res_frame, text="Resolution:").pack(side=tk.LEFT, padx=5)
        
        self.resolution_var = tk.IntVar(value=128)
        resolutions = [128, 256, 512]
        res_dropdown = ttk.Combobox(res_frame, textvariable=self.resolution_var, values=resolutions, state="readonly", width=15)
        res_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Number of patterns
        num_frame = ttk.Frame(settings_frame)
        num_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(num_frame, text="Number of Patterns:").pack(side=tk.LEFT, padx=5)
        
        self.num_patterns_var = tk.IntVar(value=4)
        num_patterns = [1, 4, 9, 16]
        num_dropdown = ttk.Combobox(num_frame, textvariable=self.num_patterns_var, values=num_patterns, state="readonly", width=15)
        num_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add some space
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.generate_btn = ttk.Button(action_frame, text="Generate Patterns", command=self.generate_patterns)
        self.generate_btn.pack(fill=tk.X, pady=5)
        
        self.save_btn = ttk.Button(action_frame, text="Save Current Pattern", command=self.save_current_pattern)
        self.save_btn.pack(fill=tk.X, pady=5)
        
        # Navigation buttons (when multiple patterns are generated)
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.prev_btn = ttk.Button(nav_frame, text="Previous", command=self.show_previous)
        self.prev_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.next_btn = ttk.Button(nav_frame, text="Next", command=self.show_next)
        self.next_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # Create right panel (preview)
        preview_frame = ttk.LabelFrame(main_frame, text="Pattern Preview")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Create canvas for image preview
        self.canvas = tk.Canvas(preview_frame, bg="light gray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Click 'Generate Patterns' to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def generate_patterns(self):
        """Generate new camouflage patterns"""
        self.status_var.set("Generating patterns...")
        self.root.update()
        
        try:
            # Generate raw patterns
            num_patterns = self.num_patterns_var.get()
            generated_images = self.model.generate_images(batch_size=num_patterns)
            
            # Apply environmental adjustments
            environment = self.environment_var.get()
            adjusted_images = self.model.apply_environmental_adjustments(generated_images, environment)
            
            # Optimize for sensor type
            sensor_type = self.sensor_var.get()
            optimized_images = self.model.optimize_for_sensor(adjusted_images, sensor_type)
            
            # Store generated images
            self.current_images = optimized_images
            self.current_image_index = 0
            
            # Display first image
            if self.current_images:
                self.display_current_image()
                self.status_var.set(f"Generated {num_patterns} pattern(s). Viewing pattern 1/{num_patterns}")
            else:
                self.status_var.set("Failed to generate patterns.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate patterns: {str(e)}")
            self.status_var.set("Error during pattern generation.")
    
    def display_current_image(self):
        """Display the current image on the canvas"""
        if not self.current_images or self.current_image_index >= len(self.current_images):
            return
        
        # Get the current image
        img = self.current_images[self.current_image_index]
        
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)
        
        # Resize image to fit the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Adjust for initial zero values
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 500
        
        # Resize while maintaining aspect ratio
        img_width, img_height = img_pil.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        
        img_resized = img_pil.resize(new_size, Image.LANCZOS)
        
        # Convert to PhotoImage
        self.current_displayed_image = ImageTk.PhotoImage(img_resized)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.current_displayed_image, anchor=tk.CENTER
        )
    
    def save_current_pattern(self):
        """Save the currently displayed pattern"""
        if not self.current_images or self.current_image_index >= len(self.current_images):
            messagebox.showinfo("Info", "No pattern available to save.")
            return
        
        # Get file path from dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Camouflage Pattern"
        )
        
        if file_path:
            try:
                # Save the current image
                img = self.current_images[self.current_image_index]
                cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                # Save metadata
                metadata_path = os.path.splitext(file_path)[0] + ".json"
                metadata = {
                    "environment": self.environment_var.get(),
                    "sensor_type": self.sensor_var.get(),
                    "resolution": self.resolution_var.get(),
                    "generated_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                self.status_var.set(f"Pattern saved to {file_path}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save pattern: {str(e)}")
                self.status_var.set("Error saving pattern.")
    
    def show_previous(self):
        """Show previous pattern"""
        if not self.current_images:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.current_images)
        self.display_current_image()
        self.status_var.set(f"Viewing pattern {self.current_image_index + 1}/{len(self.current_images)}")
    
    def show_next(self):
        """Show next pattern"""
        if not self.current_images:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.current_images)
        self.display_current_image()
        self.status_var.set(f"Viewing pattern {self.current_image_index + 1}/{len(self.current_images)}")
    
    def on_resize(self, event):
        """Handle window resize event"""
        if self.current_images:
            self.display_current_image()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Camouflage Pattern Generator')
    
    # If you want to directly generate patterns without GUI
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
    parser.add_argument('--num-patterns', type=int, default=1, help='Number of patterns to generate in headless mode')
    parser.add_argument('--env', type=str, default='woodland', help='Environment type')
    parser.add_argument('--sensor', type=str, default='visual', help='Sensor type to optimize for')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for saved patterns')
    
    args = parser.parse_args()
    
    if args.headless:
        # Run in headless mode
        import time
        
        print("Running in headless mode...")
        model = CamoGAN()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate patterns
        print(f"Generating {args.num_patterns} pattern(s)...")
        generated_images = model.generate_images(batch_size=args.num_patterns)
        adjusted_images = model.apply_environmental_adjustments(generated_images, args.env)
        optimized_images = model.optimize_for_sensor(adjusted_images, args.sensor)
        
        # Save patterns
        for i, img in enumerate(optimized_images):
            file_path = os.path.join(args.output_dir, f"camo_{args.env}_{args.sensor}_{i+1}.png")
            cv2.imwrite(file_path, cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR))
            
            # Save metadata
            metadata_path = os.path.splitext(file_path)[0] + ".json"
            metadata = {
                "environment": args.env,
                "sensor_type": args.sensor,
                "generated_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Saved pattern to {file_path}")
        
        print("Done!")
    else:
        # Run GUI mode
        root = tk.Tk()
        app = CamouflageApp(root)
        
        # Bind resize event
        root.bind("<Configure>", app.on_resize)
        
        # Start the main loop
        root.mainloop()
