import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import glob
import cv2
from tqdm import tqdm

class DataPreparation:
    def __init__(self, source_dir, output_dir, img_size=128, augment=True):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.augment = augment
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def process_images(self):
        """Process and prepare images for training"""
        # Get all image files
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
        image_files = []
        
        for ext in extensions:
            pattern = os.path.join(self.source_dir, f"*.{ext}")
            image_files.extend(glob.glob(pattern))
            # Also check subdirectories
            pattern = os.path.join(self.source_dir, f"**/*.{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(image_files)} images to process")
        
        if not image_files:
            print(f"No images found in {self.source_dir}")
            return
        
        # Process each image
        for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Read image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Resize
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                
                # Normalize to [-1, 1]
                img_normalized = (img_resized.astype(np.float32) - 127.5) / 127.5
                
                # Save as numpy file
                output_path = os.path.join(self.output_dir, f"image_{i:05d}.npy")
                np.save(output_path, img_normalized)
                
                # Apply augmentation if enabled
                if self.augment:
                    # Flip horizontally
                    img_flipped = np.fliplr(img_normalized)
                    output_path = os.path.join(self.output_dir, f"image_{i:05d}_flip_h.npy")
                    np.save(output_path, img_flipped)
                    
                    # Flip vertically
                    img_flipped_v = np.flipud(img_normalized)
                    output_path = os.path.join(self.output_dir, f"image_{i:05d}_flip_v.npy")
                    np.save(output_path, img_flipped_v)
                    
                    # Rotate 90 degrees
                    img_rot90 = np.rot90(img_normalized)
                    output_path = os.path.join(self.output_dir, f"image_{i:05d}_rot90.npy")
                    np.save(output_path, img_rot90)
            
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        print(f"Processing complete. Prepared dataset saved to {self.output_dir}")
        
    def create_tf_dataset(self, batch_size=32, shuffle_buffer=1000):
        """Create a TensorFlow dataset from processed images"""
        # Get all numpy files
        npy_files = glob.glob(os.path.join(self.output_dir, "*.npy"))
        
        if not npy_files:
            print(f"No processed images found in {self.output_dir}")
            return None
        
        print(f"Creating dataset from {len(npy_files)} processed images")
        
        # Create a dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices(npy_files)
        
        # Load and parse each file
        def load_image(file_path):
            img = np.load(file_path.numpy())
            return img
        
        def load_image_tf(file_path):
            img = tf.py_function(load_image, [file_path], tf.float32)
            img.set_shape([self.img_size, self.img_size, 3])
            return img
        
        dataset = dataset.map(load_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Shuffle and batch
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def visualize_samples(self, num_samples=5):
        """Visualize sample images from the processed dataset"""
        npy_files = glob.glob(os.path.join(self.output_dir, "*.npy"))
        
        if not npy_files or len(npy_files) < num_samples:
            print(f"Not enough processed images found in {self.output_dir}")
            return
        
        # Randomly select samples
        sample_files = np.random.choice(npy_files, num_samples, replace=False)
        
        # Create figure
        plt.figure(figsize=(15, 3))
        
        for i, file_path in enumerate(sample_files):
            # Load image
            img = np.load(file_path)
            
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            
            # Display
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Sample {i+1}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "samples.png"))
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for camouflage pattern generation')
    parser.add_argument('--source', type=str, required=True, help='Source directory containing images')
    parser.add_argument('--output', type=str, default='./training_data', help='Output directory for processed data')
    parser.add_argument('--img-size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--no-augment', action='store_false', dest='augment', help='Disable data augmentation')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample processed images')
    
    args = parser.parse_args()
    
    # Create data preparation instance
    data_prep = DataPreparation(
        source_dir=args.source,
        output_dir=args.output,
        img_size=args.img_size,
        augment=args.augment
    )
    
    # Process images
    data_prep.process_images()
    
    # Visualize samples if requested
    if args.visualize:
        data_prep.visualize_samples()
