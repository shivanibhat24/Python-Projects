import os
import time
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
import datetime
import glob
import json

class TrainCamoGAN:
    def __init__(self, data_dir, log_dir='./logs', model_dir='./models', 
                 img_size=128, channels=3, latent_dim=100, batch_size=32):
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.img_size = img_size
        self.channels = channels
        self.img_shape = (img_size, img_size, channels)
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        # Build networks
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Create checkpoint manager
        self.checkpoint_prefix = os.path.join(model_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.optimizer,
            discriminator_optimizer=self.optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        
        # Setup TensorBoard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(log_dir, current_time)
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)
    
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
    
    def _create_dataset(self):
        """Create a TensorFlow dataset from processed image files"""
        # Get all numpy files
        npy_files = glob.glob(os.path.join(self.data_dir, "*.npy"))
        
        if not npy_files:
            raise ValueError(f"No processed images found in {self.data_dir}")
        
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
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @tf.function
    def _train_step(self, images):
        """Single training step"""
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images
            generated_images = self.generator(noise, training=True)
            
            # Discriminator outputs
            real_output = self.discriminator(images, training=True)
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
    
    def _generate_and_save_images(self, epoch, test_input):
        """Generate and save images during training"""
        predictions = self.generator(test_input, training=False)
        
        # Scale images from [-1, 1] to [0, 1]
        predictions = (predictions + 1) / 2.0
        
        fig = plt.figure(figsize=(4, 4))
        
        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i+1)
            plt.imshow(predictions[i])
            plt.axis('off')
        
        # Save the figure
        save_path = os.path.join(self.log_dir, f'image_at_epoch_{epoch:04d}.png')
        plt.savefig(save_path)
        plt.close()
        
        # Log image to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.image("Generated Images", predictions, step=epoch, max_outputs=4)
    
    def train(self, epochs=50, save_interval=10):
        """Train the GAN model"""
        # Create dataset
        dataset = self._create_dataset()
        
        # Create seed for image generation
        seed = tf.random.normal([4, self.latent_dim])
        
        # Start training
        print(f"Starting training for {epochs} epochs...")
        
        try:
            for epoch in range(epochs):
                start_time = time.time()
                
                gen_loss_list = []
                disc_loss_list = []
                
                # Train over dataset
                for image_batch in dataset:
                    gen_loss, disc_loss = self._train_step(image_batch)
                    gen_loss_list.append(gen_loss)
                    disc_loss_list.append(disc_loss)
                
                # Calculate average losses
                avg_gen_loss = tf.reduce_mean(gen_loss_list)
                avg_disc_loss = tf.reduce_mean(disc_loss_list)
                
                # Log to TensorBoard
                with self.summary_writer.as_default():
                    tf.summary.scalar('Generator Loss', avg_gen_loss, step=epoch)
                    tf.summary.scalar('Discriminator Loss', avg_disc_loss, step=epoch)
                
                # Print status
                time_taken = time.time() - start_time
                print(f'Epoch {epoch+1}/{epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {time_taken:.2f} sec')
                
                # Generate and save images periodically
                if (epoch + 1) % save_interval == 0 or epoch == 0:
                    self._generate_and_save_images(epoch + 1, seed)
                    
                    # Save checkpoint
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                    print(f"Checkpoint saved at epoch {epoch+1}")
            
            # Final save
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            
            # Save model architecture and hyperparameters
            model_info = {
                'img_size': self.img_size,
                'channels': self.channels,
                'latent_dim': self.latent_dim,
                'training_epochs': epochs,
                'date_trained': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(self.model_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=4)
            
            print(f"Training complete. Model saved to {self.model_dir}")
            
        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print("Checkpoint saved.")
    
    def generate_sample_images(self, num_images=16, output_dir='./samples'):
        """Generate sample images using the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate images
        noise = tf.random.normal([num_images, self.latent_dim])
        predictions = self.generator(noise, training=False)
        
        # Scale images from [-1, 1] to [0, 255]
        predictions = ((predictions + 1) / 2.0) * 255
        predictions = tf.cast(predictions, tf.uint8)
        
        # Save individual images
        for i in range(num_images):
            img = predictions[i].numpy()
            save_path = os.path.join(output_dir, f'sample_{i+1}.png')
            plt.imsave(save_path, img)
        
        # Create a grid visualization
        rows = int(np.sqrt(num_images))
        cols = int(np.ceil(num_images / rows))
        
        fig = plt.figure(figsize=(cols * 2, rows * 2))
        
        for i in range(num_images):
            plt.subplot(rows, cols, i+1)
            plt.imshow(predictions[i])
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_grid.png'))
        plt.close()
        
        print(f"Generated {num_images} sample images in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Camouflage GAN model')
    parser.add_argument('--data', type=str, required=True, help='Directory containing processed training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension size')
    parser.add_argument('--img-size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--save-interval', type=int, default=10, help='Epochs between model saves')
    parser.add_argument('--generate-samples', action='store_true', help='Generate sample images after training')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of sample images to generate')
    
    args = parser.parse_args()
    
    # Create and train the model
    trainer = TrainCamoGAN(
        data_dir=args.data,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size
    )
    
    # Train the model
    trainer.train(epochs=args.epochs, save_interval=args.save_interval)
    
    # Generate sample images if requested
    if args.generate_samples:
        trainer.generate_sample_images(num_images=args.num_samples)
