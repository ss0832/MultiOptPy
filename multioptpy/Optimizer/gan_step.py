import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os


class Generator(nn.Module):
    """
    Generator network: Generates improved step scaling factors from 
    the current molecular structure, gradient, and original step
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64]):
        super(Generator, self).__init__()
        
        # Input features: concatenation of coordinates, gradients, and original step vectors
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            if i < len(hidden_dims)-2:
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Tanh())  # Limit step scale factor to -1 to 1 range
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    Discriminator network: Determines whether a molecular structure and step vector pair 
    represents a "good" optimization step
    """
    def __init__(self, input_dim=4, hidden_dims=[64, 32]):
        super(Discriminator, self).__init__()
        
        # Input features: coordinates, gradients, step vectors, and energy change
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())  # Output as probability between 0 and 1
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    """Experience replay buffer: Stores data for GAN training"""
    
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Random sampling from buffer"""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class GANStep:
    """
    GAN-Step optimizer:
    Uses a generative adversarial network to modify optimization steps
    """
    
    def __init__(self):
        # Basic parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 3  # Feature dimensions for each coordinate, gradient, and original step
        self.batch_size = 32
        self.training_steps = 5  # Number of GAN training steps per iteration
        self.dtype = torch.float32  # Explicitly set data type to float32
        
        # GAN settings
        self.gen_hidden_dims = [64, 128, 64]
        self.dis_hidden_dims = [64, 32]
        self.gen_learning_rate = 0.0002
        self.dis_learning_rate = 0.0001
        self.beta1 = 0.5  # Beta1 parameter for Adam optimizer
        
        # Step modification parameters
        self.min_scale = 0.2    # Minimum scaling coefficient
        self.max_scale = 3.0    # Maximum scaling coefficient
        self.step_clip = 0.5    # Maximum step size
        self.mix_ratio = 0.7    # Mixture ratio with original step
        
        # Training history and buffers
        self.min_samples_for_training = 10  # Minimum samples required for GAN training
        self.good_buffer = ReplayBuffer(1000)  # Steps that decreased energy
        self.bad_buffer = ReplayBuffer(1000)   # Steps that increased energy
        
        # Learning curves and state tracking
        self.gen_losses = []
        self.dis_losses = []
        self.geom_history = []
        self.energy_history = []
        self.gradient_history = []
        self.step_history = []
        self.iter = 0
        
        # Model initialization
        self._init_models()
    
    def _init_models(self):
        """Initialize generator and discriminator networks"""
        self.generator = Generator(
            input_dim=self.feature_dim, 
            hidden_dims=self.gen_hidden_dims
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_dim=self.feature_dim+1,  # Additional feature for energy change
            hidden_dims=self.dis_hidden_dims
        ).to(self.device)
        
        # Configure optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=self.gen_learning_rate, 
            betas=(self.beta1, 0.999)
        )
        
        self.dis_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.dis_learning_rate, 
            betas=(self.beta1, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Ensure models use consistent data type
        self.generator.to(dtype=self.dtype)
        self.discriminator.to(dtype=self.dtype)
    
    def _update_history(self, geometry, energy, gradient, step=None):
        """Update optimization history"""
        self.geom_history.append(geometry.copy())
        self.energy_history.append(energy)
        self.gradient_history.append(gradient.copy())
        
        if step is not None:
            self.step_history.append(step.copy())
        elif len(self.geom_history) > 1:
            # Calculate step from previous geometry
            calc_step = geometry - self.geom_history[-2]
            self.step_history.append(calc_step)
    
    def _update_replay_buffers(self):
        """Update replay buffers"""
        if len(self.energy_history) < 2 or len(self.step_history) < 1:
            return
        
        # Calculate energy change for the latest step
        energy_change = self.energy_history[-1] - self.energy_history[-2]
        prev_geom = self.geom_history[-2]
        prev_grad = self.gradient_history[-2]
        step = self.step_history[-1]
        
        # Create features for each coordinate point
        n_coords = prev_geom.shape[0]
        for i in range(n_coords):
            features = np.hstack([
                prev_geom[i], 
                prev_grad[i], 
                step[i]
            ]).astype(np.float32)  # Explicitly use float32
            
            # Add energy change as a feature
            features_with_energy = np.append(features, energy_change).astype(np.float32)  # Explicitly use float32
            
            # Add to appropriate buffer based on energy change
            experience = (features, features_with_energy, energy_change)
            if energy_change <= 0:  # Energy decreased = good step
                self.good_buffer.add(experience)
            else:  # Energy increased = bad step
                self.bad_buffer.add(experience)
    
    def _train_gan(self):
        """Train the GAN model"""
        # Skip training if insufficient samples
        if len(self.good_buffer) < self.min_samples_for_training:
            return False
        
        # Sample from buffer
        for _ in range(self.training_steps):
            # Mini-batch sampling
            real_batch_size = min(self.batch_size // 2, len(self.good_buffer))
            fake_batch_size = min(self.batch_size // 2, len(self.bad_buffer))
            
            if real_batch_size == 0 or fake_batch_size == 0:
                continue
                
            # Good step samples (real/positive)
            good_samples = self.good_buffer.sample(real_batch_size)
            good_features = torch.tensor([s[0] for s in good_samples], 
                                         device=self.device, 
                                         dtype=self.dtype)  # Explicitly set dtype
            good_features_with_energy = torch.tensor([s[1] for s in good_samples],
                                                    device=self.device,
                                                    dtype=self.dtype)  # Explicitly set dtype
            
            # Bad step samples (fake/negative)
            bad_samples = self.bad_buffer.sample(fake_batch_size)
            bad_features = torch.tensor([s[0] for s in bad_samples], 
                                       device=self.device,
                                       dtype=self.dtype)  # Explicitly set dtype
            bad_features_with_energy = torch.tensor([s[1] for s in bad_samples],
                                                   device=self.device,
                                                   dtype=self.dtype)  # Explicitly set dtype
            
            # Create labels
            real_labels = torch.ones(real_batch_size, 1, device=self.device, dtype=self.dtype)
            fake_labels = torch.zeros(fake_batch_size, 1, device=self.device, dtype=self.dtype)
            
            # --------------------
            # Train Discriminator
            # --------------------
            self.dis_optimizer.zero_grad()
            
            # Classify good samples
            good_outputs = self.discriminator(good_features_with_energy)
            d_loss_real = self.criterion(good_outputs, real_labels)
            
            # Classify bad samples
            bad_outputs = self.discriminator(bad_features_with_energy)
            d_loss_fake = self.criterion(bad_outputs, fake_labels)
            
            # Classify generator-modified steps
            gen_scale = self.generator(bad_features)
            gen_step_features = bad_features.clone()
            
            # Apply scaling (maintain direction, adjust scale only)
            for i in range(gen_step_features.shape[0]):
                # Get original step vector (last dimension)
                orig_step = gen_step_features[i, -1].item()
                
                # Calculate scale factor (convert from -1 to 1 range to appropriate scale)
                scale = ((gen_scale[i].item() + 1) / 2) * (self.max_scale - self.min_scale) + self.min_scale
                
                # Apply scale to compute modified step
                gen_step_features[i, -1] = orig_step * scale
            
            # Add energy change
            gen_features_with_energy = torch.cat([
                gen_step_features, 
                torch.zeros(gen_step_features.shape[0], 1, device=self.device, dtype=self.dtype)  # Explicit dtype
            ], dim=1)
            
            g_outputs = self.discriminator(gen_features_with_energy)
            d_loss_gen = self.criterion(g_outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake + d_loss_gen
            d_loss.backward()
            self.dis_optimizer.step()
            
            # --------------------
            # Train Generator
            # --------------------
            self.gen_optimizer.zero_grad()
            
            # Generate modified steps
            gen_scale = self.generator(bad_features)
            gen_step_features = bad_features.clone()
            
            # Apply scaling
            for i in range(gen_step_features.shape[0]):
                orig_step = gen_step_features[i, -1].item()
                scale = ((gen_scale[i].item() + 1) / 2) * (self.max_scale - self.min_scale) + self.min_scale
                gen_step_features[i, -1] = orig_step * scale
            
            # Add energy change
            gen_features_with_energy = torch.cat([
                gen_step_features, 
                torch.zeros(gen_step_features.shape[0], 1, device=self.device, dtype=self.dtype)  # Explicit dtype
            ], dim=1)
            
            # Get discriminator evaluation
            g_outputs = self.discriminator(gen_features_with_energy)
            
            # Train generator to produce "good" steps
            g_loss = self.criterion(g_outputs, real_labels)
            g_loss.backward()
            self.gen_optimizer.step()
            
            # Record losses
            self.gen_losses.append(g_loss.item())
            self.dis_losses.append(d_loss.item())
        
        if len(self.gen_losses) > 0:
            print(f"GAN training - Gen loss: {np.mean(self.gen_losses[-self.training_steps:]):.4f}, " 
                  f"Dis loss: {np.mean(self.dis_losses[-self.training_steps:]):.4f}")
            
        return True
    
    def _generate_improved_step(self, geometry, gradient, original_step):
        """Generate improved step using the GAN"""
        # Don't modify step if original norm is near zero
        orig_norm = np.linalg.norm(original_step)
        if orig_norm < 1e-10:
            return original_step
        
        # Create features
        n_coords = geometry.shape[0]
        features = []
        
        for i in range(n_coords):
            feat = np.hstack([
                geometry[i], 
                gradient[i], 
                original_step[i]
            ]).astype(np.float32)  # Explicitly use float32
            features.append(feat)
        
        # Prepare batch for model evaluation
        features_tensor = torch.tensor(features, device=self.device, dtype=self.dtype)  # Explicitly set dtype
        
        # Generate scaling factors with the generator
        self.generator.eval()
        with torch.no_grad():
            scale_factors = self.generator(features_tensor)
            
            # Convert from -1 to 1 range to actual scale range
            scale_factors = ((scale_factors + 1) / 2) * (self.max_scale - self.min_scale) + self.min_scale
            scale_factors = scale_factors.cpu().numpy()
        
        # Generate modified step
        gan_step = original_step.copy()
        
        for i in range(n_coords):
            # Apply scale to each coordinate point
            gan_step[i] = original_step[i] * scale_factors[i, 0]
        
        # Mix steps
        mixed_step = self.mix_ratio * gan_step + (1 - self.mix_ratio) * original_step
        
        # Limit step size
        step_norm = np.linalg.norm(mixed_step)
        if step_norm > self.step_clip:
            mixed_step = mixed_step * (self.step_clip / step_norm)
        
        # Output norms before and after modification
        print(f"Step norm - Original: {orig_norm:.6f}, GAN: {np.linalg.norm(gan_step):.6f}, "
              f"Mixed: {np.linalg.norm(mixed_step):.6f}")
        
        return mixed_step
    
    def run(self, geom_num_list, energy, gradient, original_move_vector):
        """
        Run GAN-Step optimization step
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current molecular geometry
        energy : float
            Current energy value
        gradient : numpy.ndarray
            Current gradient
        original_move_vector : numpy.ndarray
            Original optimization step
            
        Returns:
        --------
        numpy.ndarray
            Modified optimization step
        """
        print("GAN-Step method")
        
        # Update history
        self._update_history(geom_num_list, energy, gradient)
        
        # Update replay buffers
        self._update_replay_buffers()
        
        # Use original step for initial iterations
        if self.iter < 3:
            print(f"Building history (step {self.iter+1}), using original step")
            self.iter += 1
            return original_move_vector
        
        # Train GAN model
        if len(self.good_buffer) >= self.min_samples_for_training:
            try:
                gan_trained = self._train_gan()
                if not gan_trained:
                    print("Failed to train GAN, using original step")
                    self.iter += 1
                    return original_move_vector
            except RuntimeError as e:
                print(f"Error during GAN training: {str(e)}")
                print("Using original step due to training error")
                self.iter += 1
                return original_move_vector
        else:
            print(f"Not enough good samples for GAN training ({len(self.good_buffer)}/{self.min_samples_for_training})")
            self.iter += 1
            return original_move_vector
        
        # Modify step using GAN
        try:
            gan_step = self._generate_improved_step(geom_num_list, gradient, original_move_vector)
        except Exception as e:
            print(f"Error generating improved step: {str(e)}")
            print("Using original step due to generation error")
            self.iter += 1
            return original_move_vector
        
        # Check for numerical issues
        if np.any(np.isnan(gan_step)) or np.any(np.isinf(gan_step)):
            print("Warning: Numerical issues in GAN step, using original step")
            self.iter += 1
            return original_move_vector
        
        self.iter += 1
        return gan_step
    
    def save_model(self, path='gan_model'):
        """Save GAN model"""
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'dis_optimizer': self.dis_optimizer.state_dict(),
            'gen_losses': self.gen_losses,
            'dis_losses': self.dis_losses,
            'iter': self.iter
        }, os.path.join(path, 'gan_step_model.pt'))
        
        print(f"Model saved to {os.path.join(path, 'gan_step_model.pt')}")
    
    def load_model(self, path='gan_model/gan_step_model.pt'):
        """Load GAN model"""
        if not os.path.exists(path):
            print(f"No model file found at {path}")
            return False
            
        try:
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
            self.gen_losses = checkpoint['gen_losses']
            self.dis_losses = checkpoint['dis_losses']
            self.iter = checkpoint['iter']
            
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False