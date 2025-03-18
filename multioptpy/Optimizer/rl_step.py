import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import copy
import os
from collections import deque

"""
ref.:
Kabir Ahuja, William H. Green, and Yi-Pei Li
Journal of Chemical Theory and Computation 2021 17 (2), 818-825
DOI: 10.1021/acs.jctc.0c00971
"""


class SelfAttention(nn.Module):
    """Self-attention module for capturing relationships between atomic coordinates."""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """Forward pass through self-attention layer."""
        # Reshape for attention: [seq_len, batch, embed_dim]
        x_reshaped = x.transpose(0, 1)
        
        # Apply self-attention
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Add & norm (residual connection)
        x_out = self.norm(x_reshaped + attn_output)
        
        # Return to original shape
        return x_out.transpose(0, 1)


class StepSizePolicy(nn.Module):
    """Policy network with self-attention for determining optimal step size scaling."""
    
    def __init__(self, state_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        super(StepSizePolicy, self).__init__()
        self.state_dim = state_dim
        
        # Explicitly define embedding dimension
        self.embed_dim = hidden_dim
        self.seq_len = 4  # Number of attention sequence elements
        
        # Feature extraction layers
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attn_proj = nn.Linear(hidden_dim, self.seq_len * self.embed_dim)
        self.attention = SelfAttention(self.embed_dim, num_heads, dropout)
        
        # Step size prediction head (mu and sigma for normal distribution)
        self.step_size_mu = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Bound to [0,1] then will be scaled
        )
        
        self.step_size_sigma = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive standard deviation
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """Forward pass to predict step size parameters and value."""
        batch_size = state.shape[0]
        
        # Extract features
        features = self.embedding(state)
        
        # Apply attention
        attn_ready = self.attn_proj(features).view(batch_size, self.seq_len, self.embed_dim)
        attn_output = self.attention(attn_ready)
        attn_output = attn_output.reshape(batch_size, -1)  # Flatten back
        
        # Predict step size parameters
        mu = self.step_size_mu(attn_output)
        sigma = self.step_size_sigma(attn_output) + 0.01  # Add minimum std for exploration
        
        # Predict state value
        value = self.value_head(attn_output)
        
        return mu, sigma, value


class PPOMemory:
    """Memory buffer for PPO algorithm."""
    
    def __init__(self, state_dim, buffer_size=1000, gamma=0.99, gae_lambda=0.95):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.float32)
        self.probs = np.zeros((buffer_size,), dtype=np.float32)
        self.vals = np.zeros((buffer_size,), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buffer_size
    
    def store(self, state, action, prob, val, reward, done):
        """Store transition in buffer."""
        idx = self.ptr % self.max_size
        
        self.states[idx] = state
        self.actions[idx] = action
        self.probs[idx] = prob
        self.vals[idx] = val
        self.rewards[idx] = reward
        self.dones[idx] = done
        
        self.ptr += 1
    
    def compute_advantages(self, last_val=0):
        """Compute Generalized Advantage Estimation."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_val)
        values = np.append(self.vals[path_slice], last_val)
        dones = np.append(self.dones[path_slice], 0)
        
        advantages = np.zeros_like(rewards[:-1])
        lastgaelam = 0
        
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * lastgaelam
        
        returns = advantages + self.vals[path_slice]
        
        self.path_start_idx = self.ptr
        return advantages, returns
    
    def get_batch(self):
        """Get all stored data as batch."""
        assert self.ptr > self.path_start_idx, "No transitions to process"
        path_slice = slice(self.path_start_idx, self.ptr)
        advantages, returns = self.compute_advantages()
        
        return (
            self.states[path_slice],
            self.actions[path_slice],
            self.probs[path_slice],
            returns,
            advantages
        )
    
    def clear(self):
        """Clear memory after policy update."""
        self.ptr, self.path_start_idx = 0, 0


class RLStepSizeOptimizer:
    """
    Reinforcement Learning optimizer for adaptive step size optimization.
    Uses PPO with self-attention to learn optimal step size scaling
    factors based on gradient and displacement history.
    """
    
    def __init__(self):
        # RL parameters
        self.history_length = 5         # Number of past steps to store in history
        self.min_step_size = 0.05       # Minimum step size scaling factor
        self.max_step_size = 2.0        # Maximum step size scaling factor
        self.default_step_size = 0.5    # Default step size when not using RL
        self.safe_step_max = 1.5        # Maximum allowed step size in safe mode
        
        # Training parameters
        self.learning_rate = 3e-4       # Learning rate for policy updates
        self.clip_ratio = 0.2           # PPO clipping parameter
        self.n_epochs = 10              # Number of policy update epochs
        self.batch_size = 64            # Batch size for policy updates
        self.gamma = 0.99               # Discount factor
        self.gae_lambda = 0.95          # GAE lambda parameter
        self.training_mode = True       # Whether to update policy during optimization
        
        # Adaptive step size control
        self.rl_weight = 0.1            # Weight of RL prediction vs default step size
        self.rl_weight_min = 0.01        # Minimum RL weight during optimization
        self.rl_weight_max = 0.5        # Maximum RL weight during optimization
        self.rl_weight_decay = 0.95     # Decay factor for RL weight on failure
        self.rl_weight_growth = 1.05    # Growth factor for RL weight on success
        
        # Performance monitoring
        self.step_success_threshold = 0.7  # Energy decrease ratio to consider step successful
        self.consecutive_failures = 0      # Count of consecutive unsuccessful steps
        self.max_failures = 3              # Max failures before reducing RL weight
        self.recovery_steps = 2            # Steps in recovery mode
        self.current_recovery = 0          # Current step in recovery
        
        # History storage
        self.geom_history = deque(maxlen=self.history_length)
        self.grad_history = deque(maxlen=self.history_length)
        self.displacement_history = deque(maxlen=self.history_length)
        self.step_history = deque(maxlen=self.history_length)
        self.energy_history = deque(maxlen=self.history_length)
        
        # Model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = None           # Will be set dynamically
        self.policy = None              # RL policy network
        self.policy_old = None          # Target network for stable updates
        self.optimizer = None           # Policy optimizer
        self.memory = None              # Experience replay buffer
        
        # Configure paths
        self.model_dir = os.path.join(os.getcwd(), 'rl_models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialization flags
        self.initialization = True
        self.iter = 0
        
        print(f"RL Step Size Optimizer initialized. Device: {self.device}")
    
    def _init_rl_components(self, state_dim):
        """Initialize RL components based on input dimensions."""
        self.state_dim = state_dim
        
        # Create policy network
        self.policy = StepSizePolicy(
            state_dim=state_dim,
            hidden_dim=128
        ).to(self.device)
        
        # Create target network
        self.policy_old = copy.deepcopy(self.policy)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Create replay buffer
        self.memory = PPOMemory(
            state_dim=state_dim,
            buffer_size=1000,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Try to load pre-trained model if available
        model_path = os.path.join(self.model_dir, "step_size_policy.pt")
        if os.path.exists(model_path):
            try:
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                self.policy_old.load_state_dict(self.policy.state_dict())
                print("Loaded pre-trained RL step size policy")
            except Exception as e:
                print(f"Could not load pre-trained model: {str(e)}")
                
        print(f"RL components initialized with state dimension: {state_dim}")
    
    def _get_state_representation(self, geom_num_list, B_g):
        """
        Construct state representation for RL policy.
        Includes current geometry, gradient, and history information.
        """
        # Current gradient and geometry are the primary components
        current_grad = B_g.flatten()
        
        # Calculate gradient norm and add as a feature
        grad_norm = np.linalg.norm(current_grad)
        norm_feature = np.array([grad_norm])
        
        # Initialize lists to store state components
        state_components = [current_grad, norm_feature]
        
        # Add gradient history
        for past_grad in list(self.grad_history):
            # Only use important information to keep state size reasonable
            if len(past_grad) > 30:  # If gradient is very large
                # Subsample or use statistics
                past_grad_norm = np.linalg.norm(past_grad)
                past_grad_stats = np.array([past_grad_norm, past_grad.mean(), past_grad.std()])
                state_components.append(past_grad_stats)
            else:
                state_components.append(past_grad)
        
        # Add displacement history
        for disp in list(self.displacement_history):
            if len(disp) > 30:  # If displacement is very large
                disp_norm = np.linalg.norm(disp)
                disp_stats = np.array([disp_norm, disp.mean(), disp.std()])
                state_components.append(disp_stats)
            else:
                state_components.append(disp)
        
        # Add step size history
        if len(self.step_history) > 0:
            step_sizes = np.array([step for step in self.step_history])
            state_components.append(step_sizes)
        else:
            state_components.append(np.array([0.5]))  # Default if no history
        
        # Add energy change history if available
        if len(self.energy_history) > 1:
            # Convert to relative energy changes
            energy_array = np.array(list(self.energy_history))
            energy_changes = np.diff(energy_array) / (np.abs(energy_array[:-1]) + 1e-10)
            state_components.append(energy_changes)
        else:
            state_components.append(np.array([0.0]))  # No energy change history
        
        # Ensure state vector has consistent size by padding or truncating
        max_state_dim = 200  # Maximum allowed state dimension
        state_array = np.concatenate([comp.flatten() for comp in state_components])
        
        if len(state_array) > max_state_dim:
            # Truncate if too large
            print(f"Warning: State dimension {len(state_array)} exceeds max {max_state_dim}, truncating")
            state_array = state_array[:max_state_dim]
        elif len(state_array) < max_state_dim:
            # Pad if too small
            padding = np.zeros(max_state_dim - len(state_array))
            state_array = np.concatenate([state_array, padding])
        
        return state_array
    
    def _predict_step_size(self, state):
        """Predict step size using the RL policy."""
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                mu, sigma, value = self.policy_old(state_tensor)
                
                # Create normal distribution
                dist = Normal(mu, sigma)
                
                # Sample action
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Convert to numpy
                action_np = action.cpu().numpy()[0, 0]
                log_prob_np = log_prob.cpu().numpy()[0]
                value_np = value.cpu().numpy()[0, 0]
                
                # Scale action from [0,1] to [min_step_size, max_step_size]
                scaled_action = self.min_step_size + action_np * (self.max_step_size - self.min_step_size)
                
                return scaled_action, log_prob_np, value_np
        except Exception as e:
            print(f"Error in step size prediction: {str(e)}")
            return self.default_step_size, 0.0, 0.0
    
    def _calculate_reward(self, energy, prev_energy, grad_norm, prev_grad_norm, step_size):
        """Calculate reward based on energy and gradient improvements."""
        # Base reward on energy improvement
        if prev_energy is not None:
            energy_change = prev_energy - energy
            energy_reward = 10.0 * energy_change / (abs(prev_energy) + 1e-10)
        else:
            energy_reward = 0.0
        
        # Add reward for gradient reduction
        if prev_grad_norm is not None:
            grad_reduction = prev_grad_norm - grad_norm
            grad_reward = 0.5 * grad_reduction / (prev_grad_norm + 1e-10)
        else:
            grad_reward = 0.0
        
        # Penalize extreme step sizes
        step_size_penalty = 0.0
        if step_size < 0.1 or step_size > 1.9:
            step_size_penalty = -0.2 * abs(step_size - 1.0)
        
        # Combine rewards
        total_reward = energy_reward + grad_reward + step_size_penalty
        
        # Strong penalty for energy increases
        if energy_change < 0 and prev_energy is not None:
            energy_increase_penalty = -5.0 * abs(energy_change) / (abs(prev_energy) + 1e-10)
            total_reward += energy_increase_penalty
        
        return total_reward
    
    def _update_policy(self):
        """Update policy using PPO algorithm."""
        if not self.training_mode or self.memory is None or self.memory.ptr <= self.memory.path_start_idx:
            return
        
        # Get batch data
        states, actions, old_log_probs, returns, advantages = self.memory.get_batch()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(self.n_epochs):
            # Process in batches
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = slice(start_idx, end_idx)
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current policy outputs
                mu, sigma, values = self.policy(batch_states)
                dist = Normal(mu, sigma)
                
                # Calculate new log probabilities
                new_log_probs = dist.log_prob(batch_actions).sum(1, keepdim=True)
                
                # Ensure consistent tensor shapes
                batch_old_log_probs = batch_old_log_probs.view(-1, 1)
                batch_returns = batch_returns.view(-1, 1)
                batch_advantages = batch_advantages.view(-1, 1)
                
                # Calculate ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss with consistent shapes
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                entropy_loss = -0.01 * entropy  # Small entropy bonus
                
                # Total loss
                total_loss = actor_loss + 0.5 * value_loss + entropy_loss
                
                # Perform optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # Rest of the method remains unchanged
        self.policy_old.load_state_dict(self.policy.state_dict())
        torch.save(self.policy.state_dict(), os.path.join(self.model_dir, "step_size_policy.pt"))
        self.memory.clear()
        
    def run(self, geom_num_list, B_g, pre_B_g, B_e, pre_B_e, original_move_vector):
        """
        Run RL-based step size optimization.
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current geometry flattened
        B_g : numpy.ndarray
            Current gradient
        pre_B_g : numpy.ndarray
            Previous gradient
        original_move_vector : numpy.ndarray
            Original optimization step
        B_e : float, optional
            Current energy
        pre_B_e : float, optional
            Previous energy
            
        Returns:
        --------
        numpy.ndarray
            Optimized move vector
        """
        print("RL Step Size Optimization")
        
        # Handle first step initialization
        if self.initialization:
            self.initialization = False
            
            # Store initial values
            if B_e is not None:
                self.energy_history.append(B_e)
            self.grad_history.append(B_g.flatten())
            
            print(f"First step, using default step size: {self.default_step_size}")
            return self.default_step_size * original_move_vector
        
        # Extract dimensions and norms
        n_coords = len(geom_num_list)
        grad_norm = np.linalg.norm(B_g)
        prev_grad_norm = np.linalg.norm(pre_B_g) if pre_B_g is not None else None
        
        # Calculate displacement
        if pre_B_g is not None and pre_B_g.flatten() is not None:
            displacement = (geom_num_list - pre_B_g).flatten()
            self.displacement_history.append(displacement)
        
        # Calculate energy delta if energies provided
        energy_decreased = False
        if B_e is not None and pre_B_e is not None:
            energy_delta = pre_B_e - B_e
            energy_decreased = energy_delta > 0
            energy_ratio = abs(energy_delta / (abs(pre_B_e) + 1e-10))
            successful_step = energy_ratio > self.step_success_threshold
        else:
            successful_step = True  # Assume success if no energies provided
        
        # Store current values in history
        if B_e is not None:
            self.energy_history.append(B_e)
        self.grad_history.append(B_g.flatten())
        self.geom_history.append(geom_num_list.flatten())
        
        # If we're in recovery mode, use conservative step size
        if self.current_recovery > 0:
            self.current_recovery -= 1
            step_size = min(0.5, self.default_step_size)
            
            print(f"In recovery mode ({self.current_recovery} steps remaining)")
            print(f"Using conservative step size: {step_size}")
            
            self.step_history.append(step_size)
            return step_size * original_move_vector
        
        # Get state representation
        state = self._get_state_representation(geom_num_list, B_g)
        
        # Initialize RL components if not already done
        if self.policy is None:
            self._init_rl_components(len(state))
        
        # Predict step size using RL policy
        rl_step_size, log_prob, value = self._predict_step_size(state)
        
        # Adjust for safety based on convergence
        if grad_norm < 0.05:  # Near convergence
            # Use more conservative step size near convergence
            safe_step_size = min(rl_step_size, self.safe_step_max)
            print(f"Near convergence (gradient norm: {grad_norm:.6f}), using safer step size: {safe_step_size:.4f}")
            rl_step_size = safe_step_size
        
        # Apply adaptive weighting between RL and default
        blended_step_size = self.rl_weight * rl_step_size + (1.0 - self.rl_weight) * self.default_step_size
        
        # Store step size in history
        self.step_history.append(blended_step_size)
        
        # Calculate reward if sufficient history exists
        if len(self.energy_history) >= 2 and B_e is not None and pre_B_e is not None:
            reward = self._calculate_reward(B_e, pre_B_e, grad_norm, prev_grad_norm, blended_step_size)
            
            # Update consecutive failures/successes tracking
            if successful_step:
                self.consecutive_failures = 0
                # Slowly increase RL weight on success
                self.rl_weight = min(self.rl_weight_max, self.rl_weight * self.rl_weight_growth)
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    # Switch to recovery mode after multiple failures
                    self.current_recovery = self.recovery_steps
                    # Reduce RL weight
                    self.rl_weight = max(self.rl_weight_min, self.rl_weight * self.rl_weight_decay)
                    print(f"Multiple failures detected, reducing RL weight to {self.rl_weight:.4f}")
                    self.consecutive_failures = 0
            
            # Store experience for learning if in training mode
            if self.training_mode and self.memory is not None:
                # Convert step size to [0,1] range for storage
                normalized_step_size = (blended_step_size - self.min_step_size) / (self.max_step_size - self.min_step_size)
                normalized_step_size = np.clip(normalized_step_size, 0, 1)
                
                # Store the experience
                done = (grad_norm < 0.01)  # Consider done when converged
                self.memory.store(
                    state=state,
                    action=normalized_step_size,
                    prob=log_prob,
                    val=value,
                    reward=reward,
                    done=done
                )
                
                # Update policy periodically
                if self.iter > 0 and self.iter % 10 == 0:
                    self._update_policy()
        
        # Generate optimized move vector
        optimized_move_vector = blended_step_size * original_move_vector
        
        # Print step information
        print(f"Original step size: 1.0")
        print(f"RL step size: {rl_step_size:.4f}")
        print(f"Blended step size: {blended_step_size:.4f} (RL weight: {self.rl_weight:.2f})")
        if B_e is not None and pre_B_e is not None:
            print(f"Energy change: {B_e - pre_B_e:.6f}")
        print(f"Gradient norm: {grad_norm:.6f}")
        
        # Safety check for numerical issues and extreme values
        if np.any(np.isnan(optimized_move_vector)) or np.any(np.isinf(optimized_move_vector)):
            print("Warning: Numerical issues in optimized step, using scaled original step")
            optimized_move_vector = 0.5 * original_move_vector
        elif np.linalg.norm(optimized_move_vector) > 5.0 * np.linalg.norm(original_move_vector):
            print("Warning: Step size too large, scaling down")
            scale_factor = 5.0 * np.linalg.norm(original_move_vector) / np.linalg.norm(optimized_move_vector)
            optimized_move_vector = scale_factor * optimized_move_vector
        
        self.iter += 1
        return optimized_move_vector