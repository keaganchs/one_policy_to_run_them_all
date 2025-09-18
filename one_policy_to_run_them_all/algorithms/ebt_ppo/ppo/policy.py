"""
EBT-based policy for multi-robot reinforcement learning.
Integrates Energy-Based Transformers with the existing RL-X framework.
"""

from typing import Sequence, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from one_policy_to_run_them_all.algorithms.ebt_ppo.ppo.ebt_layers import EBTEncoder, RMSNorm


def get_policy(config, env):
    """Factory function to create EBT policy based on environment configuration."""
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (
            EBTPolicy(
                # Standard policy parameters
                std_dev=config.algorithm.std_dev,
                softmax_temperature=config.algorithm.softmax_temperature,
                softmax_temperature_min=config.algorithm.softmax_temperature_min,
                stability_epsilon=config.algorithm.stability_epsilon,
                policy_mean_abs_clip=config.algorithm.policy_mean_abs_clip,
                policy_std_min_clip=config.algorithm.policy_std_min_clip,
                policy_std_max_clip=config.algorithm.policy_std_max_clip,
                # EBT-specific parameters
                ebt_dim=config.algorithm.ebt_dim,
                ebt_n_layers=config.algorithm.ebt_n_layers,
                ebt_n_heads=config.algorithm.ebt_n_heads,
                ebt_max_seq_len=config.algorithm.ebt_max_seq_len,
                ebt_dropout=config.algorithm.ebt_dropout,
                ebt_use_energy_attention=config.algorithm.ebt_use_energy_attention,
                ebt_energy_scale=config.algorithm.ebt_energy_scale,
                ebt_state_embedding_dim=config.algorithm.ebt_state_embedding_dim
            ), 
            get_processed_action_function()
        )
    else:
        raise NotImplementedError(f"EBT policy not implemented for action space {action_space_type} and observation space {observation_space_type}")


class StateEmbedding(nn.Module):
    """
    Embeds multi-robot state components into transformer-ready representations.
    
    This module handles the complex multi-robot state structure and creates
    embeddings that can be processed by the EBT encoder.
    """
    embedding_dim: int
    stability_epsilon: float = 1e-8

    @nn.compact
    def __call__(self, dynamic_joint_description: jnp.ndarray, 
                 dynamic_joint_state: jnp.ndarray,
                 dynamic_foot_description: jnp.ndarray, 
                 dynamic_foot_state: jnp.ndarray,
                 general_state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Embed multi-robot state components.
        
        Args:
            dynamic_joint_description: Joint descriptions (batch, n_joints, desc_dim)
            dynamic_joint_state: Joint states (batch, n_joints, state_dim)
            dynamic_foot_description: Foot descriptions (batch, n_feet, desc_dim)
            dynamic_foot_state: Foot states (batch, n_feet, state_dim)
            general_state: General state information (batch, general_dim)
            
        Returns:
            Tuple of (embedded_states, attention_mask)
        """
        batch_size = dynamic_joint_description.shape[0]
        
        # Process joint information with attention-based aggregation
        joint_desc_embed = nn.Dense(self.embedding_dim // 2)(dynamic_joint_description)
        joint_state_embed = nn.Dense(self.embedding_dim // 2)(dynamic_joint_state)
        
        # Combine joint description and state
        joint_combined = jnp.concatenate([joint_desc_embed, joint_state_embed], axis=-1)
        joint_combined = nn.LayerNorm()(joint_combined)
        joint_combined = nn.elu(joint_combined)
        
        # Process foot information
        foot_desc_embed = nn.Dense(self.embedding_dim // 2)(dynamic_foot_description)
        foot_state_embed = nn.Dense(self.embedding_dim // 2)(dynamic_foot_state)
        
        # Combine foot description and state
        foot_combined = jnp.concatenate([foot_desc_embed, foot_state_embed], axis=-1)
        foot_combined = nn.LayerNorm()(foot_combined)
        foot_combined = nn.elu(foot_combined)
        
        # Process general state
        general_embed = nn.Dense(self.embedding_dim)(general_state)
        general_embed = nn.LayerNorm()(general_embed)
        general_embed = nn.elu(general_embed)
        general_embed = jnp.expand_dims(general_embed, axis=1)  # Add sequence dimension
        
        # Combine all state components into a sequence
        # Shape: (batch, seq_len, embedding_dim) where seq_len = n_joints + n_feet + 1
        state_sequence = jnp.concatenate([
            joint_combined,
            foot_combined, 
            general_embed
        ], axis=1)
        
        # Create attention mask (all positions are valid)
        seq_len = state_sequence.shape[1]
        attention_mask = jnp.zeros((seq_len, seq_len))
        
        return state_sequence, attention_mask


class EBTPolicy(nn.Module):
    """
    Energy-Based Transformer Policy for multi-robot control.
    
    This policy uses EBT to process multi-robot state sequences and
    generate actions for each robot jointly, enabling coordination.
    """
    # Standard policy parameters (inherited from original)
    std_dev: float
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float
    
    # EBT-specific parameters
    ebt_dim: int = 256
    ebt_n_layers: int = 4
    ebt_n_heads: int = 8
    ebt_max_seq_len: int = 64
    ebt_dropout: float = 0.1
    ebt_use_energy_attention: bool = True
    ebt_energy_scale: float = 1.0
    ebt_state_embedding_dim: int = 128

    def setup(self):
        # State embedding module
        self.state_embedding = StateEmbedding(
            embedding_dim=self.ebt_state_embedding_dim,
            stability_epsilon=self.stability_epsilon
        )
        
        # Project state embeddings to EBT dimension
        self.embed_to_ebt = nn.Dense(self.ebt_dim)
        
        # EBT encoder for processing state sequences
        self.ebt_encoder = EBTEncoder(
            dim=self.ebt_dim,
            n_layers=self.ebt_n_layers,
            n_heads=self.ebt_n_heads,
            dropout=self.ebt_dropout,
            max_seq_len=self.ebt_max_seq_len,
            use_energy_attention=self.ebt_use_energy_attention,
            energy_scale=self.ebt_energy_scale
        )
        
        # Project EBT output back to action space
        self.ebt_to_action = nn.Dense(512)
        self.action_norm = nn.LayerNorm()
        
        # Action head layers (similar to original policy)
        self.action_latent_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.action_latent_2 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

    @nn.compact
    def __call__(self, dynamic_joint_description: jnp.ndarray, 
                 dynamic_joint_state: jnp.ndarray,
                 dynamic_foot_description: jnp.ndarray, 
                 dynamic_foot_state: jnp.ndarray,
                 general_state: jnp.ndarray, 
                 training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass of EBT policy.
        
        Args:
            dynamic_joint_description: Joint descriptions for all robots
            dynamic_joint_state: Joint states for all robots  
            dynamic_foot_description: Foot descriptions for all robots
            dynamic_foot_state: Foot states for all robots
            general_state: General state information
            training: Whether in training mode
            
        Returns:
            Tuple of (action_mean, action_logstd)
        """
        # Embed state components into transformer-ready format
        state_sequence, attention_mask = self.state_embedding(
            dynamic_joint_description, dynamic_joint_state,
            dynamic_foot_description, dynamic_foot_state, general_state
        )
        
        # Project to EBT dimension
        ebt_input = self.embed_to_ebt(state_sequence)
        ebt_input = nn.LayerNorm()(ebt_input)
        
        # Process through EBT encoder
        ebt_output = self.ebt_encoder(ebt_input, mask=attention_mask, training=training)
        
        # Global pooling over sequence dimension (mean pooling)
        # This aggregates information from all robots and state components
        pooled_output = jnp.mean(ebt_output, axis=1)  # (batch, ebt_dim)
        
        # Project to action space
        action_latent = self.ebt_to_action(pooled_output)
        action_latent = self.action_norm(action_latent)
        action_latent = nn.elu(action_latent)
        
        # Action processing layers
        action_latent = self.action_latent_1(action_latent)
        action_latent = nn.elu(action_latent)
        action_latent = self.action_latent_2(action_latent)
        
        # For each joint, generate action mean and std
        n_joints = dynamic_joint_description.shape[1]
        
        # Expand action latent for each joint
        action_latent_expanded = jnp.expand_dims(action_latent, axis=1)  # (batch, 1, 128)
        action_latent_expanded = jnp.broadcast_to(
            action_latent_expanded, 
            (action_latent_expanded.shape[0], n_joints, action_latent_expanded.shape[2])
        )  # (batch, n_joints, 128)
        
        # Combine with joint descriptions for joint-specific actions
        joint_desc_processed = nn.Dense(128)(dynamic_joint_description)
        joint_desc_processed = nn.LayerNorm()(joint_desc_processed)
        joint_desc_processed = nn.elu(joint_desc_processed)
        
        combined_action_input = jnp.concatenate([
            action_latent_expanded, 
            stop_gradient(nn.Dense(4)(dynamic_joint_state)),  # Stop gradient like original
            joint_desc_processed
        ], axis=-1)
        
        # Generate action mean
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_action_input)
        policy_mean = nn.LayerNorm()(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_mean = jnp.clip(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)
        
        # Generate action std
        policy_logstd = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(np.log(self.std_dev)))(joint_desc_processed)
        policy_logstd = jnp.clip(policy_logstd, np.log(self.policy_std_min_clip), np.log(self.policy_std_max_clip))
        
        return policy_mean.squeeze(-1), policy_logstd.squeeze(-1)


def get_processed_action_function():
    """Returns the action processing function (same as original)."""
    def get_clipped_and_scaled_action(action):
        return action
    return jax.jit(get_clipped_and_scaled_action)