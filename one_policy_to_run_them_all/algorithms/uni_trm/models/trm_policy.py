from typing import Sequence, Optional
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, normal

class SwiGLU(nn.Module):
    hidden_size: int
    expansion: float = 4.0

    @nn.compact
    def __call__(self, x):
        features = int(self.hidden_size * self.expansion)
        x1 = nn.Dense(features)(x)
        x2 = nn.Dense(features)(x)
        return nn.Dense(self.hidden_size)(nn.silu(x1) * x2)

class TRMBlock(nn.Module):
    hidden_size: int
    expansion: float
    num_heads: int
    mlp_t: bool
    seq_len: int # Max seq len for mlp_t

    @nn.compact
    def __call__(self, x, mask=None):
        # x: (..., Seq, Hidden)
        # mask: (..., Seq) or None. 1 for valid, 0 for padding.
        
        residual = x
        
        # Time mixing
        if self.mlp_t:
            # MLP over sequence dimension
            # x: (..., Seq, Hidden) -> (..., Hidden, Seq)
            y = jnp.swapaxes(x, -1, -2)
            # We need to handle padding if seq_len > actual_len?
            # The MLP weights are fixed size (seq_len).
            # We assume x is already padded to seq_len.
            y = SwiGLU(hidden_size=self.seq_len, expansion=self.expansion)(y)
            y = jnp.swapaxes(y, -1, -2)
            # Apply mask to zero out padding contributions?
            # If we pad with zeros, and MLP has bias, it might generate non-zeros.
            # But usually we just ignore the output at padding positions.
            if mask is not None:
                 y = y * mask[..., None]
            x = nn.RMSNorm()(residual + y)
        else:
            # Attention
            # Flax SelfAttention expects (..., Seq, Hidden)
            # mask needs to be (..., 1, Seq, Seq) or similar for attention bias.
            # nn.make_attention_mask can help.
            attn_mask = None
            if mask is not None:
                # Create (..., 1, 1, Seq) mask for broadcasting?
                # Flax expects mask shape broadcastable to (..., Heads, Seq, Seq).
                # If we want to mask out padding tokens from being attended TO:
                # mask is (..., Seq).
                # We want (..., 1, 1, Seq).
                attn_mask = nn.make_attention_mask(mask > 0, mask > 0, dtype=jnp.float32)
            
            y = nn.SelfAttention(num_heads=self.num_heads)(x, mask=attn_mask)
            x = nn.RMSNorm()(residual + y)
        
        # Channel mixing
        residual = x
        y = SwiGLU(hidden_size=self.hidden_size, expansion=self.expansion)(x)
        if mask is not None:
             y = y * mask[..., None]
        x = nn.RMSNorm()(residual + y)
        
        return x

class TRM(nn.Module):
    hidden_size: int
    expansion: float
    num_heads: int
    mlp_t: bool
    seq_len: int
    H_cycles: int
    L_cycles: int
    
    @nn.compact
    def __call__(self, x, mask=None):
        # x: (..., Seq, Hidden)
        
        # Init carry
        # We use learned initialization for z_H and z_L
        # They should be broadcasted to (..., Seq, Hidden)
        
        z_H_init = self.param('z_H_init', normal(stddev=1.0/np.sqrt(self.hidden_size)), (1, self.hidden_size))
        z_L_init = self.param('z_L_init', normal(stddev=1.0/np.sqrt(self.hidden_size)), (1, self.hidden_size))
        
        input_shape = x.shape
        seq_len = input_shape[-2]
        batch_shape = input_shape[:-2]
        
        z_H = jnp.broadcast_to(z_H_init, (*batch_shape, seq_len, self.hidden_size))
        z_L = jnp.broadcast_to(z_L_init, (*batch_shape, seq_len, self.hidden_size))
        
        # Block
        block = TRMBlock(
            hidden_size=self.hidden_size,
            expansion=self.expansion,
            num_heads=self.num_heads,
            mlp_t=self.mlp_t,
            seq_len=self.seq_len
        )
        
        # Recursion
        # We can unroll loops since cycles are small constants
        
        # H_cycles-1 without grad (stop_gradient on z_H feedback?)
        # The PyTorch code uses torch.no_grad() for the first H_cycles-1 loops.
        # In JAX, we can use stop_gradient.
        
        for _ in range(self.H_cycles - 1):
            z_H_in = jax.lax.stop_gradient(z_H)
            for _ in range(self.L_cycles):
                # z_L = block(z_L + (z_H_in + x))
                z_L = block(z_L + z_H_in + x, mask=mask)
            
            # z_H = block(z_H + z_L)
            z_H = block(z_H + z_L, mask=mask)
            
        # Last cycle with grad
        for _ in range(self.L_cycles):
             z_L = block(z_L + z_H + x, mask=mask)
        z_H = block(z_H + z_L, mask=mask)
        
        return z_H

class TRMPolicy(nn.Module):
    std_dev: float
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float
    
    # TRM Config
    hidden_size: int = 256
    expansion: float = 2.0
    num_heads: int = 4
    mlp_t: bool = False
    H_cycles: int = 2
    L_cycles: int = 2
    max_seq_len: int = 64 # Max expected sequence length (Joints + Feet + General)

    @nn.compact
    def __call__(self, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_state):
        # Embeddings
        # Joint: (..., Nr_Joints, Desc_Dim) + (..., Nr_Joints, State_Dim)
        # We project both to hidden_size and sum? Or concat and project?
        # Policy.py uses Dense(64) on description, Dense(4) on state, then attention-like masking.
        # Let's simplify: Project everything to hidden_size.
        
        # Joint embedding
        # dynamic_joint_description: (..., Nr_Joints, Desc_Dim)
        # dynamic_joint_state: (..., Nr_Joints, State_Dim)
        joint_emb = nn.Dense(self.hidden_size)(jnp.concatenate([dynamic_joint_description, dynamic_joint_state], axis=-1))
        
        # Foot embedding
        foot_emb = nn.Dense(self.hidden_size)(jnp.concatenate([dynamic_foot_description, dynamic_foot_state], axis=-1))
        
        # General embedding
        # general_state: (..., State_Dim) -> (..., 1, Hidden)
        general_emb = nn.Dense(self.hidden_size)(general_state)[..., None, :]
        
        # Concatenate to sequence
        # x: (..., Seq, Hidden)
        x = jnp.concatenate([joint_emb, foot_emb, general_emb], axis=-2)
        
        input_shape = x.shape
        seq_len = input_shape[-2]
        batch_shape = input_shape[:-2]
        
        # Padding if using mlp_t
        mask = None
        if self.mlp_t:
            if seq_len > self.max_seq_len:
                # Truncate? Should not happen if max_seq_len is set correctly.
                x = x[..., :self.max_seq_len, :]
            elif seq_len < self.max_seq_len:
                pad_len = self.max_seq_len - seq_len
                # Pad last dimension (hidden) with 0, and second to last (seq) with pad_len
                # jnp.pad expects ((before, after), ...) for each dimension.
                # We need to construct padding config for all dimensions.
                # Only pad seq dimension.
                pad_width = [(0, 0)] * (len(input_shape) - 2) + [(0, pad_len), (0, 0)]
                x = jnp.pad(x, pad_width)
                
                # Create mask
                # mask: (..., Seq)
                # ones: (..., seq_len)
                # zeros: (..., pad_len)
                ones = jnp.ones((*batch_shape, seq_len))
                zeros = jnp.zeros((*batch_shape, pad_len))
                mask = jnp.concatenate([ones, zeros], axis=-1)
            else:
                mask = jnp.ones((*batch_shape, seq_len))
        
        # Run TRM
        trm = TRM(
            hidden_size=self.hidden_size,
            expansion=self.expansion,
            num_heads=self.num_heads,
            mlp_t=self.mlp_t,
            seq_len=self.max_seq_len if self.mlp_t else seq_len,
            H_cycles=self.H_cycles,
            L_cycles=self.L_cycles
        )
        
        out = trm(x, mask=mask)
        
        # Extract outputs
        # We want to predict actions for joints.
        # The first Nr_Joints tokens correspond to joints.
        nr_joints = joint_emb.shape[-2]
        joint_out = out[..., :nr_joints, :]
        
        # Predict mean and logstd
        policy_mean = nn.Dense(1, kernel_init=orthogonal(0.01))(joint_out)
        policy_mean = jnp.clip(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)
        
        # For logstd, we can use the output or a separate parameter like in original policy
        # Original policy: policy_logstd = nn.Dense(1)(action_description_latent)
        # We can use joint_out
        policy_logstd = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(np.log(self.std_dev)))(joint_out)
        policy_logstd = jnp.clip(policy_logstd, np.log(self.policy_std_min_clip), np.log(self.policy_std_max_clip))
        
        return policy_mean.squeeze(-1), policy_logstd.squeeze(-1)
