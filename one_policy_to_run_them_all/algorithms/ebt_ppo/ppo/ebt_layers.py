"""
EBT (Energy-Based Transformer) components ported to JAX/Flax for RL applications.
Based on the original PyTorch implementation in the EBT repository.
"""

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, normal
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        dims = x.shape[-1]
        weight = self.param('weight', lambda rng, shape: jnp.ones(shape), (dims,))
        
        # Compute RMS norm
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.epsilon)
        
        return normed * weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[:dim // 2] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    # Convert to complex exponentials
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis


def apply_rotary_emb(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply rotary embeddings to query and key tensors.
    """
    # Reshape to complex representation
    xq_ = jnp.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = jnp.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    
    # Apply rotation
    freqs_cis = freqs_cis[:xq_.shape[1]]  # Adjust to sequence length
    xq_out = jnp.view_as_real(xq_ * freqs_cis).reshape(xq.shape)
    xk_out = jnp.view_as_real(xk_ * freqs_cis).reshape(xk.shape)
    
    return xq_out, xk_out


class EBTAttention(nn.Module):
    """
    Energy-Based Transformer Attention mechanism.
    
    This implements the core EBT attention that can attend to both 
    original states and predicted next states with energy-based weighting.
    """
    dim: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout: float = 0.0
    use_energy_attention: bool = True
    energy_scale: float = 1.0

    def setup(self):
        self.n_kv_heads = self.n_kv_heads or self.n_heads
        self.head_dim = self.dim // self.n_heads
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        
        # Linear projections for queries, keys, values
        self.wq = nn.Dense(self.n_heads * self.head_dim, use_bias=False)
        self.wk = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False)
        self.wv = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False)
        self.wo = nn.Dense(self.dim, use_bias=False)
        
        self.dropout_layer = nn.Dropout(self.dropout)

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 freqs_cis: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of EBT attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask
            freqs_cis: Precomputed rotation frequencies
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape
        
        # Linear projections
        xq = self.wq(x)  # (batch, seq_len, n_heads * head_dim)
        xk = self.wk(x)  # (batch, seq_len, n_kv_heads * head_dim)
        xv = self.wv(x)  # (batch, seq_len, n_kv_heads * head_dim)
        
        # Reshape for multi-head attention
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings if provided
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Transpose for attention computation
        xq = jnp.transpose(xq, (0, 2, 1, 3))  # (batch, n_heads, seq_len, head_dim)
        xk = jnp.transpose(xk, (0, 2, 1, 3))
        xv = jnp.transpose(xv, (0, 2, 1, 3))
        
        # Compute attention scores
        scores = jnp.matmul(xq, jnp.transpose(xk, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        # Apply energy-based attention if enabled
        if self.use_energy_attention:
            # Compute energy-based weighting (simplified version)
            energy = jnp.sum(xq * xk, axis=-1, keepdims=True) * self.energy_scale
            scores = scores + energy.squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = self.dropout_layer(attn_weights, deterministic=not training)
        
        # Apply attention to values
        output = jnp.matmul(attn_weights, xv)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape back
        output = jnp.transpose(output, (0, 2, 1, 3))  # (batch, seq_len, n_heads, head_dim)
        output = output.reshape(bsz, seqlen, -1)  # (batch, seq_len, dim)
        
        # Final linear projection
        return self.wo(output)


class EBTFeedForward(nn.Module):
    """Feed-forward network for EBT."""
    dim: int
    hidden_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # SwiGLU activation (similar to original EBT)
        w1 = nn.Dense(self.hidden_dim, use_bias=False)(x)
        w2 = nn.Dense(self.dim, use_bias=False)
        w3 = nn.Dense(self.hidden_dim, use_bias=False)(x)
        
        # Apply SiLU (swish) activation
        hidden = nn.silu(w1) * w3
        hidden = nn.Dropout(self.dropout)(hidden, deterministic=not training)
        
        return w2(hidden)


class EBTTransformerBlock(nn.Module):
    """Single transformer block with EBT attention."""
    dim: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout: float = 0.0
    use_energy_attention: bool = True
    energy_scale: float = 1.0

    def setup(self):
        self.attention = EBTAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            dropout=self.dropout,
            use_energy_attention=self.use_energy_attention,
            energy_scale=self.energy_scale
        )
        
        # Feed-forward with 4x hidden dimension (standard transformer)
        hidden_dim = 4 * self.dim
        self.feed_forward = EBTFeedForward(
            dim=self.dim,
            hidden_dim=hidden_dim,
            dropout=self.dropout
        )
        
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 freqs_cis: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        # Pre-norm architecture
        # Attention block with residual connection
        h = x + self.attention(
            self.attention_norm(x), 
            mask=mask, 
            freqs_cis=freqs_cis, 
            training=training
        )
        
        # Feed-forward block with residual connection
        out = h + self.feed_forward(self.ffn_norm(h), training=training)
        
        return out


class EBTEncoder(nn.Module):
    """
    Energy-Based Transformer encoder for RL state processing.
    
    This processes robot state sequences and outputs representations
    suitable for policy and value function approximation.
    """
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    dropout: float = 0.1
    max_seq_len: int = 64
    use_energy_attention: bool = True
    energy_scale: float = 1.0

    def setup(self):
        # Precompute rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            self.dim // self.n_heads, self.max_seq_len
        )
        
        # Transformer layers
        self.layers = [
            EBTTransformerBlock(
                dim=self.dim,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                dropout=self.dropout,
                use_energy_attention=self.use_energy_attention,
                energy_scale=self.energy_scale
            )
            for _ in range(self.n_layers)
        ]
        
        self.norm = RMSNorm()

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass through EBT encoder.
        
        Args:
            x: Input embeddings of shape (batch, seq_len, dim)
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Encoded representations of shape (batch, seq_len, dim)
        """
        seqlen = x.shape[1]
        freqs_cis = self.freqs_cis[:seqlen]
        
        # Pass through transformer layers
        h = x
        for layer in self.layers:
            h = layer(h, mask=mask, freqs_cis=freqs_cis, training=training)
        
        # Final layer norm
        return self.norm(h)