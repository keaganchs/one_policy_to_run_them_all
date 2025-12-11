import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from one_policy_to_run_them_all.algorithms.uni_trm.models.trm_policy import TRM

class TRMCritic(nn.Module):
    # TRM Config
    hidden_size: int = 256
    expansion: float = 2.0
    num_heads: int = 4
    mlp_t: bool = False
    H_cycles: int = 2
    L_cycles: int = 2
    max_seq_len: int = 64

    @nn.compact
    def __call__(self, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_state):
        # Embeddings (Same as Policy)
        joint_emb = nn.Dense(self.hidden_size)(jnp.concatenate([dynamic_joint_description, dynamic_joint_state], axis=-1))
        foot_emb = nn.Dense(self.hidden_size)(jnp.concatenate([dynamic_foot_description, dynamic_foot_state], axis=-1))
        general_emb = nn.Dense(self.hidden_size)(general_state)[..., None, :]
        
        x = jnp.concatenate([joint_emb, foot_emb, general_emb], axis=-2)
        
        input_shape = x.shape
        seq_len = input_shape[-2]
        batch_shape = input_shape[:-2]
        
        mask = None
        if self.mlp_t:
            if seq_len > self.max_seq_len:
                x = x[..., :self.max_seq_len, :]
            elif seq_len < self.max_seq_len:
                pad_len = self.max_seq_len - seq_len
                pad_width = [(0, 0)] * (len(input_shape) - 2) + [(0, pad_len), (0, 0)]
                x = jnp.pad(x, pad_width)
                
                ones = jnp.ones((*batch_shape, seq_len))
                zeros = jnp.zeros((*batch_shape, pad_len))
                mask = jnp.concatenate([ones, zeros], axis=-1)
            else:
                mask = jnp.ones((*batch_shape, seq_len))
        
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
        
        # Critic output: Value
        # We can pool the output or use a specific token.
        # Let's use mean pooling over the sequence (masked).
        
        if mask is not None:
            # out: (..., Max_Seq, Hidden)
            # mask: (..., Max_Seq)
            out = out * mask[..., None]
            pooled = out.sum(axis=-2) / jnp.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
        else:
            pooled = out.mean(axis=-2)
            
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(pooled)
        return value
