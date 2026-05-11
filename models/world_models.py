import jax
import jax.numpy as jnp
from flax import nnx
import optax
from modules import MultiStateOutputDeepLSTM
from modules import MLP




class WorldModel(nnx.Module):
    def __init__(
            self,
            num_decoder_layers:int,
            num_attention_heads:int,
            num_deep_lstm_layers:int,
            batch_size,
            model_size:int,
            *,
            rngs:nnx.Rngs
    ):
        self.encoder:WorldModelEncoder = WorldModelEncoder(
            num_heads=num_attention_heads, 
            model_size=model_size, 
            batch_size=batch_size, 
            num_hidden_layers=num_deep_lstm_layers,
            rngs=rngs
        )

        self.decoders:list[WorldModelDecoder] = [
            WorldModelDecoder(
                num_heads=num_attention_heads,
                model_size=model_size,
                rngs=rngs
            ) for _ in range(num_decoder_layers)
        ]

        self.final_norm = nnx.LayerNorm(model_size, rngs=rngs)

        pass
        

    def __call__(self,x_input, q_query):

        kv_memory = self.encoder(x_input)
        x = q_query

        for layer in self.decoders:
            x = layer(x, kv_memory)

        return self.final_norm(x)
    

class WorldModelEncoder(nnx.Module):

    def __init__(self, num_heads:int, model_size:int, batch_size:int, hidden_size:int, num_hidden_layers:int,*, rngs:nnx.Rngs):
        
        self.self_attention = nnx.MultiHeadAttention(
            num_heads= num_heads,
            in_features=model_size,
            qkv_features=model_size,
            out_features=model_size,
            rngs = rngs
        )

        self.layerNorm = nnx.LayerNorm(
            num_features=model_size
        )

        self.feed_forward = nnx.Linear(model_size,model_size,rngs=rngs)

        self.deep_lstm = MultiStateOutputDeepLSTM(
            d_in = model_size,
            d_hidden = hidden_size,
            num_layers = num_hidden_layers,
            batch_size = batch_size,
            rngs = rngs
        )   

    def __call__(self, x:jax.Array):
        # X = [Batch, Entities, Features]
        Batch, Entities, Features = x.shape
        norm_x = self.layerNorm(x)
        attention_x = self.self_attention(norm_x)

        x_fused:jax.Array = attention_x + x

        x_lstm_in = x_fused.reshape(Batch, Entities*Features)
        full_history = self.deep_lstm(x_lstm_in) # [Batch, num_hidden_layers, Lstm Output]

        return full_history



class WorldModelDecoder(nnx.Module):
    def __init__(self, num_heads:int, model_size:int,*, rngs:nnx.Rngs):
        
        self.self_attention = nnx.MultiHeadAttention(
            num_heads= num_heads,
            in_features=model_size,
            qkv_features=model_size,
            out_features=model_size,
            rngs = rngs
        )

        self.norm1 = nnx.LayerNorm(
            num_features=model_size
        )
        self.norm2 = nnx.LayerNorm(
            num_features=model_size
        )
        self.norm3 = nnx.LayerNorm(
            num_features=model_size
        )

        self.feed_forward = nnx.Linear(model_size,model_size,rngs=rngs)

        self.cross_attention = nnx.MultiHeadAttention(
            num_heads= num_heads,
            in_features=model_size,
            qkv_features=model_size,
            out_features=model_size,
            rngs = rngs
        )

    def __call__(self, q, kv):
        # q = [Batch, Entities, Features]
        # kv = [Batch, num_hidden_layers, features]
        residual = q
        q = self.norm1(q)
        attn_out = self.self_attention(inputs_q=q)
        q = residual + attn_out

        residual = q
        q = self.norm2(q)
        cross_out = self.cross_attention(
            inputs_q=q, 
            inputs_k=kv, 
            inputs_v=kv
        )
        q = residual + cross_out

        residual = q
        q = self.norm3(q)
        ff_out = self.feed_forward(q)
        q = residual + ff_out  # [Batch, Entities, Model]

        return q
        
