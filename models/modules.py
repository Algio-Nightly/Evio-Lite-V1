import jax
import jax.numpy as jnp
from flax import nnx
import optax

class MLP(nnx.Module):
    def __init__(self, d_in:int, hidden_states:list, d_out:int, *, rngs:nnx.Rngs):
        layers = []
        current_dims = d_in

        for hidden in hidden_states:
            layers.extend([
                nnx.Linear(current_dims,hidden,rngs=rngs),
                nnx.Dropout(rate = 0.1, deterministic=False),
                nnx.BatchNorm(hidden, use_running_average=True),
                nnx.relu
            ])
            current_dims = hidden
        
        layers.append(nnx.Linear(current_dims, d_out, rngs=rngs))
        
        self.sequential = nnx.Sequential(*layers)


    def __call__(self, x:jax.Array):
        return self.sequential(x)
    

class LSTMState(nnx.Variable):
    pass

class StatefulLSTM(nnx.Module):
    def __init__(self, d_in:int, d_hidden:int, *, rngs:nnx.Rngs):
        self.cell = nnx.LSTMCell(d_in, d_hidden,rngs=rngs)
        

        self.c:LSTMState = LSTMState(jnp.zeros((1, d_hidden)))
        self.h:LSTMState = LSTMState(jnp.zeros((1, d_hidden)))

    def __call__(self, x:jax.Array):
        batch_size = x.shape[0]

        if self.c.value.shape[0] != batch_size:
            self.c.value = jnp.zeros((batch_size, self.d_hidden))
            self.h.value = jnp.zeros((batch_size, self.d_hidden))
        
        carry = (self.c.value, self.h.value)

        carry, y = self.cell(carry, x)
        
        self.c.value, self.h.value = carry

        return y
        
    def reset(self):
        self.c.value = jnp.zeros_like(self.c.value)
        self.h.value = jnp.zeros_like(self.h.value)
        

class MultiStateOutputDeepLSTM(nnx.Module):
    def __init__(self, d_in:int, d_hidden:int, num_layers:int, *, rngs:nnx.Rngs):
        
        self.layers = []

        self.layers.append(StatefulLSTM(d_in,d_hidden, rngs = rngs))

        for _ in range(num_layers-1):
            self.layers.append(StatefulLSTM(d_hidden, d_hidden, rngs = rngs))
        
        
    
    def __call__(self, x:jax.Array):
        all_layer_states = []

        layer:StatefulLSTM
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            else:
                x = x + layer(x)
                
            all_layer_states.append(x)

        multi_layer_memory = jnp.stack(all_layer_states, axis=1)

        return multi_layer_memory

    def reset_all(self):
        layer:StatefulLSTM
        for layer in self.layers:
            layer.reset()

# Use nnx.jit   
