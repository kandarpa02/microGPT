import jax.numpy as jnp
from jax.random import normal, uniform

map_ = {
    'normal': lambda rng, shape, dtype:normal(key=rng, shape=shape, dtype=dtype) * 0.02,
    'normal_scaled': lambda rng, shape, dtype:normal(key=rng, shape=shape, dtype=dtype) * (0.02/jnp.sqrt(2)),
}

class Normal:
    def __init__(self, type:str) -> None:
        self.type = type

    def __call__(self, rng, shape, dtype):
        try:
            fun = map_.get(self.type, lambda *args:None)
            
        except KeyError:
            raise KeyError(f"no such functions '{self.type}', choose from {list(map_.keys())}")
        return fun(rng, shape, dtype)
    
class Constant:
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype):
        return jnp.full(shape=shape, fill_value=self.value, dtype=dtype)