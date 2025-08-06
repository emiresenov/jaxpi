from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count

from torch.utils.data import Dataset


class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        batch = self.data_generation(keys)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")


class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

        return batch
    
class FrequencyConcentrationSampler(BaseSampler):
    def __init__(self, freq_bounds, concentrations, batch_size, rng_key=random.PRNGKey(1234), density_power=2.0):
        """
        Args:
            freq_bounds: jnp.array([w0, wn]) - starting and ending frequency
            concentrations: jnp.array([a, b, ..., n]) - specific concentrations
            batch_size: number of samples per batch
            rng_key: random key for sampling
            density_power: power for non-uniform sampling (higher = more dense toward end)
        """
        super().__init__(batch_size, rng_key)
        self.freq_bounds = freq_bounds
        self.concentrations = concentrations
        self.density_power = density_power
        self.dim = 2  # frequency and concentration
        
        # Calculate how many samples per concentration
        self.n_concentrations = len(concentrations)
        self.samples_per_conc = batch_size // self.n_concentrations
        self.remainder = batch_size % self.n_concentrations

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        """Generates data containing batch_size samples with non-uniform frequency sampling"""
        
        # Split the key for different random operations
        freq_key, conc_key = random.split(key)
        
        # Generate frequency samples with higher density toward the end
        # Use power transformation: u^(1/p) where u ~ Uniform(0,1) and p > 1
        uniform_samples = random.uniform(
            freq_key, 
            shape=(self.batch_size,)
        )
        
        # Transform to get higher density toward 1 (end frequency)
        transformed_samples = uniform_samples ** (1.0 / self.density_power)
        
        # Scale to frequency domain
        w0, wn = self.freq_bounds[0], self.freq_bounds[1]
        frequencies = w0 + transformed_samples * (wn - w0)
        
        # Generate concentration samples
        # Repeat each concentration for approximately equal representation
        conc_indices = jnp.repeat(
            jnp.arange(self.n_concentrations), 
            self.samples_per_conc
        )
        
        # Handle remainder samples
        if self.remainder > 0:
            extra_indices = random.choice(
                conc_key, 
                self.n_concentrations, 
                shape=(self.remainder,)
            )
            conc_indices = jnp.concatenate([conc_indices, extra_indices])
        
        # Shuffle the concentration indices to avoid ordering bias
        shuffle_key = random.split(conc_key)[0]
        conc_indices = random.permutation(shuffle_key, conc_indices)
        
        concentrations_sampled = self.concentrations[conc_indices]
        
        # Combine into batch format [frequency, concentration]
        batch = jnp.column_stack([frequencies, concentrations_sampled])
        
        return batch


class SpaceSampler(BaseSampler):
    def __init__(self, coords, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.coords = coords

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = random.choice(key, self.coords.shape[0], shape=(self.batch_size,))
        batch = self.coords[idx, :]

        return batch


class TimeSpaceSampler(BaseSampler):
    def __init__(
        self, temporal_dom, spatial_coords, batch_size, rng_key=random.PRNGKey(1234)
    ):
        super().__init__(batch_size, rng_key)

        self.temporal_dom = temporal_dom
        self.spatial_coords = spatial_coords

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2 = random.split(key)

        temporal_batch = random.uniform(
            key1,
            shape=(self.batch_size, 1),
            minval=self.temporal_dom[0],
            maxval=self.temporal_dom[1],
        )

        spatial_idx = random.choice(
            key2, self.spatial_coords.shape[0], shape=(self.batch_size,)
        )
        spatial_batch = self.spatial_coords[spatial_idx, :]
        batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)

        return batch
