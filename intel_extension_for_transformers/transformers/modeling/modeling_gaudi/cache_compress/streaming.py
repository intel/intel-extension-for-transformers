from typing import Any, Dict, List, Optional, Tuple
from .cache import Cache
import torch
from .function import (
    true_uniform_quantization_compress,
    true_uniform_quantization_decompress,
    true_outlier_quantization_compress,
    true_outlier_quantization_decompress,
    true_gear_compress,
    true_gear_decompress,
)
from .function import (
    true_uniform_quantization_compress_batchwise,
    true_uniform_quantization_decompress_batchwise,
    true_outlier_quantization_compress_batchwise,
    true_outlier_quantization_decompress_batchwise,
    true_gear_compress,
    true_gear_decompress_batchwise,
    true_gear_compress_batchwise,
)

compress_function = {
    "uniform": true_uniform_quantization_compress,
    "outlier": true_outlier_quantization_compress,
    "gear": true_gear_compress,
    "uniform_batch": true_uniform_quantization_compress_batchwise,
    "outlier_batch": true_outlier_quantization_compress_batchwise,
    "gear_batch": true_gear_compress_batchwise,
}
decompress_function = {
    "uniform": true_uniform_quantization_decompress,
    "outlier": true_outlier_quantization_decompress,
    "gear": true_gear_decompress,
    "uniform_batch": true_uniform_quantization_decompress_batchwise,
    "outlier_batch": true_outlier_quantization_decompress_batchwise,
    "gear_batch": true_gear_decompress_batchwise,
}


class StreamCompressedUnion:
    def __init__(self, compress_kwargs: Optional[Dict[str, Any]] = None):
        self.quantize_bit = compress_kwargs["quantize_bit"]
        self.compress_mode = compress_kwargs["compress_mode"]
        self.min = None
        self.step = None
        self.left = compress_kwargs["left"]
        self.rank = compress_kwargs["rank"]
        self.loop = compress_kwargs["loop"]
        self.dtype = None
        self.shape = None
        self.is_compressed = False
        self.cache = None
        self.values = None
        self.indices = None
        self.p_base = None
        self.q_base = None
        self.counter = 0
        self.gap = compress_kwargs["streaming_gap"]
        self.cache_shape = None
        self.buffer = None
        self.cache_shape = None

    def set_cache(self, input: torch.Tensor):

        self.shape = input.shape
        # # has_inf = torch.isinf(input)
        # # has_nan = torch.isnan(input)
        # # print(self.counter,has_inf.any(),has_nan.any())
        # if self.counter != 1 and self.counter % self.gap == 0:
        #     # self.cache = input[:,:,0:-self.counter,:]
        #     self.buffer = input[:,:,-self.counter:,:].clone()
        #     del input
        # else:

        #     self.cache = input
        #     # self.cache = input
        #     pass
        # print("set_cache",self.counter)
        # print(self.cache.dtype)
        if self.counter == 0 or self.counter % self.gap == 0:
            self.cache = input
            self.cache_shape = input.shape
            self.buffer = None
        else:
            buffer_token = self.counter % self.gap
            self.buffer = input[:, :, -buffer_token:, :].clone()
            del input

    def get_cache(self):
        if self.counter == 0 or self.counter % self.gap == 0:
            return self.cache
        else:
            return self.decompress(True)

    def compress(self):

        input = self.cache
        self.dtype = input.dtype
        self.is_compressed = True
        if self.counter == 0 or self.counter % self.gap == 0:
            if self.compress_mode == "uniform":
                output, shape, min, step = compress_function[self.compress_mode](
                    input, self.quantize_bit
                )
                self.cache = output
                self.min = min
                self.step = step

            elif self.compress_mode == "outlier":
                output, shape, min, step, values, indices = compress_function[
                    self.compress_mode
                ](input, self.quantize_bit, self.left)
                self.cache = output
                self.min = min
                self.step = step

                self.values = values
                self.indices = indices
            elif self.compress_mode == "gear":
                output, shape, min, step, values, indices, p_base, q_base = (
                    compress_function[self.compress_mode](
                        input, self.quantize_bit, self.left, self.rank, self.loop
                    )
                )
                self.cache = output
                self.min = min
                self.step = step

                self.values = values
                self.indices = indices
                self.p_base = p_base
                self.q_base = q_base
            elif self.compress_mode == "uniform_batch":
                output, shape, min, step = compress_function[self.compress_mode](
                    input, self.quantize_bit
                )
                self.cache = output
                self.min = min
                self.step = step

            elif self.compress_mode == "outlier_batch":
                output, shape, min, step, values, indices = compress_function[
                    self.compress_mode
                ](input, self.quantize_bit, self.left)
                self.cache = output
                self.min = min
                self.step = step

                self.values = values
                self.indices = indices
            elif self.compress_mode == "gear_batch":
                output, shape, min, step, values, indices, p_base, q_base = (
                    compress_function[self.compress_mode](
                        input, self.quantize_bit, self.left, self.rank, self.loop
                    )
                )
                self.cache = output
                self.min = min
                self.step = step

                self.values = values
                self.indices = indices
                self.p_base = p_base
                self.q_base = q_base
            self.buffer = None
        else:
            pass
        # print("compress",self.counter)
        # print(self.cache.dtype)

    def decompress(self, flag=False):
        self.is_compressed = flag
        # print("decompress",self.counter)
        # print(self.cache.dtype)
        if self.compress_mode == "uniform":
            output = decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.cache_shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier":
            output = decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.cache_shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear":
            output = decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.cache_shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
                self.p_base,
                self.q_base,
            )
        elif self.compress_mode == "uniform_batch":
            output = decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.cache_shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier_batch":
            output = decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.cache_shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear_batch":
            output = decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.cache_shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
                self.p_base,
                self.q_base,
            )
        # self.clean_cache()
        if self.buffer is not None:
            output = torch.cat([output, self.buffer], dim=2)
        # print(self.counter,output.shape)
        return output

    def clean_cache(self):
        self.is_compressed = False
        self.cache = None
        self.values = None
        self.indices = None
        self.p_base = None
        self.q_base = None
        self.min = None
        self.step = None


class StreamCompressedCache(Cache):
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def increase_idx(self, layer_idx):
        self.key_cache[layer_idx].counter += 1
        self.value_cache[layer_idx].counter += 1

    def __setitem__(
        self, layer_idx: int, key_value_states: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Support for backwards-compatible `past_key_value` assignment, e.g. `past_key_value[0] = (key_states,
        value_states)` to update the cache for the first layer.
        """
        key_states, value_states = key_value_states
        self.key_cache[layer_idx], self.value_cache[layer_idx] = (
            key_states,
            value_states,
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        compress_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]
        # print(isinstance(key_states, Cache))
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # apply compress here if needed
            if compress_kwargs is not None:

                key_union = StreamCompressedUnion(compress_kwargs)
                value_union = StreamCompressedUnion(compress_kwargs)
                key_union.set_cache(key_states)
                value_union.set_cache(value_states)
                self.key_cache.append(key_union)
                self.value_cache.append(value_union)
            else:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
        else:
            if compress_kwargs is not None:
                key_union = self.key_cache[layer_idx]
                value_union = self.value_cache[layer_idx]
                if key_union.is_compressed and value_union.is_compressed:
                    previous_key = key_union.decompress()
                    previous_value = value_union.decompress()
                key_union.set_cache(torch.cat([previous_key, key_states], dim=-2))
                value_union.set_cache(torch.cat([previous_value, value_states], dim=-2))
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
        if compress_kwargs is not None:
            return key_union.get_cache(), value_union.get_cache()
        else:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def compress(self, layer_idx: int):
        if len(self.key_cache) <= layer_idx:
            return
        key_union = self.key_cache[layer_idx]
        value_union = self.value_cache[layer_idx]
        if not key_union.is_compressed and not value_union.is_compressed:
            key_union.compress()
            value_union.compress()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):

                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache