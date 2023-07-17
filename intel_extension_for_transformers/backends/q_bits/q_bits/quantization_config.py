import copy
import json
import os
import torch
from typing import Any, Dict, Union


class QBitsConfig:
    def __init__(
        self,
        quant_bits=8,
        llm_int8_skip_modules=None,
        compute_dtype=None,
        quant_type="int8",
        use_double_quant=False,
        group_size=None,
        scheme="sym",
        **kwargs,
    ):
        self.quant_bits = quant_bits
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.quant_type = quant_type
        self.use_double_quant = use_double_quant
        self.scheme = scheme

        if group_size is None:
            self.group_size = 32
        else:
            self.group_size = group_size
        if compute_dtype is None:
            self.compute_dtype = torch.float32
        elif isinstance(compute_dtype, str):
            self.compute_dtype = getattr(torch, compute_dtype)
        elif isinstance(compute_dtype, torch.dtype):
            self.compute_dtype = compute_dtype
        else:
            raise ValueError("bit4_compute_dtype must be a string or a torch.dtype")

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise ValueError("llm_int8_skip_modules must be a list of strings")

        if self.compute_dtype is not None and not isinstance(self.compute_dtype, torch.dtype):
            raise ValueError("compute_dtype must be torch.dtype")

        if not isinstance(self.quant_type, str):
            raise ValueError("quant_type must be a string")

        if not isinstance(self.use_double_quant, bool):
            raise ValueError("use_double_quant must be a boolean")

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be a int")

        if not isinstance(self.scheme, str):
            raise ValueError("scheme must be a string")

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model.
        """
        if self.quant_bits == 8:
            return "int8"
        elif self.quant_bits == 4 and self.quant_type == "int4":
            return "int4"
        else:
            raise ValueError("Only support int8 and int4 quantization now!")

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs, **kwargs):
        """
        Instantiates a [`QBitsConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QBitsConfig`]: The configuration object instantiated from those parameters.
        """

        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """

        output = copy.deepcopy(self.__dict__)
        output["compute_dtype"] = str(output["compute_dtype"]).split(".")[1]

        return output

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `QBitsConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = QBitsConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict