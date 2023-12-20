import os
from typing import Union, Optional
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
sentence_transformers = LazyImport("sentence_transformers")

def get_module_path(model_name_or_path: str, 
                    path: str,
                    token: Optional[Union[bool, str]], 
                    cache_folder: Optional[str]):
    is_local = os.path.isdir(model_name_or_path)
    if is_local:
        return os.path.join(model_name_or_path, path)
    else:
        return sentence_transformers.util.load_dir_path(
            model_name_or_path, path, token=token, cache_folder=cache_folder)