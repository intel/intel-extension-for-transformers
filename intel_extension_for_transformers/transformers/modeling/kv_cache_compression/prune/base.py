class PruneConfig(dict):
    def __init__(self, real_drop=True):
        self.real_drop = real_drop

class KVPruner:
    def __init__(self, prune_config) -> None:
        self._past_length = 0
        self.prune_kv_cache_size = None

    def self_attn_init(self, module):
        pass

    def prune(self, module, query_states, key_states, value_states, **kwargs):
        pass

    def before_generate(self, model, inputs, *args, **kwargs):
        self.past_length = 0

    def after_generate(self, model, inputs, *args, **kwargs):
        pass

    def get_mask(self, model, **kwargs):
        pass

    @property
    def past_length(self):
        return self._past_length
    
    @past_length.setter
    def past_length(self, value):
        self._past_length = value