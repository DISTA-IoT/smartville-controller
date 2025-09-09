class AttrDict(dict):
    """Dict with attribute-style access, recursive for nested dicts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__