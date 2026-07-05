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

    def to_dict(self):
        """Recursively convert AttrDict back to a plain dict."""
        result = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def coerce_config_types(obj):
    """
    Best-effort recursive coercion of config values that arrive as strings.

    Config travels between the dashboard, the controller, and the offline
    drivers as JSON / HTML-form text, so numeric hyperparameters routinely
    arrive as strings ("50", "-1", "0.999") and boolean flags as
    "true"/"false". Historically every consumer had to remember to wrap its
    read in int()/float(), and a single miss is a silent bug -- e.g. the
    string "-1" is never == the int -1, so a `cti_period != -1` guard fires
    when it should not. Coercing once at the config boundary
    (TigerBrain.__init__) makes the values arrive already typed, so the
    scattered int()/float() call sites become defensive rather than load-bearing.

    Only strings that unambiguously parse as a bool/int/float are converted;
    everything else (agent names, paths, class lists, model source text, IPs)
    is returned unchanged, so this cannot corrupt genuinely-string fields.
    """
    if isinstance(obj, dict):
        return {k: coerce_config_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_config_types(v) for v in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        low = stripped.lower()
        if low in ('true', 'false'):
            return low == 'true'
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            return float(stripped)
        except ValueError:
            pass
    return obj
