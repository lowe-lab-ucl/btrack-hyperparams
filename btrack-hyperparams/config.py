import btrack


def build_config(params: dict) -> dict:
    """Build a config from a set of params."""

    t_config = {
        "MotionModel": btrack.utils.read_motion_model(params),
        "ObjectModel": btrack.utils.read_object_model(params),
        "HypothesisModel": btrack.optimise.hypothesis.read_hypothesis_model(params),
    }

    return t_config


with open("/Users/arl/Dropbox/Code/py3/BayesianTracker/models/cell_config.json", 'r') as config:
    DEFAULT_CONFIG = json.load(config)
