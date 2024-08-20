from ..utils import Config



def load_cfg(cfg_path: str):

    cfg_dir = cfg_path
    cfg = Config.fromfile(cfg_dir)

    return cfg