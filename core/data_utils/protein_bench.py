from torchdrug import core, datasets, tasks, models, layers

def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
    cfg = yaml.load(raw_text, Loader=yaml.CLoader)
    cfg = easydict.EasyDict(cfg)

    return cfg

path = "/projects/slmreasoning/yifang/datasets/PEER/download_data.yaml"
cfg = load_config(path)
dataset = core.Configurable.load_config_dict(cfg.dataset)
print(dataset)