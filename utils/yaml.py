import yaml

def load_yaml(path: str):
    with open(path, 'rt') as f:
        stream = f.read()
    
    conf  = yaml.safe_load(stream)
    return conf