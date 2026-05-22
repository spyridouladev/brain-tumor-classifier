import os

def get_base_path():
    path = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(path, ".."))