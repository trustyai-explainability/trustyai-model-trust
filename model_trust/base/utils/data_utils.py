import numpy as np


def nparray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        if len(obj) == 0:
            return obj
        else:
            return [nparray_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        for key in obj.keys():
            obj[key] = nparray_to_list(obj[key])
    return obj
