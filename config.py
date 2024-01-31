
def find_target_size(model_name):
    target_sizes = {
        "vggface": (224, 224),
        "facenet128": (160, 160),
        "facenet512": (160, 160),
        "openface": (96, 96),
        "deepid": (47, 55),
        "arcFace": (112, 112),
    }
    target_size = target_sizes.get(model_name.lower())
    if target_size == None:
        raise ValueError(f"unimplemented model name - {model_name}")
    return target_size

def find_threshold(model_name,distance_metric):
    thresholds = {
        "vggface": {"cosine": 0.68,"euclidean": 1.17 },
        "facenet128": {"cosine": 0.40, "euclidean": 10 },
        "facenet512": {"cosine": 0.30, "euclidean": 23.56 },
        "arcface": {"cosine": 0.68, "euclidean": 4.15 },
        "openface": {"cosine": 0.10, "euclidean": 0.55 },
        "deepid": {"cosine": 0.015, "euclidean": 45 },
    }
    if model_name not in thresholds:
        raise ValueError(f"unimplemented model name - {model_name}")
    threshold = thresholds.get(model_name).get(distance_metric, 0.4)
    if threshold == None :
        raise ValueError(f"unimplemented model name - {model_name} - {distance_metric}")
    return threshold

