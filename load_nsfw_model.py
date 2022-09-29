# %%
import autokeras as ak
from tensorflow.keras.models import load_model  # type: ignore
from urllib.request import urlretrieve
import zipfile
import torch
import torch.nn as nn
import numpy as np

MAIN = __name__ == "__main__"

# %%
def nsfw_normalize(a, nsfw_model):
    return (a - nsfw_model.get_weights()[5]) / (nsfw_model.get_weights()[6]) ** 0.5


def load_pytorch_and_keras_nsfw_models(device):
    """
    Loads pretrained keras NSFW image classifier and transfers weight and architecture to pytorch - returns a tuple of both models.
    """
    path_to_zip_file = "clip_autokeras_binary_nsfw.zip"
    url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
    urlretrieve(url_model, path_to_zip_file)
    import zipfile  # pylint: disable=import-outside-toplevel

    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall("")

    nsfw_model = load_model(
        "clip_autokeras_binary_nsfw", custom_objects=ak.CUSTOM_OBJECTS
    )

    nsfw = nn.Sequential(
        nn.Linear(768, 64),
        nn.ReLU(),
        nn.Linear(64, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    weights = nsfw_model.get_weights()
    nsfw[0].weight.data = torch.from_numpy(weights[8].T)
    nsfw[0].bias.data = torch.from_numpy(weights[9])
    nsfw[2].weight.data = torch.from_numpy(weights[10].T)
    nsfw[2].bias.data = torch.from_numpy(weights[11])
    nsfw[4].weight.data = torch.from_numpy(weights[12].T)
    nsfw[4].bias.data = torch.from_numpy(weights[13])
    nsfw[6].weight.data = torch.from_numpy(weights[14].T)
    nsfw[6].bias.data = torch.from_numpy(weights[15])
    nsfw.to(device)

    return nsfw, nsfw_model
