import numpy as np
import requests
import torch
from PIL import Image


def infer(image: Image.Image, endpoint: str, transform) -> float:
    """Sends an image to the MLflow/Triton inference server and returns the sigmoid probability.

    Args:
        image (Image.Image): The input PIL image.
        endpoint (str): The HTTP URL destination for the model invocations.
        transform: The evaluation image transformation pipeline.

    Returns:
        float: The confidence score/probability of the positive class.

    Raises:
        requests.exceptions.RequestException: If the server connection fails.
        RuntimeError: If the server returns a non-200 status code.
    """
    tensor_numpy = transform(image).cpu().numpy()
    payload = {"inputs": np.expand_dims(tensor_numpy, axis=0).tolist()}

    response = requests.post(endpoint, json=payload, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(
            f"Ошибка сервера (Код {response.status_code}): {response.text}"
        )

    logits = response.json()["predictions"]
    output = logits["output"] if isinstance(logits, dict) else logits
    probability = float(torch.sigmoid(torch.tensor(output)).item())

    return probability
