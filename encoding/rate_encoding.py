import numpy as np

def rate_encode(image, time_steps=100):
    """
    Converts image pixels into spike trains using rate coding.

    Higher pixel intensity → higher spike probability.
    """
    height, width = image.shape
    spikes = np.zeros((height, width, time_steps))

    for t in range(time_steps):
        spikes[:, :, t] = np.random.rand(height, width) < image

    return spikes
