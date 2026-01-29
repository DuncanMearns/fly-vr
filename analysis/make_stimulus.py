import yaml
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


directory = Path(r"C:\Users\murthylab\Desktop\new_stim_test")


if __name__ == "__main__":
    with open(directory.joinpath('actuator_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    duration = 10.  # seconds
    fps = cfg["update_rate"]
    n_frames = int(duration * fps)
    n_cycles = 3
    xmax = 3.  # mm

    x0 = cfg["actuator_origin"]["x"]
    y0 = cfg["actuator_origin"]["y"]

    xy = np.zeros((n_frames, 2))
    xy[:, 1] = y0

    cycle_length = int(n_frames / n_cycles)
    cycle = np.sin(np.arange(cycle_length) * 2 * np.pi / cycle_length)
    xy[:n_cycles * cycle_length, 0] = np.concatenate([cycle] * n_cycles) * xmax + x0

    # plt.plot(xy[:, 0], xy[:, 1])
    # plt.show()

    # np.save(directory.joinpath('stimulus.npy'), xy)
    # velocity = np.diff(xy, axis=0, prepend=xy[[0]]) * 60
    # stimulus = np.hstack([xy, velocity])
    #
    # t = np.linspace(0, duration, n_frames)

    plt.plot(np.arange(len(xy)) / fps, xy[:, 0])
    plt.plot(np.arange(len(xy)) / fps, xy[:, 1])
    plt.show()
