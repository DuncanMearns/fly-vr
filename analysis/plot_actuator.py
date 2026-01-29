from pathlib import Path
import h5py
from matplotlib import pyplot as plt
import numpy as np


directory = Path(r"C:\Users\murthylab\Desktop\Duncan_data\2026_01_29\fly01-2")
filepath = directory / "20260129_1700.video_server.h5"


if __name__ == "__main__":
    with h5py.File(filepath, "r") as f:
        data = f["video/stimulus/actuator_dsm"][:]

    # plt.plot(data[:, 2], data[:, 3])
    # plt.plot(data[:, 2], data[:, 4])

    plt.scatter(data[:, 3], data[:, 4], c=np.arange(len(data[:, 3])))
    plt.gca().axis("equal")
    plt.show()

    # plt.plot(data[:, 2], data[:, 5])
    # plt.plot(data[:, 2], data[:, 6])
    # plt.plot(data[:, 2], data[:, -1])
    # plt.show()
