from typing import List

from matplotlib import pyplot as plt


def plot_x_vs_y(X: List[int], Y: List[float]) -> None:
    plt.scatter(X, Y, label='Data')
