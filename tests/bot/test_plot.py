import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print(matplotlib.get_backend())

# matplotlib.use('Qt5Cairo')


def plot_test():
    plt.plot(np.arange(100))
    plt.show()


if __name__ == '__main__':
    plot_test()
