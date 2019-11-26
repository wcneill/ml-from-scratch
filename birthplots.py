import birth_names as bn
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def getAxPopularity(name, gender):
    files = bn.getPaths()
    ave = bn.getAverageRank(name, gender, select=False, filez=files)
    highest = bn.yearOfHighestRank(name, gender, select=False, filez=files)
    rank_data = bn.getAllRanks(name, gender, files)

    x = rank_data[:,0]
    y = rank_data[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(211) # type:plt.Axes
    ax.plot(x,y, '-')
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Births named {}".format(name))
    plt.axhline(ave, color='red', linestyle='dashed', label="Average")
    plt.legend()
    # plt.show()

    return fig


if __name__ == '__main__':
    fig = getAxPopularity("Victoria", "F")
    ax2 = fig.add_subplot(212)
    ax2.plot(np.random.randint(1,10,10))
    plt.show()


