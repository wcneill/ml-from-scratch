import birth_names as bn
import matplotlib.pyplot as plt


def plotRankAndScores(name, gender):
    files = bn.getPaths()
    x1, y1 = bn.getAllRanks(name, gender, files)
    x2, y2 = bn.getAllScores(name, gender, files)

    # For those years when the rank of a name was 0 (eg no new births given that name)
    # we need to assign last place to that name
    min_rank = max(y1)
    for i, rank in enumerate(y1):
        if y1[i] == 0:
            y1[i] = min_rank + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(10, 10))
    plt.xlabel("Year")
    ax1.plot(x1, y1, 'b')
    ax1.set_ylabel("Rank")
    ax1.axhline(y1.mean(), label='average = {}'.format(y1.mean()), linestyle='--', color='red')
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.legend()
    ax2.plot(x2, y2, 'b')
    ax2.set_ylabel("Number of Births")
    ax2.axhline(y2.mean(), label='average = {}'.format(y2.mean()), linestyle='--', color='red')
    ax2.legend()
    plt.suptitle("Name Rank and Number of Births by Year: {}".format(name))
    plt.show()


if __name__ == '__main__':
    plotRankAndScores("Victoria", "F")
