import numpy as np
from braindance.analysis.connectivity import ccg
import matplotlib.pyplot as plt


def test_ccg():
    bt1 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    bt2 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    ccg_win = [-10, 10]
    ccg_out, lags = ccg(bt1, bt2, ccg_win=ccg_win)


    # Plot
    # fig, ax = plt.subplots()
    # ax.bar(lags, ccg_out)
    # ax.set_xlabel('Lags')
    # ax.set_ylabel('Count')
    # ax.set_title('Cross-correlogram')
    # plt.show()

    assert (lags == np.arange(-10, 11)).all()
    assert (ccg_out == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 0, 0])).all()


# Run tests
if __name__ == '__main__':
    test_ccg()
    print('All tests passed!')