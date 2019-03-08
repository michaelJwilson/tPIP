import numpy as np


def bin_smooth(x, iterate=0):
  if iterate == 0:
    return  (x + np.roll(x, -1))[:-1] / 2.

  else:
    x = bin_smooth(x, iterate=0)

    return  bin_smooth(x, iterate-1)
