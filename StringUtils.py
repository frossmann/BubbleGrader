#%%
import string
import numpy as np
from datetime import datetime


def tprint(string):
    print("    " + string)


def get_alpha(num_options):
    alphabet = list(string.ascii_lowercase)
    alpha = dict(zip(np.arange(num_options), alphabet[:num_options]))
    return alpha


def timestamp():
    now = datetime.now()
    timestamp = (
        str(now.year)
        + "_"
        + str(now.month)
        + "_"
        + str(now.day)
    )
    return timestamp
# %%
