#%%
import string
import numpy as np


def tprint(string):
    print("    " + string)


def get_alpha(num_options):
    alphabet = list(string.ascii_lowercase)
    alpha = dict(zip(np.arange(num_options), alphabet[:num_options]))
    return alpha

# %%
