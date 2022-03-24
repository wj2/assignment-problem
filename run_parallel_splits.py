
import os

import numpy as np
import assignment.overlapping_features as am
import pickle as p

if __name__ == '__main__':
    common_kwargs = {'pop_func':'random', 'n_pops':8, 
                     'n_samps_per_pop':20}

    pwrs = np.linspace(20, 60, 10)
    n_units = np.linspace(100, 2000, 5)
    dims = 2
    div = 2

    out_file = 'multi_splits.pkl'
    out_dict = am.estimate_mse(pwrs, n_units, dims, div,
                               **common_kwargs)
    p.dump(out_dict, open(out_file, 'wb'))
    
