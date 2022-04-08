
import os

import numpy as np
import assignment.overlapping_features as am
import pickle as p

if __name__ == '__main__':
    common_kwargs = {'pop_func':'random', 'n_pops':50, 
                     'n_samps_per_pop':20}

    pwrs = np.linspace(5, 80, 20)
    n_units = np.linspace(100, 2000, 5)
    dims = 2
    div = 2

    out_file = 'multi_splits.pkl'
    out_dict = am.estimate_mse(pwrs, n_units, dims, div,
                               **common_kwargs)
    out_dict['params'] = (pwrs, n_units, dims, div)
    p.dump(out_dict, open(out_file, 'wb'))
    
