
import numpy as np
import assignment.overlapping_features as am
import os 
import pickle

if __name__ == '__main__':
    n_features = (4, 5, 6, 7, 8, 9, 10, 12)
    total_units = np.logspace(3, 6, 50, dtype=int)
    total_pwrs = np.logspace(1, 4, 50)
    overlaps = (1, 2, 3, 4, 5)
    opt_kind = 'brute'

    out = am.explore_fi_tradeoff_parallel(total_units, n_features, overlaps,
                                          total_pwrs, opt_kind=opt_kind)
    savename = 'many_tradeoffs.pkl'
    pickle.dump(out, open(os.path.join('assignment/', savename), 'wb'))
