
import numpy as np
import assignment.overlapping_features as am
import os 
import pickle

if __name__ == '__main__':
    n_features = (4, 5, 6, 7, 8, 9, 10, 12)
    total_units = np.logspace(3, 8, 55, dtype=int)
    total_pwrs = np.logspace(1, 5, 50)
    overlaps = (1, 2, 3, 4, 5)

    out = am.explore_mse_tradeoff_parallel(total_units, n_features, overlaps,
                                           total_pwrs, opt_kind=opt_kind,
                                           n_regions=(1, 2, 3))
    savename = 'many_mse_tradeoffs-nr3.pkl'
    pickle.dump(out, open(os.path.join('assignment/', savename), 'wb'))
