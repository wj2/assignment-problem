
import numpy as np
import assignment.overlapping_features as am
import os 
import pickle

if __name__ == '__main__':
    n_features = (4, 5, 6, 7, 8, 9, 10, 12)
    total_units = np.logspace(3, 6, 15, dtype=int)
    total_pwrs = np.logspace(1, 4, 15)
    overlaps = (1, 2, 3, 4, 5)

    out = am.explore_fi_tradeoff_parallel(total_units, n_features, overlaps,
                                          total_pwrs)
    savename = 'many_tradeoffs.pkl'
    pickle.dump(out, open(os.path.join('assignment/', savename), 'wb'))