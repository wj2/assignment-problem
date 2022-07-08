
import pickle
import numpy as np
import superposition_codes.codes as spc

if __name__ == '__main__':
    pwr_range = np.logspace(.5, 2, 100)
    nu_range = np.logspace(2, 3.5, 100, dtype=int)
    fix_ind = 50
    dims = (1, 2)
    n_samps = 1000

    fname = 'assignment/param_sweep_rfs.pkl'
    
    out_pwr = spc.sweep_code_performance(pwr_range, nu_range[fix_ind], dims,
                                         n_samps=n_samps)
    out_nu = spc.sweep_code_performance(pwr_range[fix_ind], nu_range, dims,
                                        n_samps=n_samps)
    out = {'params':(pwr_range, nu_range, dims), 'sweep_ind':fix_ind,
           'pwr_sweep':out_pwr, 'nu_sweep':out_nu}
    pickle.dump(out, open(fname, 'wb'))
    
