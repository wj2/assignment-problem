
import argparse
import pickle
import numpy as np
import superposition_codes.codes as spc


def create_parser():
    parser = argparse.ArgumentParser(description='fit several modularizers')
    parser.add_argument('-o', '--output_file',
                        default='assignment/code_param_sweep.pkl', type=str)
    return parser

if __name__ == '__main__':
    pwr_range = np.logspace(.5, 2, 100)
    nu_range = np.logspace(2, 3.2, 100, dtype=int)
    pwr_fix_ind = 50
    nus_fix_ind = 75
    dims = (1, 2)
    n_samps = 10

    parser = create_parser()
    args = parser.parse_args()

    fname = args.output_file
    
    out_pwr = spc.sweep_code_performance(pwr_range, nu_range[pwr_fix_ind], dims,
                                         n_samps=n_samps)
    out_nu = spc.sweep_code_performance(pwr_range[nus_fix_ind], nu_range, dims,
                                        n_samps=n_samps)
    out = {'params':(pwr_range, nu_range, dims), 'pwr_sweep_ind':pwr_fix_ind,
           'nus_sweep_ind':nus_fix_ind,
           'pwr_sweep':out_pwr, 'nu_sweep':out_nu}
    pickle.dump(out, open(fname, 'wb'))
    
