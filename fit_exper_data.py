#!/usr/bin/env python

import argparse
import numpy as np
import pickle as p
import datetime
import os
import arviz as az

import general.stan_utility as su
import assignment.data_analysis as da
import general.utility as u

def create_parser():
    parser = argparse.ArgumentParser(description='fit mixture model to recall'
                                     ' task experimental data using Stan')
    parser.add_argument('--data_folder', type=str, default='../data/bays_data/',
                        help='where to look for the data')
    parser.add_argument('experiments', type=int, nargs='+',
                        help='experiment numbers to fit')
    parser.add_argument('--runfolder', default='./', type=str,
                        help='path to run the script from')
    parser.add_argument('--output_pattern', default='recall_{}.pkl',
                        type=str, help='pattern for output filename')
    parser.add_argument('--outfolder', default='../data/recall_fits/',
                        type=str, help='folder to save output in')
    parser.add_argument('--chains', type=int, default=4, help='number of '
                        'Monte Carlo chains to use')
    parser.add_argument('--length', type=int, default=2000, help='length of '
                        'each chain')
    parser.add_argument('--adapt_delta', type=float, default=.8,
                        help='adapt_delta value to use')
    parser.add_argument('--max_treedepth', type=int, default=10,
                        help='maximum tree depth to use')
    parser.add_argument('--model_path', default=da.assignment_model,
                        help='path to pkld stan model to use')
    parser.add_argument('--no_arviz', default=False, action='store_true',
                        help='do not store arviz inference data')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    os.chdir(args.runfolder)

    experiments = args.experiments
    data_folder = args.data_folder

    data = da.load_data(data_folder, experiment_num=experiments,
                        sort_pos=True, collapse_subs=False)

    stan_format = da.format_experiments_stan(data)

    control = {'adapt_delta':args.adapt_delta,
               'max_treedepth':args.max_treedepth}
    stan_params = {'iter':args.length, 'control':control, 'chains':args.chains}
    
    rbmm = 5
    rbmv = 10
    rbvm = 2
    rbvv = 5
    
    dbmm = rbmm
    dbmv = rbmv
    dbvm = rbvm
    dbvv = rbvv
    
    mdmm = 1
    mdmv = 2
    mdvm = .5
    mdvv = .5

    enmm = 3
    enmv = 2
    envm = 2
    envv = 1

    lrbk = 7.5
    lrbt = .5
    lrak = 1
    lrat = .5

    prior_dict = {'report_bits_mean_mean':rbmm, 'report_bits_mean_var':rbmv,
                  'report_bits_var_mean':rbvm, 'report_bits_var_var':rbvv,
                  'dist_bits_mean_mean':dbmm, 'dist_bits_mean_var':dbmv,
                  'dist_bits_var_mean':dbvm, 'dist_bits_var_var':dbvv,
                  'mech_dist_mean_mean':mdmm, 'mech_dist_mean_var':mdmv,
                  'mech_dist_var_mean':mdvm, 'mech_dist_var_var':mdvv,
                  'encoding_rate_mean_mean':enmm, 'encoding_rate_mean_var':enmv,
                  'encoding_rate_var_mean':envm, 'encoding_rate_var_var':envv,
                  'lr_beta_k':lrbk, 'lr_beta_t':lrbt, 'lr_alpha_k':lrak,
                  'lr_alpha_t':lrat}
    
    fits_dict = {}
    for k, v in stan_format.items():
        f = da.fit_stan_model(v, prior_dict, model_path=args.model_path,
                              **stan_params)
        fits_dict[k] = f
        
    fit_models = su.store_models(fits_dict, store_arviz=not args.no_arviz)
    dt = str(datetime.datetime.now()).replace(' ', '-')
    fname = args.output_pattern.format(dt)
    fname = os.path.join(args.outfolder, fname)
    out_dict = {'models':fit_models, 'prior':prior_dict, 'data':data,
                'stan':stan_params}
    p.dump(out_dict, open(fname, 'wb'))
    

