#!/usr/bin/env python

import argparse
import numpy as np
import pickle as p
import datetime
import os

import general.stan_utility as su
import assignment.data_analysis as da
import general.utility as u

def create_parser():
    parser = argparse.ArgumentParser(description='fit mixture model to recall'
                                     ' task experimental data using Stan')
    parser.add_argument('--data_file', type=str,
                        default='../data/spatial_data/dataExp1.csv',
                        help='where to look for the data')
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
    parser.add_argument('--no_arviz', default=False, action='store_true',
                        help='do not store arviz inference data')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    os.chdir(args.runfolder)

    data_file = args.data_file

    stan_format = da.load_spatial_data_stan(data_file)
    stan_format = {'exp1':stan_format}

    control = {'adapt_delta':args.adapt_delta,
               'max_treedepth':args.max_treedepth}
    stan_params = {'iter':args.length, 'control':control, 'chains':args.chains}
    
    rbmm = 8
    rbmv = 100
    rbvm = 10
    rbvv = 100
    dbmm = rbmm
    dbmv = rbmv
    dbvm = rbvm
    dbvv = rbvv
    mdmm = 1
    mdmv = 5
    mdvm = 5
    mdvv = 10
    ermm = 1
    ermv = 5
    ervm = 5
    ervv = 10

    
    prior_dict = {'report_bits_mean_mean':rbmm, 'report_bits_mean_var':rbmv,
                  'report_bits_var_mean':rbvm, 'report_bits_var_var':rbvv,
                  'dist_bits_mean_mean':dbmm, 'dist_bits_mean_var':dbmv,
                  'dist_bits_var_mean':dbvm, 'dist_bits_var_var':dbvv,
                  'mech_dist_mean_mean':mdmm, 'mech_dist_mean_var':mdmv,
                  'mech_dist_var_mean':mdvm, 'mech_dist_var_var':mdvv,
                  'enc_rate_mean_mean':ermm, 'enc_rate_mean_var':ermv,
                  'enc_rate_var_mean':ervm, 'enc_rate_var_var':ervv}
    
    fits_dict = {}
    for k, v in stan_format.items():
        f = da.fit_stan_model(v, prior_dict, model_path=da.spatial_model_guess,
                              **stan_params)
        fits_dict[k] = f
        
    fit_models = su.store_models(fits_dict, store_arviz=not args.no_arviz)
    dt = str(datetime.datetime.now()).replace(' ', '-')
    fname = args.output_pattern.format(dt)
    fname = os.path.join(args.outfolder, fname)
    out_dict = {'models':fit_models, 'prior':prior_dict, 'data':stan_format,
                'stan':stan_params}
    p.dump(out_dict, open(fname, 'wb'))
    

