#!/usr/bin/env python

import argparse
import numpy as np
import pickle as p
import datetime
import os

import general.stan_utility as su
import assignment.correlated_estimates as ce

def create_parser():
    parser = argparse.ArgumentParser(description='empirically compute '
                                     'posterior for RF model')
    parser.add_argument('-k', '--n_neurs', default=100, type=int,
                        help='numbers of neurons in the population')
    parser.add_argument('-s','--stim_size', default=100, type=float,
                        help='stimulus space size')
    parser.add_argument('-t', '--n_stims', default=2, type=int,
                        help='number of stimuli in the scene')
    parser.add_argument('-n', '--n_samps', default=200, type=int,
                        help='number of samples for each scene')
    parser.add_argument('--rf_scale', default=1, type=float,
                        help='peak RF response')
    parser.add_argument('--rf_width', default=15, type=float,
                        help='RF width')
    parser.add_argument('--noise_var', default=.2, type=float,
                        help='neuron response variance')
    parser.add_argument('--first_stim', default=5, type=float,
                        help='closest distance between stimuli')
    parser.add_argument('--end_stim', default=10, type=float,
                        help='furthest distance between stimuli')
    parser.add_argument('--n_scenes', default=2, type=int,
                        help='number of scenes between two distances')
    parser.add_argument('--buffer', default=20, type=float,
                        help='buffer away from edges of stim space')
    parser.add_argument('--model_path', default=ce.gt_model,
                        help='path to stan model pkl file')
    parser.add_argument('--not_parallel', default=False, action='store_true',
                        help='do not run in parallel (done by default)')
    parser.add_argument('--runfolder', default='./', type=str,
                        help='path to run the script from')
    parser.add_argument('--output_pattern', default='rf-posterior_{}.pkl',
                        type=str, help='pattern for output filename')
    parser.add_argument('--outfolder', default='../data/posterior_fits/',
                        type=str, help='folder to save output in')
    parser.add_argument('--chains', type=int, default=4, help='number of '
                        'Monte Carlo chains to use')
    parser.add_argument('--length', type=int, default=2000, help='length of '
                        'each chain')
    parser.add_argument('--adapt_delta', type=float, default=.8,
                        help='adapt_delta value to use')
    parser.add_argument('--max_treedepth', type=int, default=10,
                        help='maximum tree depth to use')
    parser.add_argument('--full_data', default=False, action='store_true',
                        help='collapse data across days and fit it all at '
                        'once -- does not do this by default')
    parser.add_argument('--init', default='random', type=str,
                        help='initial value for model fitter')
    parser.add_argument('--init_eps', default=10e-4, type=float,
                        help='initial value offset epsilon')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    os.chdir(args.runfolder)

    if args.init != 'random':
        init_stim = ({'stims':np.linspace(-args.init_eps, args.init_eps,
                                          args.n_stims)},)*args.chains
    else:
        init_stim = args.init

    control = {'adapt_delta':args.adapt_delta,
               'max_treedepth':args.max_treedepth}
    stan_params = {'iter':args.length, 'control':control, 'chains':args.chains,
                   'init':init_stim}
    if args.not_parallel:
        n_jobs = 1
    else:
        n_jobs = -1
    stan_params['n_jobs'] = n_jobs
    
    n_neurs = args.n_neurs
    size = args.stim_size
    n_stim = args.n_stims
    n_samps = args.n_samps
    mult_scale = args.rf_scale
    mult_wid = args.rf_width
    noise_var = args.noise_var
    stim_beg = args.first_stim
    stim_end = args.end_stim
    n_scenes = args.n_scenes
    buff = args.buffer
    model_path = args.model_path

    cents = np.linspace(-size/2, size/2, n_neurs)
    scales = np.ones(n_neurs)*mult_scale
    wids = np.ones(n_neurs)*mult_wid
    cov = np.identity(n_neurs)*noise_var
    
    dists = np.linspace(stim_beg, stim_end, n_scenes)

    data = {'N':n_samps, 'K':n_neurs, 'C':n_stim, 'S':size, 
        'buffer':buff}
    rf_params = {'cents':cents, 'scales':scales, 'wids':wids, 'cov_mat':cov}

    fits = ce.estimate_posterior_series(data, dists, rf_params,
                                        model_path=model_path,
                                        verbose=True, **stan_params)

    fit_models = su.store_models(fits)
    dt = str(datetime.datetime.now()).replace(' ', '-')
    fname = args.output_pattern.format(dt)
    fname = os.path.join(args.outfolder, fname)
    out_dict = {'models':fit_models, 'rf_params':rf_params, 'data':data,
                'dists':dists}
    p.dump(out_dict, open(fname, 'wb'))
    
