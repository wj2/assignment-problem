
import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
from datetime import datetime

import general.utility as u
import assignment.ff_integration as ff
import assignment.overlapping_features as am

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file',
                        default='assignment/ff_models/ff_{}.pkl',
                        type=str,
                        help='folder to save the output in')
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--common_dims', default=1, type=int)
    parser.add_argument('--in_out_units', default=200, type=int)
    parser.add_argument('--integ_units', default=2000, type=int)
    parser.add_argument('--n_samples', default=50000, type=int)
    parser.add_argument('--hidden_units', default=None, type=int,
                        nargs='+')
    parser.add_argument('--pop_func', default='random', type=str)
    parser.add_argument('--connectivity_gen', default='learn_nonlinear_piece',
                        type=str)
    parser.add_argument('--err_n_est', default=5000, type=int)
    parser.add_argument('--dist_args', default=(0, .5, 50),
                        type=float, nargs=3)
    parser.add_argument('--n_stim', default=(2,), type=int,
                        nargs='+')
    parser.add_argument('--input_snr', default=np.sqrt(20), type=float)
    parser.add_argument('--input2_snr', default=None, type=float)
    parser.add_argument('--no_integ', default=False, action='store_true')
    parser.add_argument('--n_reps', default=10, type=int)
    parser.add_argument('--f1_ramp_tuning', default=None, nargs='+',
                        type=int)
    parser.add_argument('--f2_ramp_tuning', default=None, nargs='+',
                        type=int)
    parser.add_argument('--activation_func', default='relu',
                        type=str)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, 'rb'))
        args = u.merge_params_dict(args, config_dict)

    if args.hidden_units is not None:
        args.hidden_units = tuple(args.hidden_units)

    p = 1
    input_dists = (sts.uniform(0, p),)*args.input_dim

    s1, s2 = am.split_integer(args.input_dim - args.common_dims, 2)
    all_dims = np.arange(args.input_dim, dtype=int)
    cds = all_dims[:args.common_dims]
    uds1 = np.concatenate((cds, all_dims[args.common_dims:args.common_dims+s1]))
    uds2 = np.concatenate((cds, all_dims[-s2:]))

    recon_inds = all_dims[-(s1+s2):]
    args.inds = (cds, uds1, uds2, recon_inds)

    dists = np.linspace(*args.dist_args[:2], int(args.dist_args[2]))
    n_est = args.err_n_est
    n_stim = args.n_stim
    noise_mag = 0

    m_rates = {ns: np.zeros((args.n_reps, len(dists), n_est)) for ns in n_stim}
    t_full_rates = {}
    t_fi_rates = {}
    t_emp_rates = {}

    for i in range(args.n_reps):
        m = ff.RandomPopsModel(args.in_out_units, args.in_out_units,
                               args.in_out_units, input_dists,
                               uds1, uds2, recon_inds, 
                               integ_units=args.integ_units,
                               pop_func=args.pop_func,
                               connectivity_gen=args.connectivity_gen,
                               epochs=args.n_epochs, verbose=True,
                               hu_units=args.hidden_units,
                               n_samples=args.n_samples,
                               inp_pwr=args.input_snr**2,
                               inp2_pwr=args.input2_snr**2,
                               f1_rd=args.f1_ramp_tuning,
                               f2_rd=args.f2_ramp_tuning,
                               no_integ=args.no_integ,
                               act_func=args.activation_func)

        for ns in n_stim:
            m_rates[ns][i] = m.estimate_ae_rate_dists(ns, dists, noise_mag,
                                                      n_est=n_est)
            t_full_rates[ns] = m.get_theoretical_ae(dists, n_stim=ns)
            t_fi_rates[ns] = m.get_theoretical_ae(dists, n_stim=ns,
                                                  use_full=False)
            t_emp_rates[ns] = m.get_theoretical_ae(
                dists,
                n_stim=ns,
                use_full=False,
                use_emp_fi_pred_ind=(0, 0),
            )

    args.date = datetime.now()
    fname = args.output_file
    to_save = (vars(args), dists, m_rates, t_full_rates, t_fi_rates, t_emp_rates)
    pickle.dump(to_save, open(fname, 'wb'))
