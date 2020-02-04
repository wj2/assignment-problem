
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.io as sio
import scipy.optimize as siopt
import general.utility as u
import general.plotting as gpl
import os
import itertools as it
import pickle
import pystan as ps

def load_data(folder, experiment_num=None, keep_experiments=None,
              pattern='E{}_subject_.*\.mat',
              exper_field='experiment_name', data_field='data',
              infer_positions=True, dict_convert=True, sort_pos=False,
              collapse_subs=False):
    if experiment_num is None:
        insert = '[0-9]*'
    else:
        insert = '({})'.format('|'.join(str(en) for en in experiment_num))
    e_pattern = pattern.format(insert)
    l = os.listdir(folder)
    fls = u.get_matching_files(folder, e_pattern)
    out_info = {}
    for i, fl in enumerate(fls):
        ff = os.path.join(folder, fl)
        data_mat = sio.loadmat(ff)
        experiment_name = data_mat[exper_field][0, 0][0]
        if keep_experiments is None or experiment_name in keep_experiments:
            subj_results = data_mat[data_field][0, 0]
            if dict_convert or infer_positions:
                subj_results = _convert_to_dict(subj_results)
            if experiment_name not in out_info.keys():
                out_info[experiment_name] = []
            if infer_positions:
                subj_results = add_relative_positions(subj_results,
                                                      sort=sort_pos)
            out_info[experiment_name].append(subj_results)
    if collapse_subs and dict_convert:
        new_out = {}
        for k in out_info.keys():
            new_sub = {}
            for sk in out_info[k][0].keys():
                sub_key_list = [sub[sk] for sub in out_info[k]]
                sub_collapsed = np.concatenate(sub_key_list, axis=1)
                new_sub[sk] = sub_collapsed
            new_out[k] = [new_sub]
        out_info = new_out
    return out_info

def _convert_to_dict(subj_results):
    d = {}
    for name in list(subj_results.dtype.fields.keys()):
        d[name] = subj_results[name]
    return d

def add_relative_positions(data, err_field='error_vec',
                           dist_field='dist_error_vec', sort=False):
    relative_distances = infer_relative_positions(data, err_field, dist_field,
                                                  sort=sort)
    rel_dists = np.array([relative_distances], dtype=object)
    data['rel_dists'] = rel_dists
    return data

def normalize_range(diff):
    if diff > np.pi:
        diff = -np.pi + (diff - np.pi)
    elif diff < -np.pi:
        diff = np.pi + (diff + np.pi)
    return diff

def infer_relative_positions(data, err_field='error_vec',
                             dist_field='dist_error_vec', sort=False):
    relative_distances = []
    for i, err in enumerate(data[err_field][0]):
        d_errs = data[dist_field][0, i]
        rel_ds = (0.,)
        for de in d_errs:
            rd = normalize_range(err + de[0])
            rel_ds = rel_ds + (rd,)
        if sort:
            sort_inds = np.argsort(np.abs(rel_ds))
            rel_ds = np.array(rel_ds)[sort_inds]
        relative_distances.append(rel_ds)
    return relative_distances

def format_experiments_stan(data, stim_spacing=None, stim_spacing_all=np.pi/4,
                            err_field='error_vec', num_field='N',
                            pos_field='rel_dists',
                            stim_err_field='dist_error_vec'):
    stan_dicts = {}
    for k in data.keys():
        exper = data[k]
        n_subs = len(exper)
        if stim_spacing is None:
            k_stim_spacing = stim_spacing_all
        else:
            k_stim_spacing = stim_spacing[k]
        sd = {}
        for i, s in enumerate(exper):
            errs = s[err_field][0]
            sub_ind = np.ones_like(errs)*(i + 1)
            n_stim = s[num_field][0]
            if i == 0:
                max_n = np.max(s[num_field])
            stim_locs = np.zeros((len(errs), max_n))
            stim_errs = np.zeros((len(errs), max_n))
            stim_errs[:, 0] = errs
            for j, t in enumerate(s[pos_field][0]):
                curr_n = s[num_field][0, j]
                stim_locs[j, :curr_n] = t
                if curr_n > 1:
                    stim_errs[j, 1:curr_n] = s[stim_err_field][0, j][:, 0]
            if i == 0:
                all_errs = errs
                all_stim_locs = stim_locs
                all_sub_ind = sub_ind
                all_n_stim = n_stim
                all_stim_errs = stim_errs
            else:
                all_errs = np.concatenate((all_errs, errs))
                all_stim_locs = np.concatenate((all_stim_locs, stim_locs))
                all_sub_ind = np.concatenate((all_sub_ind, sub_ind))
                all_n_stim = np.concatenate((all_n_stim, n_stim))
                all_stim_errs = np.concatenate((all_stim_errs, stim_errs))
        sd['report_err'] = all_errs
        sd['num_stim'] = all_n_stim
        sd['stim_locs'] = all_stim_locs
        sd['stim_errs'] = all_stim_errs
        sd['subj_id'] = all_sub_ind.astype(int)
        sd['S'] = n_subs
        sd['N'] = np.max(all_n_stim)
        sd['T'] = len(all_errs)
        sd['stim_spacing'] = k_stim_spacing
        stan_dicts[k] = sd
    return stan_dicts

assignment_model = 'assignment/stan_models/exper_mix.pkl'
def fit_stan_model(stan_data, prior_dict, model_path=assignment_model,
                   **stan_params):
    fit_dict = {}
    fit_dict.update(stan_data)
    fit_dict.update(prior_dict)
    sm = pickle.load(open(model_path, 'rb'))
    fit = sm.sampling(data=fit_dict, **stan_params)
    diags = ps.diagnostics.check_hmc_diagnostics(fit)
    out = (fit, fit_dict, diags)
    return out

def mse_by_load(data, **field_keys):
    load_field = field_keys['load_field']
    err_field = field_keys['err_field']
    loads = np.unique(data[load_field])
    errs = []
    for i, l in enumerate(loads):
        mask = data[load_field] == l
        load_dat = data[err_field][mask]
        errs.append(load_dat)
    return loads, errs

def mse_by_dist(data, dist_field='rel_dists', load_field='N',
                err_field='error_vec'):
    loads = np.unique(data[load_field])
    dists = data[dist_field]
    errs = []
    for i, l in enumerate(loads):
        mask = data[load_field] == l
        load_errs = data[err_field][mask]
        load_dists = data[dist_field][mask]
        load_dists = np.stack(load_dists, axis=1).T
        errs.append((load_errs, load_dists))
    return loads, errs

def subj_org(data, org_func=mse_by_load, **field_keys):
    all_mses = []
    all_ls = []
    for i, d in enumerate(data):
        ls, es = org_func(d, **field_keys)
        all_mses.append(es)
        all_ls.append(ls)
    return all_ls, all_mses

def experiment_subj_org(data, org_func=mse_by_load, dist_field='rel_dists',
                        load_field='N', err_field='error_vec'):
    org_dict = {}
    for k in data.keys():
        org_dict[k] = subj_org(data[k], org_func, load_field=load_field,
                               dist_field=dist_field, err_field=err_field)
    return org_dict

def plot_load_mse(data, ax=None, plot_fit=True, max_load=np.inf, **plot_args):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ls, errs = data
    for i, err in enumerate(errs):
        l = ls[i]
        mask = l <= max_load
        l = l[mask]
        subj = i + 1
        mse = np.array(err)[mask]**2
        gpl.plot_trace_werr(l, mse, ax=ax, label='S{}'.format(subj),
                            jagged=True, **plot_args)
        if plot_fit:
            pred_f, b, rms = fit_mse_assign_dependence(mse, l)
            pred = pred_f(l)
            fl = gpl.plot_trace_werr(l, pred, ax=ax)
            pred_f2, b2, rms2 = fit_mse_dependence(mse, l)
            pred2 = pred_f2(l)
            gpl.plot_trace_werr(l, pred2, ax=ax, linestyle='--',
                                color=fl[0].get_color())
            out = (rms, rms2)
    return ax    

def plot_dist_dependence(data, ax=None, eps=1e-4, plot_fit=True, **plot_args):
    n_bins = plot_args.pop('n_bins')
    need_trials = plot_args.pop('need_trials')
    if ax is None:
        f, ax = plt.subplots(1,1)
    ls, load_errs = data
    for j, load in enumerate(load_errs):
        for i, (errs, dists) in enumerate(load):
            if ls[j][i] > 1:
                dists = np.abs(dists[:, 1])
                bins = np.linspace(0, np.max(dists) + eps, n_bins)
                bin_inds = np.digitize(dists, bins)
                binned_errs = []
                bin_cents = []
                for k, bi in enumerate(np.unique(bin_inds)):
                    mask = bin_inds == bi
                    if np.sum(mask) > need_trials:
                        binned_errs.append(errs[mask]**2)
                        bin_cents.append((bins[bi-1] + bins[bi])/2)
                bin_cents = np.array(bin_cents)
                binned_errs = np.array(binned_errs)
                gpl.plot_trace_werr(bin_cents, binned_errs, ax=ax, jagged=True,
                                **plot_args)
                if plot_fit:
                    if 'central_tendency' in plot_args.keys():
                        cent_func = plot_args['central_tendency']
                    else:
                        cent_func = np.nanmean
                    cents =list(cent_func(be) for be in binned_errs)
                    flat_val = np.mean(cents)
                    pred = np.ones_like(bin_cents)*flat_val
                    gpl.plot_trace_werr(bin_cents, pred, ax=ax)

    return ax

def plot_experiment_func(data, plot_func=plot_load_mse, ax_size=(2,2),
                         x_ax='set size (N)', y_ax='MSE', use_plot=None,
                         use_same_ax=False, **plot_args):
    if use_plot is None:
        n_exper = len(data.keys())
        if use_same_ax:
            f, ax = plt.subplots(1, 1, figsize=ax_size)
            axs = (ax,)*n_exper
        else:
            figsize = (ax_size[0], ax_size[1]*n_exper)
            f, axs = plt.subplots(n_exper, 1, figsize=figsize)
            if n_exper == 1:
                axs = (axs,)
    else:
        f, axs = use_plot
    for i, k in enumerate(data.keys()):
        ax = axs[i]
        ax = plot_func(data[k], ax=ax, **plot_args)
        ax.set_xlabel(x_ax)
        ax.set_ylabel(y_ax)
        ax.set_title(k)
    return f, axs

def rd_gaussian(mse, n, tiny_eps=1e-5, correct=True):
    if correct:
        mse[mse < tiny_eps] = tiny_eps
    b = n*np.log(np.sqrt(2*np.pi/mse))
    return b

def dr_gaussian(b, n):
    d = 2*np.pi*np.exp(-2*b/n)
    return d

def compute_naive_mse_dependence(mse_distrib, n=1):
    mse_char = np.mean(mse_distrib)
    b = rd_gaussian(mse, n)
    pred_func = lambda other_n: 2*np.pi*np.exp(-2*b/other_n)
    return pred_func, b

def fit_mse_dependence(mses, ns, init_eps=1e-2, tiny_eps=1e-5):
    all_mses = np.array(list(np.mean(m) for m in mses))

    def _min_func(fixed_mse):
        b = rd_gaussian(all_mses - fixed_mse, ns)
        b_var = np.sum((b - np.mean(b))**2)
        return b_var

    min_mse = np.min(all_mses)
    res = siopt.minimize(_min_func, min_mse - init_eps,
                         bounds=((tiny_eps, min_mse - tiny_eps),))
    final_b = np.mean(rd_gaussian(all_mses - res.x, ns))
    pred_func = lambda n: dr_gaussian(final_b, n) + res.x
    fit_rms = np.sqrt(np.mean((all_mses - pred_func(ns))**2))
    return pred_func, res, fit_rms

def discrete_space(sz, n, check=False):
    sz1 = sz - 1
    n1 = n - 1
    total = ss.comb(sz1, n1)
    both = ss.comb(sz1 - 2, n1 - 2)
    either = 2*ss.comb(sz1 - 2, n1 - 1)
    if check:
        combs = list(it.combinations(range(1, sz), n - 1))
        print('total', total, len(combs))
        print('both', both, np.sum(1 in c and 2 in c for c in combs))
        print('either', either, 2*np.sum((1 in c and 2 not in c)
                                         for c in combs))
        print(combs)
    return both/total, either/total, total
    
def ae_var_discrete(loc_b, ns, spacing=np.pi/4, sz=8, tiny_eps=1e-5):
    dist = dr_gaussian(loc_b, ns)
    d = sts.norm(0, np.sqrt(dist))
    prob = d.cdf(-spacing) 
    boths, eithers, total = discrete_space(sz, ns)
    ae_likely = prob*boths*2 + prob*eithers
    ae_likely = np.min((np.stack(ae_likely),
                        np.ones_like(ae_likely) - tiny_eps),
                       axis=0)
    ae_mag = ((1/12)*(np.pi*2)**2)
    return ae_likely, ae_mag

def ae_var_continuous(loc_b, ns, tiny_eps=1e-5):
    dist = dr_gaussian(loc_b, ns)
    b = 1/np.sqrt(2*dist)
    distrib = sts.norm(0, 1)
    ft = distrib.cdf(-b*np.pi)
    st = distrib.pdf(-b*np.pi)
    tt = distrib.pdf(0)
    ae_mag = ((1/12)*(np.pi*2)**2)
    pre_fact = (1/np.pi)*(ns - 1)
    ae_likely = pre_fact*(tt/b + np.pi*ft - st/b)
    ae_likely = np.min((np.stack(ae_likely),
                        np.ones_like(ae_likely) - tiny_eps),
                       axis=0)
    return ae_likely, ae_mag

def get_reg_mse(ams, fms, ae_prob, ae_mag):
    raw_reg = ams - (1 - ae_prob)*fms - ae_prob*ae_mag
    return raw_reg/(1 - ae_prob)

def get_total_mse(rms, fms, ae_prob, ae_mag):
    ams = (1 - ae_prob)*(rms + fms) + ae_prob*ae_mag
    return ams

def fit_mse_assign_dependence(mses, ns, init_loc_b=10, init_eps=1e-2,
                              tiny_eps=1e-5, ae_var=ae_var_discrete):
    all_mses = np.array(list(np.mean(m) for m in mses))
    min_mse = np.min(all_mses)

    def mse_constraint(x):
        fixed_mse, loc_b = x
        ae_likely, ae_mag = ae_var(loc_b, ns)
        rms = get_reg_mse(all_mses, fixed_mse, ae_likely, ae_mag)
        return np.min(rms - tiny_eps)

    def fixed_mse_constraint(x):
        fixed_mse, _ = x
        out = min(min_mse - tiny_eps - fixed_mse, fixed_mse - tiny_eps)
        return out

    def loc_b_constraint(x):
        _, loc_b = x
        return loc_b - tiny_eps
        
    def _min_func(x):
        fixed_mse, loc_b = x
        ae_likely, ae_mag = ae_var(loc_b, ns)
        reg_ms = get_reg_mse(all_mses, fixed_mse, ae_likely, ae_mag)
        b = rd_gaussian(reg_ms, ns)
        b_var = np.sum((b - np.mean(b))**2)
        return b_var

    mse_constraint = {'type':'ineq', 'fun':mse_constraint}
    fixed_mse_constraint = {'type':'ineq', 'fun':fixed_mse_constraint}
    loc_b_constraint = {'type':'ineq', 'fun':loc_b_constraint}
    constraints = (mse_constraint, fixed_mse_constraint, loc_b_constraint)
    init_conds = (min_mse - init_eps,
                  init_loc_b)
    res = siopt.minimize(_min_func, init_conds,
                         constraints=constraints)
    final_ae_likely, final_ae_mag = ae_var(res.x[1], ns)
    final_reg_rms = get_reg_mse(all_mses, res.x[0], final_ae_likely,
                                final_ae_mag)
    final_b = np.mean(rd_gaussian(final_reg_rms, ns))
    print(dr_gaussian(res.x[1], ns))
    print(final_ae_likely*final_ae_mag)
    print(res.x[0], res.x[1])
    pred_func = lambda n: get_total_mse(dr_gaussian(final_b, n), res.x[0],
                                        *ae_var(res.x[1], n))
    fit_rms = np.sqrt(np.mean((all_mses - pred_func(ns))**2))
    return pred_func, res, fit_rms
    
