
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.io as sio
import scipy.optimize as siopt
import functools as ft
import general.utility as u
import general.plotting as gpl
import os
import itertools as it
import pickle
import pystan as ps
import pandas as pd
import arviz as av

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

def convert_spatial_data(stan_data):
    out = {}    
    for k, data in stan_data.items():
        subs = np.unique(data['subj_id'])
        out_k = []
        for i, sub in enumerate(subs):
            sub_mask = data['subj_id'] == sub
            sub_dict = {}
            ev = np.expand_dims(data['report_err'][sub_mask], 0)
            sub_dict['error_vec'] = ev
            dev = np.expand_dims(data['stim_errs'][sub_mask], 0)
            sub_dict['dist_error_vec'] = dev
            ns = np.expand_dims(data['num_stim'][sub_mask], 0)
            sub_dict['N'] = ns
            rds = np.expand_dims(data['stim_locs'][sub_mask], 0)
            sub_dict['rel_dists'] = rds
            spos = np.expand_dims(data['stim_poss'][sub_mask], 0)
            sub_dict['stim_poss'] = spos
            out_k.append(sub_dict)
    out[k] = out_k
    return out

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
    diff = np.array(diff)
    g_mask = diff > np.pi
    l_mask = diff < -np.pi
    diff[g_mask] = -np.pi + (diff[g_mask] - np.pi)
    diff[l_mask] = np.pi + (diff[l_mask] + np.pi)
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

def load_spatial_data_stan(file_, sort_pos=True, n_stim=6,
                           report_field='reportColor', trunc=None):
    data = pd.read_csv(file_,)
    if trunc is not None:
        data = data[:trunc]
    mask = data[report_field] == 1
    nan_mask = np.logical_not(np.isnan(data['targetColor']))
    comb_mask = mask & nan_mask
    data = data[comb_mask]
    sd = {}
    sd['subj_id'] = np.array(data['subject'])
    sd['num_stim'] = np.ones_like(sd['subj_id'], dtype=int)*n_stim
    cnames = list('nonTargetColors_{}'.format(i + 1)
                  for i in range(n_stim - 1))
    cnames = ['targetColor'] + cnames
    col_locs = np.array(data[cnames])
    sd['stim_locs'] = normalize_range(col_locs - col_locs[:, 0:1])
    errs = normalize_range(np.expand_dims(data['responseColor'], 1) - col_locs)
    sd['stim_errs'] = errs
    sd['report_err'] = errs[:, 0]

    cnames = list('nonTargetLocations_{}'.format(i + 1)
                  for i in range(n_stim - 1))
    cnames = ['targetLocation'] + cnames
    pos_locs = np.array(data[cnames])
    sd['stim_poss'] = normalize_range(pos_locs - pos_locs[:, 0:1])

    sd['S'] = len(np.unique(data['subject']))
    sd['N'] = n_stim
    sd['T'] = len(errs)
    return sd

def load_model(path):
    lm = pickle.load(open(path, 'rb'))['models']
    assert len(lm.keys()) == 1
    m = list(lm.values())[0]
    return m

def load_models(folder, pattern):
    fls = u.get_matching_files(folder, pattern)
    model_dict = {}
    func_dict = {}
    for fl in fls:
        pth = os.path.join(folder, fl)
        lm = pickle.load(open(pth, 'rb'))
        model_dict.update(lm['models'])
    
    for k, md in model_dict.items():
        fm = get_forward_model(md[0])
        func_dict[k] = fm
    return model_dict, func_dict

def sample_forward_model(n, report_dists, spatial_dists=None, n_samples=1,
                         report_bits=1, dist_bits=1, mech_dist=1, sz=8,
                         spacing=np.pi/4, encoding_rate=None):
    if encoding_rate is not None:
        n_enc_stim = min(sts.poisson(encoding_rate).rvs(), n)
        enc_stim_inds = np.sort(np.random.choice(range(n), n_enc_stim,
                                                 replace=False))
        n = n_enc_stim
        report_dists = report_dists[enc_stim_inds]
        if spatial_dists is not None:
            spatial_dists = spatial_dists[enc_stim_inds]
    else:
        enc_stim_inds = range(n)
    if 0 in enc_stim_inds:
        out = _get_ae_probabilities(n, report_dists, spatial_dists=spatial_dists,
                                    report_bits=report_bits, dist_bits=dist_bits,
                                    mech_dist=mech_dist, sz=sz, spacing=spacing)
        report_distortion, ae_probs = out
        ae_probs[0] = 1 - np.sum(ae_probs[1:n])
        if ae_probs[0] < 0:
            ae_probs[0] = 0
            ae_probs[1:n] = ae_probs[1:n]/np.sum(ae_probs[1:n])
        resps = np.random.choice(range(n), n_samples, p=ae_probs)
        means = report_dists[resps]
        var = sts.norm(0, np.sqrt(report_distortion)).rvs(means.shape)
        samples = means + var
    else:
        samples = sts.uniform(-np.pi, 2*np.pi).rvs(n_samples)
    return samples

def load_spatial_models(folder, pattern):
    fls = list(u.get_matching_files(folder, pattern))
    spatial_model = pickle.load(open(os.path.join(folder, fls[0]), 'rb'))
    models = spatial_model['models']
    data = convert_spatial_data(spatial_model['data'])
    funcs = {}
    for k, md in models.items():
        fm = get_forward_model(md[0], fm=sample_forward_model)
        funcs[k] = fm
    return models, funcs, data

def simulate_data(data, func, spatial=False, n_samples=1):
    data_model = {}
    for (k, d) in data.items():
        fm = func[k]
        k_dat = []
        for i, sub_d in enumerate(d):
            md = model_data(sub_d, fm[i], spatial=spatial, n_samples=n_samples)
            k_dat.append(md)
        data_model[k] = k_dat
    return data_model    


mix_nb_man = {'observed_data':'report_err',
              'log_likelihood':{'report_err':'log_lik'},
              'posterior_predictive':'err_hat',
              'dims':{'report_bits':['subject'],
                      'dist_bits':['subject'],
                      'mech_dist':['subject']}}
mix_man = {'observed_data':'report_err',
           'log_likelihood':{'report_err':'log_lik'},
           'posterior_predictive':'err_hat',
           'dims':{'report_bits':['subject'],
                   'dist_bits':['subject'],
                   'mech_dist':['subject']}}
mix_uniform_man = {'observed_data':'report_err',
                   'log_likelihood':{'report_err':'log_lik'},
                   'posterior_predictive':'err_hat',
                   'dims':{'report_bits':['subject'],
                           'dist_bits':['subject'],
                           'mech_dist':['subject'],
                           'encoding_rate':['subject']}}
mix_uniform_lapse_man = {'observed_data':'report_err',
                         'log_likelihood':{'report_err':'log_lik'},
                         'posterior_predictive':'err_hat',
                         'dims':{'report_bits':['subject'],
                                 'dist_bits':['subject'],
                                 'mech_dist':['subject'],
                                 'encoding_rate':['subject'],
                                 'lapse_rate':['subject']}}
mix_spatial_man = {'observed_data':'report_err',
                   'log_likelihood':{'report_err':'log_lik'},
                   'posterior_predictive':'err_hat',
                   'dims':{'report_bits':['subject'],
                           'dist_bits':['subject'],
                           'mech_dist':['subject']}}
mix_spatial_sg_man = {'observed_data':'report_err',
                      'log_likelihood':{'report_err':'log_lik'},
                      'posterior_predictive':'err_hat',
                      'dims':{'report_bits':['subject'],
                              'dist_bits':['subject'],
                              'mech_dist':['subject'],
                              'enc_rate':['subject']}}
mix_snmd = {'observed_data':'report_err',
            'log_likelihood':{'report_err':'log_lik'},
            'posterior_predictive':'err_hat',
            'dims':{'report_bits':['subject'],
                    'dist_bits':['subject'],
                    'enc_rate':['subject']}}
mix_fu = {'observed_data':'report_err',
          'log_likelihood':{'report_err':'log_lik'},
          'posterior_predictive':'err_hat',
          'dims':{'report_bits':['subject'],
                  'dist_bits':['subject'],
                  'mech_dist':['subject'],
                  'stim_mem':['subject']}}
mix_u = {'observed_data':'report_err',
          'log_likelihood':{'report_err':'log_lik'},
          'posterior_predictive':'err_hat',
          'dims':{'report_bits':['subject'],
                  'mech_dist':['subject'],
                  'stim_mem':['subject']}}

arviz_manifests = {'exper_mix_nb.pkl':mix_nb_man,
                   'exper_mix.pkl':mix_man,
                   'exper_mix_spatial.pkl':mix_spatial_man,
                   'exper_mix_sguess.pkl':mix_spatial_sg_man,
                   'exper_mix_uniform.pkl':mix_uniform_man,
                   'exper_mix_nb_uniform.pkl':mix_uniform_man,
                   'exper_mix_nbu_lapse.pkl':mix_uniform_lapse_man,
                   'exper_mix_snmd.pkl':mix_snmd,
                   'exper_mix_f.pkl':mix_man,
                   'exper_mix_fu.pkl':mix_fu,
                   'exper_mix_noae.pkl':mix_u}

assignment_model = 'assignment/stan_models/exper_mix.pkl'
guess_model = 'assignment/stan_models/exper_mix_noae.pkl'
assignment_f = 'assignment/stan_models/exper_mix_f.pkl'
spatial_model = 'assignment/stan_models/exper_mix_spatial.pkl'
spatial_model_guess = 'assignment/stan_models/exper_mix_sguess.pkl'
spatial_model_snmd = 'assignment/stan_models/exper_mix_snmd.pkl'
spatial_model_set = 'assignment/stan_models/exper_mix_set.pkl'
def fit_stan_model(stan_data, prior_dict, model_path=assignment_model,
                   **stan_params):
    fit_dict = {}
    fit_dict.update(stan_data)
    fit_dict.update(prior_dict)    
    sm = pickle.load(open(model_path, 'rb'))
    fit = sm.sampling(data=fit_dict, **stan_params)
    diags = ps.diagnostics.check_hmc_diagnostics(fit)

    model_name = os.path.split(model_path)[1]
    arv_man = arviz_manifests[model_name]
    n_subj = fit_dict['S']
    arv_man['coords'] = {'subject': list(range(1, n_subj + 1))}
    fit_dict['arviz_manifest'] = arv_man
    out = (fit, fit_dict, diags)
    return out

def ae_spatial_probability(dist_bits, ns, spatial_dists):
    dist = dr_gaussian(dist_bits, ns)
    ae_probs = sts.norm(0, np.sqrt(dist)).cdf(-spatial_dists)
    return ae_probs

def _get_ae_probabilities(n, report_dists, spatial_dists=None, report_bits=1,
                          dist_bits=1, mech_dist=1, sz=8, spacing=np.pi/4):
    report_distortion = dr_gaussian(report_bits, n) + mech_dist
    if n > 1:
        if spatial_dists is None:
            ae_prob, _ = ae_var_discrete(dist_bits, n, spacing=spacing, sz=sz)
            ae_probs = np.ones(n)*ae_prob/(n - 1)
            ae_probs[0] = 0
        else:
            ae_probs = ae_spatial_probability(dist_bits, n,
                                              np.abs(spatial_dists))
    else:
        ae_probs = np.zeros(n)
    return report_distortion, ae_probs

def mse_forward_model(n, report_dists, spatial_dists=None, report_bits=1,
                      dist_bits=1, mech_dist=1, sz=8, spacing=np.pi/4,
                      n_samples=1):
    out = _get_ae_probabilities(n, report_dists, spatial_dists=spatial_dists,
                                report_bits=report_bits, dist_bits=dist_bits,
                                mech_dist=mech_dist, sz=sz, spacing=spacing)
    report_distortion, ae_probs = out
    ae_prob = np.sum(ae_probs)
    ae_pm = np.sum(ae_probs[1:n]*(report_dists[1:n]**2
                                  + report_distortion))
    aes = np.random.rand(n_samples) > ae_prob
    err_ms = np.zeros(n_samples)
    err_ms[:] = np.nan
    err_ms[aes] = 0
    cent_inds = np.random.randint(1, n, size=np.sum(~aes))
    err_ms[~aes] = report_dists[cent_inds]
    err = sts.norm(err_ms, np.sqrt(report_distortion)).rvs()
    return err

def get_forward_model(params, rb='report_bits', dm='mech_dist',
                          db='dist_bits', sz=8, spacing=np.pi/4,
                          fm=mse_forward_model):
    report_bits = np.mean(params.samples[rb], axis=0)
    dist_bits = np.mean(params.samples[db], axis=0)
    mech_dist = np.mean(params.samples[dm], axis=0)
    funcs = []
    for i in range(report_bits.shape[0]):
        rb = report_bits[i]
        db = dist_bits[i]
        md = mech_dist[i]
        fm = ft.partial(fm, report_bits=rb, dist_bits=db,
                        mech_dist=md, sz=sz, spacing=spacing)
        funcs.append(fm)
    return funcs

def generate_spatial_models(rb_m, rb_v, db_m, db_v, md_m, md_v, er_m, er_v,
                            exper_datapath, fm=sample_forward_model, sz=8,
                            spacing=np.pi/4):
    data = pd.read_csv(exper_datapath)
    subids = np.unique(data['subject'])
    rbs = sts.norm(rb_m, rb_v).rvs(len(subids))
    dbs = sts.norm(db_m, db_v).rvs(len(subids))
    mds = sts.norm(md_m, md_v).rvs(len(subids))
    ers = sts.norm(er_m, er_v).rvs(len(subids))
    params = {'report_bits':rbs, 'dist_bits':dbs, 'mech_dist':mds,
              'encoding_rates':ers}
    fms = {}
    for i, subid in enumerate(subids):
        fm = ft.partial(fm, report_bits=rbs[i], dist_bits=dbs[i],
                        mech_dist=mds[i], sz=sz, spacing=spacing,
                        encoding_rate=ers[i])
        fms[subid] = fm
    return fms, params

def generate_pseudo_data(fms, exper_datapath, outname='pseudo.csv',
                         report_field='reportColor'):
    data = pd.read_csv(exper_datapath)
    mask = data[report_field] == 1
    nan_mask = np.logical_not(np.isnan(data['targetColor']))
    comb_mask = np.logical_and(mask, nan_mask)
    data = data[comb_mask]
    reformat_data = load_spatial_data_stan(exper_datapath,
                                           report_field=report_field)
    for i, (row_j, row) in enumerate(data.iterrows()):
        sub = row['subject']
        fm = fms[sub]
        out = fm(reformat_data['num_stim'][i], reformat_data['stim_errs'][i],
                 spatial_dists=reformat_data['stim_poss'][i])
        out_orig = normalize_range(out + row['targetColor'])
        data.at[row_j, 'responseColor'] = out_orig
    path, _ = os.path.split(exper_datapath)
    outpath = os.path.join(path, outname)
    data.to_csv(outpath)
    return data

def model_data(data, fm, dist_field='rel_dists', load_field='N',
               outfield='error_vec', spatial_field='stim_poss',
               spatial=False, n_samples=1):
    out = {}
    mses = np.zeros((data[dist_field].shape[1], n_samples))
    for i, rd in enumerate(data[dist_field][0]):
        n = data[load_field][0, i]
        if spatial:
            samps = fm(n, rd, data[spatial_field][0, i], n_samples=n_samples)
        else:
            samps = fm(n, rd, n_samples=n_samples)
        mses[i] = samps
    out[outfield] = np.expand_dims(mses, 0)
    out[dist_field] = data[dist_field]
    out[load_field] = data[load_field]
    if spatial:
        out[spatial_field] = data[spatial_field]
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

def _make_subj_structure(d, subj_field, *fields):
    subs = np.unique(d[subj_field])
    subs_all = []
    for i, s in enumerate(subs):
        mask = d[subj_field] == s
        sub_d = {}
        for f in fields:
            sub_d[f] = d[f][mask]
        subs_all.append(sub_d)
    return subs_all, subs

def uniform_prob(sm, n):
    prob = 0
    for i in range(0, n + 1):
        p = sts.poisson(sm).pmf(i)
        if i == n:
            p = p + (1 - sts.poisson(sm).cdf(i))
        prob = prob + p*i/n
    return 1 - prob

def model_subj_org(data, org_func=mse_by_load, dist_field='stim_locs',
                   load_field='num_stim', err_field='report_err',
                   subj_field='subj_id', model=None, model_gen='err_hat'):
    org_dict = {}
    for k, ed in data.items():
        if model is not None:
            ed = ed.copy()
            ed[err_field] = model[k].samples[model_gen].T
        sd, subs = _make_subj_structure(ed, subj_field, dist_field, load_field,
                                        err_field)
        org_dict[k] = subj_org(sd, org_func, load_field=load_field,
                               dist_field=dist_field, err_field=err_field)
    return org_dict

def experiment_subj_org(data, org_func=mse_by_load, dist_field='rel_dists',
                        load_field='N', err_field='error_vec'):
    org_dict = {}
    for k in data.keys():
        org_dict[k] = subj_org(data[k], org_func, load_field=load_field,
                               dist_field=dist_field, err_field=err_field)
    return org_dict

def plot_load_mse(data, ax=None, plot_fit=True, max_load=np.inf, boots=None,
                  sep_subj=False, data_color=(.7, .7, .7),
                  **plot_args):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ls, errs = data
    for i, err in enumerate(errs):
        if sep_subj:
            use_ax = ax[i]
        else:
            use_ax = ax
        l = ls[i]
        mask = l <= max_load
        l = l[mask]
        subj = i + 1
        mse = np.array(err)[mask]**2
        if boots is not None:
            mse = np.array(list(u.bootstrap_list(mse_i.flatten(),
                                                 np.nanmean, boots)
                                for mse_i in mse))
        gpl.plot_trace_werr(l, mse, ax=use_ax, label='S{}'.format(subj),
                            jagged=True, color=data_color, **plot_args)
    return ax    

def read_models(dir_, pattern):
    fs = u.get_matching_files(dir_, pattern)
    mod_dict = {}
    for f in fs:
        p = os.path.join(dir_, f)
        m = pickle.load(open(p, 'rb'))
        mod_dict.update(m['models'])
    return mod_dict

def compare_models(dir_, p1, p2, include_sus=True, exclude=('divergence',)):
    md1 = read_models(dir_, p1)
    md2 = read_models(dir_, p2)
    common_mods = set(md1.keys()).intersection(set(md2.keys()))
    print(md2.keys())
    print(common_mods)
    comparisons = {}
    for k in common_mods:
        m1, _, d1 = md1[k]
        m2, _, d2 = md2[k]
        list(d1.pop(e) for e in exclude)
        list(d2.pop(e) for e in exclude)
        if include_sus or (np.all(list(d1.values()))
                           and np.all(list(d2.values()))):
            l1 = av.loo(m1.arviz)
            l2 = av.loo(m2.arviz)
            d = {'p1':m1.arviz, 'p2':m2.arviz}
            c = av.compare(d)
            comparisons[k] = (l1, l2, c)
    return comparisons

def plot_subj_ppc(data, ax=None, plot_fit=True, max_load=np.inf, boots=None,
                  sep_subj=False, data_color=None,
                  **plot_args):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ls, errs = data
    for i, err in enumerate(errs):
        if sep_subj:
            use_ax = ax[i]
        else:
            use_ax = ax
        l = ls[i]
        mask = l <= max_load
        l = l[mask]
        subj = i + 1
        errs_all = np.concatenate(err).flatten()
        use_ax.hist(errs_all, histtype='step', color=data_color,
                    density=True)
    return ax    

def plot_dist_dependence(data, ax=None, eps=1e-4, plot_fit=True, boots=None,
                         sep_subj=False, n_bins=5, need_trials=20,
                         data_color=None, digit_percentile=95, **plot_args):
    if ax is None:
        f, ax = plt.subplots(1,1)
    ls, load_errs = data
    for j, load in enumerate(load_errs):
        if sep_subj:
            use_ax = ax[j]
        else:
            use_ax = ax
        for i, (errs, dists) in enumerate(load):
            if ls[j][i] > 1:
                l = ls[j][i]
                dists = np.min(np.abs(dists[:, 1:l]), axis=1)
                max_bin = np.percentile(dists, digit_percentile)
                bins = np.linspace(0, max_bin + eps, n_bins + 1)
                bin_inds = np.digitize(dists, bins)
                binned_errs = []
                bin_cents = []
                for bi, binbeg in enumerate(bins[:-1]): 
                    mask = bin_inds == bi
                    if np.sum(mask) > need_trials:
                        be = errs[mask].flatten()**2
                        binned_errs.append(be)
                        bin_cents.append((binbeg + bins[bi+1])/2)
                bin_cents = np.array(bin_cents)
                binned_errs = np.array(binned_errs, dtype=object)
                if boots is not None:
                    binned_errs = np.array(list(u.bootstrap_list(be_i,
                                                                 np.mean,
                                                                 boots)
                                                for be_i in binned_errs))
                gpl.plot_trace_werr(bin_cents, binned_errs, ax=use_ax, jagged=True,
                                    color=data_color, **plot_args)
    return ax

def plot_experiment_func(data, plot_func=plot_load_mse, ax_size=(1,1),
                         x_ax='set size (N)', y_ax='color report MSE',
                         use_plot=None, use_same_ax=False, model_data=None,
                         sep_subj=False, boots=None, set_xs=None,
                         **plot_args):
    if model_data is None:
        model_data = {}
    if use_plot is None:
        n_exper = len(data.keys())
        if use_same_ax:
            f, ax = plt.subplots(1, 1, figsize=ax_size)
            axs = (ax,)*n_exper
        elif sep_subj:
            f, ax = {}, {}
            for k in data.keys():
                n_subjs = len(data[k][0])
                figsize = (ax_size[0]*n_subjs, ax_size[1])                
                f_k, ax_k = plt.subplots(1, n_subjs, figsize=figsize,
                                         sharex=True, sharey=True)
                f[k] = f_k
                ax[k] = ax_k
        else:
            figsize = (ax_size[0], ax_size[1]*n_exper)
            f, axs = plt.subplots(n_exper, 1, figsize=figsize)
            if n_exper == 1:
                axs = (axs,)
    else:
        f, axs = use_plot
        ax = {k:axs for k in data.keys()}
    for i, k in enumerate(data.keys()):
        if sep_subj:
            use_ax = ax[k]
        else:
            use_ax = axs[i]
        if k in model_data.keys():
            model = model_data[k]
        else:
            model = None
        _ = plot_func(data[k], ax=use_ax, sep_subj=sep_subj, boots=boots,
                      **plot_args)
        if model is not None:
            orig_color = plot_args['data_color']
            plot_args['data_color'] = None
            _ = plot_func(model, ax=use_ax, sep_subj=sep_subj,
                          boots=boots, **plot_args)
            plot_args['data_color'] = orig_color
        if sep_subj:
            if set_xs is None:
                set_xs = range(use_ax)
            use_ax[0].set_ylabel(y_ax)
            for i, ax_i in enumerate(use_ax):
                if i in set_xs:
                    ax_i.set_xlabel(x_ax)
        else:
            use_ax.set_xlabel(x_ax)
            use_ax.set_ylabel(y_ax)
            use_ax.set_title(k)
    return f, ax

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
    ae_mag = (1/12)*(np.pi*2)**2
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
    
