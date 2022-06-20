
import warnings
import numpy as np
import itertools as it
import functools as ft
import scipy.special as ss
import scipy.stats as sts
import scipy.integrate as sin
import scipy.optimize as sio
import joblib as jl

import general.utility as u
import general.rf_models as rfm

def _split_integer(num, parts):
    """ taken from 
    https://stackoverflow.com/questions/55465884/
    how-to-divide-an-unknown-integer-into-a-given-
    number-of-even-parts-using-python """
    quotient, remainder = divmod(num, parts)
    lower_elements = list(quotient for i in range(parts - remainder))
    higher_elements = list(quotient + 1 for j in range(remainder))
    return lower_elements + higher_elements

def _compute_assignment_distortion(n_feats, overlaps, num_regions):
    overlaps = np.array(overlaps)
    left_out = np.zeros((len(n_feats), len(overlaps)))
    for (i, j) in u.make_array_ind_iterator(left_out.shape):
        nf = n_feats[i]
        overlap = overlaps[j]
        left_out[i, j] = nf - max(_split_integer(nf + overlap, num_regions))
    left_out[left_out <= 0] = 0
    ws = left_out/6
    # ws[ws <= 0] = 0
    ws = np.expand_dims(ws, 0)
    ws = np.expand_dims(ws, -1)
    return ws    

def weighted_tradeoff(distorts, ae_rates, n_features, overlaps, overlap_dim=2,
                      n_feat_dim=1):
    weight = {nr:_compute_assignment_distortion(n_features, overlaps, nr)
              for nr in distorts.keys()}
    w_arr = {nr:distorts[nr] + weight[nr]*ae_rates[nr] for nr in distorts.keys()}
    w_arr[1] = np.mean(w_arr[1], axis=overlap_dim, keepdims=True)
    diff_mat2 = w_arr[1] - w_arr[2]
    diff_mat3 = w_arr[1] - w_arr[3]
    opt_overlap2 = np.argmin(w_arr[2], axis=2).astype(float)
    opt_overlap3 = np.argmin(w_arr[3], axis=2).astype(float)
    opt_overlap2 = np.argmax(diff_mat2, axis=2).astype(float)
    opt_overlap3 = np.argmax(diff_mat3, axis=2).astype(float)
    opt_overlap2[np.all(diff_mat2 <= 0, axis=overlap_dim)] = np.nan
    opt_overlap3[np.all(diff_mat3 <= 0, axis=overlap_dim)] = np.nan
    return w_arr, opt_overlap2, opt_overlap3


def max_fi_per_feature(pwr_per_feature, n_units_per_feature, dims,
                       ret_min_max=True, use_w=None, **kwargs):
    fi_out = np.zeros((len(dims), len(n_units_per_feature)))
    w_opt_out = np.zeros_like(fi_out)
    pwr_out = np.zeros_like(fi_out)
    for i, dim in enumerate(dims):
        pwr_i = dim*pwr_per_feature
        n_units_i = n_units_per_feature*dim
        for j, n_units_ij in enumerate(n_units_i):
            if use_w is not None:
                use_w_ij = use_w[i, j]
            else:
                use_w_ij = None
            out = rfm.max_fi_power(pwr_i, n_units_ij, dim,
                                   ret_min_max=ret_min_max,
                                   use_w=use_w_ij, **kwargs)
            fi_out[i, j] = out[0]
            w_opt_out[i, j] = out[3]
            pwr_out[i, j] = out[2]
    return fi_out, w_opt_out, pwr_out

def post_estimated_mse(est_arr, conf_dict, pwr=None, n_units=None,
                       dim=None):
    new_arr = np.zeros(est_arr.shape + est_arr[0, 0, 0, 0].shape)
    mse_fi_arr = np.zeros(est_arr.shape)
    
    for ind in u.make_array_ind_iterator(est_arr.shape):
        new_arr[ind] = est_arr[ind]
        mse_fi_arr[ind] = 1/conf_dict[ind[:2]][0][0, 0]
        if pwr is not None and n_units is not None:
            pwr_i = pwr[ind[0]]
            n_i = n_units[ind[1]]
            w_use = conf_dict[ind[:2]][1]
            resp_use = conf_dict[ind[:2]][2]
            prob, mag = rfm.compute_threshold_err_prob(pwr_i, n_i, dim, w_use,
                                                       resp_scale=resp_use)
            mse_fi_arr[ind] = (1 - prob)*mse_fi_arr[ind] + prob*mag
    return new_arr, mse_fi_arr

def estimate_mse(pwrs, n_units, dims, div=2, **kwargs):
    pwrs = np.array(pwrs)
    n_units = np.array(n_units).astype(int)
    out_full, c_full = rfm.emp_rf_decoding(pwrs, n_units, dims, **kwargs)
    out_div, c_div = rfm.emp_rf_decoding(pwrs/div, (n_units/div).astype(int),
                                         dims/div, **kwargs)
    ret_dict = {0:(out_full, c_full),
                div:(out_div, c_div)}
    return ret_dict

@u.arg_list_decorator
def explore_fi_tradeoff(n_units, total_dims, overlaps, total_pwrs,
                        n_regions=(1, 2), **kwargs):
    arr_shape = (len(n_units), len(total_dims), len(overlaps),
                 len(total_pwrs))
    distorts = {nr:np.zeros(arr_shape) for nr in n_regions}
    ae_rates = {nr:np.zeros_like(distorts[nr]) for nr in n_regions}
    for ind in u.make_array_ind_iterator(arr_shape):
        i, j, k, l = ind
        ds, aers = fi_tradeoff(n_units[i], total_dims[j], n_regions=n_regions,
                               overlap=overlaps[k], total_pwr=total_pwrs[l],
                               **kwargs)
        for k, d_k in ds.items():
            distorts[k][ind] = np.mean(d_k)
            ae_rates[k][ind] = np.mean(aers[k])
    return distorts, ae_rates

@u.arg_list_decorator
def explore_mse_tradeoff_parallel(n_units, total_dims, overlaps, total_pwrs,
                                  n_regions=(1, 2), n_jobs=-1, **kwargs):
    arr_shape = (len(n_units), len(total_dims), len(overlaps),
                 len(total_pwrs))
    totals = {nr:np.zeros(arr_shape) for nr in n_regions}
    distorts = {nr:np.zeros(arr_shape) for nr in n_regions}
    ae_rates = {nr:np.zeros_like(distorts[nr]) for nr in n_regions}
    mis_prob = {nr:np.zeros(arr_shape, dtype=object) for nr in n_regions}
    mis_err = {nr:np.zeros(arr_shape, dtype=object) for nr in n_regions}

    def _mse_tradeoff_helper(ind):
        i, j, k, l = ind
        out = mse_tradeoff(n_units[i], total_dims[j], n_regions=n_regions,
                           overlap=overlaps[k], total_pwr=total_pwrs[l],
                           **kwargs)
        total, ds, aers, mp, me = out
        return ind, total, ds, aers, mp, me

    args = (n_units, total_dims, overlaps, total_pwrs)
    ind_iter = u.make_array_ind_iterator(arr_shape)
    par = jl.Parallel(n_jobs=n_jobs)
    out = par(jl.delayed(_mse_tradeoff_helper)(ind) for ind in ind_iter)
    for ind, total, ds, aers, mp, me in out:
        for k, d_k in ds.items():
            totals[k][ind] = total[k]
            distorts[k][ind] = np.sum(d_k) # np.product(d_k)/np.sum(d_k)
            ae_rates[k][ind] = np.sum(aers[k])
            mis_prob[k][ind] = mp[k]
            mis_err[k][ind] = me[k]
            
    return args, totals, distorts, ae_rates, mis_prob, mis_err

@u.arg_list_decorator
def explore_fi_tradeoff_parallel(n_units, total_dims, overlaps, total_pwrs,
                                 n_regions=(1, 2), n_jobs=-1, **kwargs):
    arr_shape = (len(n_units), len(total_dims), len(overlaps),
                 len(total_pwrs))
    distorts = {nr:np.zeros(arr_shape) for nr in n_regions}
    ae_rates = {nr:np.zeros_like(distorts[nr]) for nr in n_regions}

    def _fi_tradeoff_helper(ind):
        i, j, k, l = ind
        ds, aers = fi_tradeoff(n_units[i], total_dims[j], n_regions=n_regions,
                               overlap=overlaps[k], total_pwr=total_pwrs[l],
                               **kwargs)
        return ind, ds, aers

    args = (n_units, total_dims, overlaps, total_pwrs)
    ind_iter = u.make_array_ind_iterator(arr_shape)
    par = jl.Parallel(n_jobs=n_jobs)
    out = par(jl.delayed(_fi_tradeoff_helper)(ind) for ind in ind_iter)
    for ind, ds, aers in out:
        for k, d_k in ds.items():
            distorts[k][ind] = np.mean(d_k) # np.product(d_k)/np.sum(d_k)
            ae_rates[k][ind] = np.mean(aers[k])
    return args, distorts, ae_rates

def mse_tradeoff(total_units, total_dims, n_regions=(1, 2), overlap=1,
                 n_stim=2, total_pwr=10, ret_min_max=True, lambda_deviation=2,
                 **kwargs):
    distorts = {}
    ae_distorts = {}
    ae_rates = {}
    mis_prob = {}
    mis_err = {}
    total_err = {}
    for nr in n_regions:
        dims_to_rep = total_dims + overlap*(nr - 1)
        dims_per_region = _split_integer(dims_to_rep, nr)
        if np.all(np.array(dims_per_region) > overlap):
            distorts[nr] = np.zeros(nr)
            ae_distorts[nr] = np.zeros(nr)
            mis_err[nr] = np.zeros(nr)
            mis_prob[nr] = np.zeros(nr)
            for i, dpr_i in enumerate(dims_per_region):
                ri_pwr = total_pwr*dpr_i/dims_to_rep
                ri_units = int(np.round(total_units*dpr_i/dims_to_rep))

                out = rfm.min_mse_vec(ri_pwr, ri_units, dpr_i,
                                      lam=lambda_deviation,
                                      **kwargs)
                # out = rfm.min_mse_power(ri_pwr, ri_units, dpr_i,
                #                         local_min_max=ret_min_max,
                #                         lambda_deviation=lambda_deviation,
                #                         **kwargs)
                _, l_mse, nl_mse, nl_prob = out
                distorts[nr][i] = l_mse*dpr_i
                ae_distorts[nr][i] = l_mse
                mis_prob[nr][i] = nl_prob
                mis_err[nr][i] = nl_mse*dpr_i
            ae_nr = np.zeros((nr, nr))
            for (j, k) in it.combinations(range(nr), 2):
                ae_risk = integrate_assignment_error(
                    (n_stim,), ae_distorts[nr][j:j+1], ae_distorts[nr][k:k+1],
                    overlap, p=1)
                ae_nr[j, k] = ae_risk
                ae_nr[k, j] = ae_risk
            ae_rates[nr] = ae_nr            
        else:
            distorts[nr] = np.zeros(nr)*np.nan
            ae_rates[nr] = np.zeros((nr, nr))*np.nan
            mis_prob[nr] = np.zeros(nr)*np.nan
            mis_err[nr] = np.zeros(nr)*np.nan

        total_err[nr] = calc_te(distorts[nr], ae_rates[nr], mis_prob[nr],
                                mis_err[nr], nr, n_stim=n_stim)            
    return total_err, distorts, ae_rates, mis_prob, mis_err


def calc_te(d_l, p_ae, p_nl, d_nl, n_regions, n_stim=2):
    # local errors only
    p_local = np.product((1 - p_nl)**n_stim)

    if n_regions > 1:
        ae_risk = 0
        for (j, k) in it.combinations(range(n_regions), 2):
            ae_risk_jk = (p_ae[j, k]*(d_nl[j] + d_nl[k])
                          + (1 - p_ae[j, k])*(d_l[j] + d_l[k]))
            ae_risk = ae_risk + ae_risk_jk
    else:
        ae_risk = d_l[0]
    te = p_local*ae_risk + (1 - p_local)*np.mean(d_nl)
    return te    

def fi_tradeoff(total_units, total_dims, n_regions=(1, 2), overlap=1,
                n_stim=2, total_pwr=10, use_theory=True, ret_min_max=True,
                lambda_deviation=2, opt_kind='basinhop', **kwargs):
    distorts = {}
    ae_rates = {}
    for nr in n_regions:
        dims_to_rep = total_dims + overlap*(nr - 1)
        dims_per_region = _split_integer(dims_to_rep, nr)
        if np.all(np.array(dims_per_region) > overlap):
            distorts[nr] = np.zeros(nr)
            for i, dpr_i in enumerate(dims_per_region):
                ri_pwr = total_pwr*dpr_i/dims_to_rep
                ri_units = int(np.round(total_units*dpr_i/dims_to_rep))
                if use_theory:
                    out = rfm.max_fi_power(ri_pwr, ri_units, dpr_i,
                                           ret_min_max=ret_min_max,
                                           lambda_deviation=lambda_deviation,
                                           opt_kind=opt_kind, **kwargs)
                    fi, fi_var, pwr, w, scale = out
                    distorts[nr][i] = fi
                else:
                    distorts[nr][i] = rf_distortion(ri_units, dpr_i,
                                                    scale=ri_pwr,
                                                    **kwargs)
            ae_nr = 0
            for (j, k) in it.combinations(range(nr), 2):
                ae_risk = integrate_assignment_error((n_stim,), distorts[nr][j:j+1],
                                                     distorts[nr][k:k+1], overlap, p=1)
                ae_nr = ae_nr + ae_risk
            ae_rates[nr] = ae_nr
        else:
            distorts[nr] = np.zeros(nr)*np.nan
            ae_rates[nr] = np.nan
    return distorts, ae_rates

def get_fi_mat(s_d, inv_cov, identity=True):
    n_dims = s_d.shape[-1]
    fi_mat = np.zeros((s_d.shape[0], n_dims, n_dims))
    fi_mat[:] = np.nan
    for (i, j) in it.product(range(n_dims), repeat=2):
        if np.all(np.isnan(fi_mat[:, i, j])):
            if identity:
                nv = inv_cov[np.identity(inv_cov.shape[0], dtype=bool)]
                s1 = np.expand_dims(nv, 0)*s_d[..., j]
            else:
                s1 = np.dot(inv_cov, s_d[..., j].T).T
            fi_mat[:, i, j] = np.sum(s_d[..., i]*s1, axis=1)
            fi_mat[:, j, i] = fi_mat[:, i, j]
    mfi_mat = np.mean(fi_mat, axis=0)
    sfi_mat = np.var(fi_mat, axis=0)
    # print('m2 emp', np.mean(fi_mat[:, 0, 0]**2, axis=0))
    # print('m  emp', np.mean(fi_mat[:, 0, 0], axis=0)**2)
    # print('v  emp', np.mean(fi_mat[:, 0, 0]**2, axis=0) -
    #       np.mean(fi_mat[:, 0, 0], axis=0)**2)
    # inv_mfi_mat = np.linalg.inv(mfi_mat)
    return mfi_mat, sfi_mat

def get_rad(s_r):
    return np.sum(s_r**2, axis=1)

def rf_distortion_theory(n_units, n_dims, input_distr_type='uniform',
                         pwr=1, var=1, use_rand_rfs=True):
    if input_distr_type != 'uniform':
        raise IOError('only "uniform" input_distr_type is supported')
    if not use_rand_rfs:
        raise IOError('only use_rand_rfs is supported')
    distrs = (sts.uniform(0, 1),)*n_dims
    stim_distr = u.MultivariateUniform(n_dims, (0, 1))

    ms, ws = rfm.get_random_uniform_fill(n_units, distrs)
    orig_pwr = rfm.random_uniform_pwr(n_units, np.sqrt(ws[0]), n_dims)
    rescale = np.sqrt(pwr/orig_pwr)
    fi_theor = rfm.random_uniform_fi(n_units, np.sqrt(ws[0]), n_dims,
                                     scale=rescale, sigma_n=var)
    return np.linalg.inv(fi_theor)[0, 0]

def rf_distortion(n_units, n_dims, input_distr_type='uniform', 
                  scale=1, wid_scaling=1, baseline=0, var=1,
                  use_rand_rfs=True, n_samps=1000, wid=None):
    distr_list = (sts.uniform(0, 1),)*n_dims
    stim_distr = u.MultivariateUniform(n_dims, (0, 1))
    n_units_pd = int(np.round(n_units**(1/n_dims)))
    inv_cov = np.identity(n_units)/var
    if use_rand_rfs:
        ms, ws = rfm.get_random_uniform_fill(n_units, distr_list, wid=wid)
    else:
        ms, ws = rfm.get_output_func_distribution_shapes(n_units_pd, distr_list,
                                                         wid_scaling=wid_scaling)
    rf, drf = rfm.make_gaussian_vector_rf(ms, ws, scale, baseline,
                                          titrate_pwr=stim_distr)
    samps = stim_distr.rvs(n_samps)
    s_d = drf(samps)
    fi, inv_fi = get_fi_mat(s_d, inv_cov)
    pwr = np.mean(get_rad(rf(samps)))
    distortion = np.mean(inv_fi[np.identity(n_dims, dtype=bool)])
    return fi[0, 0], pwr, distortion

def compute_fi(ms, ws, scale, stim_distr, baseline=0, n_samps=10000,
               **kwargs):
    rf, drf = rfm.make_gaussian_vector_rf(ms, ws, scale, baseline,
                                          titrate_pwr=stim_distr)
    samps = stim_distr.rvs(n_samps)
    s_d = drf(samps)
    inv_cov = np.identity(ms.shape[0])
    fi, inv_fi = get_fi_mat(s_d, inv_cov)
    pwr = np.mean(get_rad(rf(samps)))
    distortion = np.mean(inv_fi[np.identity(ms.shape[1], dtype=bool)])
    return fi[0, 0], pwr, distortion


def fixed_distance_errors(d, dx, dy, n_ests=1000, p=100, c=1, boot=False):
    err = np.zeros(n_ests)
    pop_ests = np.array((dx, dy))
    for i in range(n_ests):
        est = simulate_estimates(p, 2, pop_ests, c=c, fixed_dist=d)
        ad, base = assign_estimates_2pop(est)
        err[i] = np.any(ad < base)
    if boot:
        err = u.bootstrap_on_axis(err, u.mean_axis0, axis=0, n=n_ests)
    return err

def fixed_distance_ds_dxdy(ds, dxs, dys, n_ests=1000, p=100, c=1, boot=True):
    errs = np.zeros((len(ds), len(dxs), n_ests))
    for i, d in enumerate(ds):
        for j, dx_j in enumerate(dxs):
            dy_j = dys[j]
            errs[i, j] = fixed_distance_errors(d, dx_j, dy_j, n_ests=n_ests,
                                               p=p, c=c, boot=boot)
    return errs

def simulate_estimates(p, s, pop_ests, c=1, constrain=False, fixed_dist=None):
    if fixed_dist is not None:
        pts = np.zeros((2, c, 1))
        d_dim = np.sqrt((fixed_dist**2)/c)
        pts[1] = d_dim
    else:
        pts = np.random.uniform(0, p, (s, c, 1))
    est_ns = np.random.normal(0, np.sqrt(pop_ests), (s, c, len(pop_ests)))
    ests = pts + est_ns
    if constrain:
        ests[ests < 0] = 0
        ests[ests > p] = p
    return ests

def calculate_distance(ests):
    cs = it.combinations(range(ests.shape[2]), 2)
    dist = 0
    for c in cs:
        sub_est = np.sum((ests[..., c[0]] - ests[..., c[1]])**2, axis=1)
        dist = dist + np.sum(sub_est)
    return dist

def assign_estimates_2pop(ests):
    s, c, pops = ests.shape
    assert pops == 2
    baseline = calculate_distance(ests)
    perturbs = list(it.combinations(range(s), 2))
    assign_dists = np.zeros(len(perturbs))
    shuff_ests = np.zeros_like(ests)
    for i, p in enumerate(perturbs):
        shuff_ests[:, :, :] = ests[:, :, :]
        shuff_ests[p[1], :, 0] = ests[p[0], :, 0]
        shuff_ests[p[0], :, 0] = ests[p[1], :, 0]
        assign_dists[i] = calculate_distance(shuff_ests)
    return assign_dists, baseline

def estimate_ae_full(p, n_stim, delta, ds, c, n_ests=1000):
    pop_ests = dxdy_from_dsdelt(ds, delta)
    return estimate_assignment_error(p, n_stim, pop_ests, c=c, n=n_ests)

def estimate_assignment_error(p, s, pop_ests, c=1, n=500):
    err = np.zeros(n)
    for i in range(n):
        est = simulate_estimates(p, s, pop_ests, c=c)
        ad, base = assign_estimates_2pop(est)
        err[i] = np.any(ad < base)
    return err

def estimate_ae_sr_range(s, srs, n_pops=2, n=500, p=100, boot=True, c=1):
    errs = np.zeros((n, len(srs)))
    for i, sr in enumerate(srs):
        pop_est = (p/sr)**2
        pop_ests = (pop_est,)*n_pops
        errs[:, i] = estimate_assignment_error(p, s, pop_ests, n=n, c=c)
    if boot:
        errs = u.bootstrap_on_axis(errs, u.mean_axis0, axis=0, n=n)
    return errs

def estimate_ae_sr_s_ranges(esses, srs, n_pops=2, n=500, p=100, c=1, boot=True):
    errs = np.zeros((len(esses), n, len(srs)))
    for i, s in enumerate(esses):
        errs[i] = estimate_ae_sr_range(s, srs, n_pops=n_pops, n=n, p=p, c=c,
                                       boot=boot)
    return errs

def error_approx_sr_s_ranges(esses, srs, n_pops=2):
    errs = np.zeros((len(esses), len(srs)))
    for i, s in enumerate(esses):
        errs[i] = error_approx_sr_range(s, srs, n_pops=n_pops)
    return errs

def error_approx_sr_range(s, srs, n_pops=2, p=100, pop_est=2):
    errs = np.zeros(len(srs))
    for i, sr in enumerate(srs):
        pop_est = (p/sr)**2
        pop_ests = (pop_est,)*n_pops
        errs[i] = error_approx(p, s, pop_ests)
    return errs

def error_approx(p, s, pop_ests, integ_step=.0001, eps=0):
    integ_step = integ_step*p
    eps = eps*p
    factor = ss.comb(s, 2)
    var = np.sum(pop_ests)
    int_func = lambda x: 2*((p - x)/(p**2))*sts.norm(x, np.sqrt(var)).cdf(0)
    pe = u.euler_integrate(int_func, eps, p, integ_step)
    return factor*pe

def error_approx_further(esses, srs, p=100, use_second=False):
    errs = np.zeros((len(esses), len(srs)))
    pop_ests = (p/srs)**2
    for i, s in enumerate(esses):
        first_term = 2*np.sqrt(2*pop_ests)/(p*np.sqrt(np.pi))
        if use_second:
            second_term = -8*pop_ests/(p**2)
        else:
            second_term = 0
        errs[i] = ss.comb(s, 2)*(first_term + second_term)
    errs[errs > 1] = 1
    return errs

def distortion_error_approx(esses, d1, d2, p=100):
    errs = np.zeros((len(esses), len(d1)))
    first_term = np.sqrt(2*(d1 + d2))/(p*np.sqrt(np.pi))
    second_term = -(d1 + d2)/(2*(p**2))
    for i, s in enumerate(esses):
        errs[i] = ss.comb(s, 2)*(first_term + second_term)
    errs[errs > 1] = 1
    return errs

def feature_info_consumption(esses, distortion, d=1, p=100):
    info = np.log(p/np.sqrt(2*np.pi*distortion))
    infos = np.ones((len(esses), len(distortion)))
    for i, s in enumerate(esses):
        infos[i] = d*s*info
    return info

def feature_redundant_info(esses, d1, d2, overlapping_d=1, p=100):
    redundancy = np.log(p/np.sqrt(2*np.pi*(d1 + d2)))
    redunds = np.ones((len(esses), len(d1)))
    for i, s in enumerate(esses):
        redunds[i] = overlapping_d*s*redundancy
    return redunds

def feature_wasted_info(esses, d1, d2, distortion, overlapping_d=1, p=100):
    waste = np.log(p*np.sqrt(d1 - distortion)/(np.sqrt(2*np.pi)*d1))
    wastes = np.zeros((len(esses), len(d1)))
    for i, s in enumerate(esses):
        wastes[i] = overlapping_d*s*waste
    return wastes                   

def constrained_distortion_info(esses, distortion, p=100, diff_ds=1000,
                                eps=.1, d_mult=10):
    d1 = np.linspace(distortion + eps, d_mult*distortion, diff_ds)
    d2 = (d1*distortion)/(d1 - distortion)
    redundancy = feature_redundant_info(esses, d1, d2, p=p)
    info1 = feature_info_consumption(esses, d1, p=p)
    info2 = feature_info_consumption(esses, d2, p=p)
    pe = distortion_error_approx(esses, d1, d2, p=p)
    return (info1, d1), (info2, d2), redundancy, pe
                        
def minimal_error_redund(esses, distortion, p=100, diff_ds=1000,
                         eps=.1, d_mult=10, lam_start=0, lam_end=.125,
                         lam_n=1000):
    t1, t2, redund, pe = constrained_distortion_info(esses, distortion, p=p,
                                                     diff_ds=diff_ds, eps=eps,
                                                     d_mult=d_mult)
    i1, d1 = t1
    i2, d2 = t2
    lams = np.linspace(lam_start, lam_end, lam_n)
    opt_d1 = np.zeros((len(esses), lam_n))
    for i, s in enumerate(esses):
        for j, l in enumerate(lams):
            lagr = pe[i] + l*redund[i]
            opt_d1[i, j] = d1[np.argmin(lagr)]
    return opt_d1, lams

def line_picking_clt(x, overlapping_d):
    mu = np.sqrt(overlapping_d*1/6)
    var = 7/120.
    p = sts.norm(mu, np.sqrt(var)).pdf(x)
    return p

def line_picking_cube(x):
    """
    Taken from: http://mathworld.wolfram.com/CubeLinePicking.html
    """
    def _arcsec(y):
        v = np.arccos(1/y)
        return v
    
    def _arccsc(y):
        v = np.arcsin(1/y)
        return v

    l1 = lambda x: -(x**2)*((x - 8)*(x**2) + np.pi*(6*x - 4))
    l2 = lambda x:  2*x*(((x**2) - 8*np.sqrt((x**2) - 1) + 3)*(x**2)
                         - 4*np.sqrt((x**2) - 1) + 12*(x**2)*_arcsec(x)
                         + np.pi*(3 - 4*x) - .5)
    l3 = lambda x: x*((1 + (x**2))*(6*np.pi + 8*np.sqrt((x**2) - 2)
                                    - 5 - (x**2))
                      - 16*x*_arccsc(np.sqrt(2 - 2*(x**-2)))
                      + 16*x*np.arctan(x*np.sqrt((x**2) - 2))
                      - 24*((x**2) + 1)*np.arctan(np.sqrt((x**2) - 2)))
    conds = (x <= 1, np.logical_and(x > 1, x <= np.sqrt(2)),
             x > np.sqrt(2))
    funcs = (l1, l2, l3)
    p = np.piecewise(x, conds, funcs)
    return p

def line_picking_square(x):
    """
    Taken from: http://mathworld.wolfram.com/SquareLinePicking.html
    """
    l1 = lambda x: 2*x*((x**2) - 4*x + np.pi)
    l2 = lambda x: 2*x*(4*np.sqrt((x**2) - 1) - ((x**2) + 2 - np.pi)
                        - 4*np.arctan(np.sqrt((x**2) - 1)))
    conds = (x <= 1, x > 1)
    funcs = (l1, l2)
    p = np.piecewise(x, conds, funcs)
    return p

def line_picking_line(x):
    p = 2*(1 - x)
    return p

def compute_ds(b, k, c, delt, n_stim, s=100, repl_nan=True,
               source_distrib='gaussian'):
    if source_distrib.lower() not in ('gaussian', 'uniform'):
        raise Exception('this function only works for Gaussian and Uniform '
                        'sources')
    """ Gaussian is N(0, s^2) """
    """ Uniform goes from 0 to s """
    ft = ((1 - delt**2)/4)**(c/(k + c))
    if source_distrib == 'gaussian':
        st = s**2
    elif source_distrib == 'uniform':
        st = (s**2)/(2*np.pi)
    tt = np.exp(-2*b/(n_stim*(k + c)))
    d = ft*st*tt
    # print('dist_contr',  s**2, 2*d/(1 - delt))
    # print('deriv_constr', b, k*n_stim*np.log(4/(1 - delt**2))/2)
    if repl_nan and len(np.array(d).shape) > 0:
        with warnings.catch_warnings(record=True) as w:
            mask = 4*d < (1 - delt)*s**2
        d[np.logical_not(mask)] = np.nan
    return d

def rdb_gaussian(d, s):
    b = .5*np.log(s/d)
    return b

def rdb_uniform(d, s):
    b = .5*np.log((s**2)/(2*np.pi*d))
    return b

def rdb(d, s, source_distrib='uniform'):
    rdb_dict = {'uniform':rdb_uniform, 'gaussian':rdb_gaussian}
    f = rdb_dict[source_distrib]
    return f(d, s)

def dxdy_from_dsdelt(ds, delt):
    dx = 2*ds/(1 + delt)
    dy = 2*ds/(1 - delt)
    return dx, dy

def _get_ae_ev(bit, k, c_xy, n_stim, s, delt, source_distrib='uniform'):
    delt = delt[0]
    ds = compute_ds(bit, k, c_xy, delt, n_stim, s=s,
                    source_distrib=source_distrib)
    dx, dy = dxdy_from_dsdelt(ds, delt)
    dx = np.array([dx])
    dy = np.array([dy])
    aes = integrate_assignment_error((n_stim,), dx, dy, c_xy,
                                     p=s)
    return aes[0, 0], ds

def _ae_ev_loss(*args, target_ae=.001, **kwargs):
    aes, ds = _get_ae_ev(*args, **kwargs)
    targ = float(100000000*(aes - target_ae)**2 + ds**2)
    print(aes, target_ae, targ)
    return targ

def ae_ev_bits(bits, k, s, n_stim, c_xys, delts,
               source_distrib='uniform', compute_redund=False):
    aes = np.zeros((len(c_xys), len(delts), len(bits)))
    evs = np.zeros_like(aes)
    redund = np.zeros_like(aes)
    n_stim = (n_stim,)
    for i, c_xy in enumerate(c_xys):
        for j, delt in enumerate(delts):
            for l, bit in enumerate(bits):
                ds = compute_ds(bit, k, c_xy, delt, n_stim[0], s=s,
                                source_distrib=source_distrib)
                dx, dy = dxdy_from_dsdelt(ds, delt)
                dx = np.array([dx])
                dy = np.array([dy])
                aes[i, j, l] = integrate_assignment_error(n_stim, dx, dy, c_xy,
                                                          p=s)
                evs[i, j, l] = ds
                redund[i, j, l] = feature_redundant_info(n_stim, dx, dy, c_xy,
                                                         p=s)
    if compute_redund:
        out = aes, evs, redund
    else:
        out = aes, evs
    return out

def mse_weighting(k, c, s, source_distrib='uniform', n_emp=10000):
    d = k - c
    if source_distrib == 'gaussian':
        num = 2*sps.gamma((d + 1)/2)
        denom = sps.gamma(d/2)
        avg_dist = s*(num/denom)**2
    elif source_distrib == 'uniform':
        pts = s*np.random.rand(2, d, n_emp)
        d_rs = np.sqrt(np.sum(np.diff(pts, axis=0)[0]**2, axis=0))
        avg_dist = np.mean(d_rs)**2
    return avg_dist

def integrate_assignment_error(esses, d1, d2, overlapping_d, p=100):
    if d1 is None and d2 is not None:
        d1 = d2
    elif d2 is None and d1 is not None:
        d2 = d1        
    d1, d2 = d1/(p**2), d2/(p**2)
    p = 1
    integ_end = np.sqrt(overlapping_d)
    if overlapping_d == 1:
        dist_pdf = line_picking_line
    elif overlapping_d == 2:
        dist_pdf = line_picking_square
    elif overlapping_d == 3:
        dist_pdf = line_picking_cube
    else:
        # if overlapping_d < 10:
        #     print('Using CLT approximation for a small overlap, probably'
        #           ' will not be accurate. {} < 10'.format(overlapping_d))
        dist_pdf = lambda x: line_picking_clt(x, overlapping_d)
    if overlapping_d <= 3 or overlapping_d > 10:
        ae = ae_integ(esses, d1, d2, p=p, integ_start=0, integ_end=integ_end,
                      dist_pdf=dist_pdf)
    else:
        ae = ae_sample(esses, d1, d2, overlapping_d, p=p)
    return ae

def distance_error_rate(dists, delta_d, overall_d):
    d1s, d2s = d_func(delta_d, overall_d)
    dist_err = np.zeros((len(d1s), len(dists)))
    for i, dist in enumerate(dists):
        for j, d1 in enumerate(d1s):
            d2 = d2s[j]
            ep1 = sts.norm(dist, np.sqrt(2*d1)).cdf(0)
            ep2 = sts.norm(dist, np.sqrt(2*d2)).cdf(0)
            dist_err[j, i] = ep1 + ep2 - 2*ep1*ep2
            # dist_err[j, i] = sts.norm(dist, np.sqrt(d1 + d2)).cdf(0)
    return dist_err

def distance_mse(dists, delta_d, overall_d, s=100):
    dist_err = distance_error_rate(dists, delta_d, overall_d)
    d1s, d2s = d_func(delta_d, overall_d)
    distortion = np.zeros_like(dist_err)
    low_bound = overall_d
    high_bound = (s**2)/6
    right_bound = np.sqrt(4*overall_d)
    for i, d1 in enumerate(d1s):
        for j, dist in enumerate(dists):
            pe = dist_err[i, j]
            distortion[i, j] = (1 - pe)*low_bound + pe*high_bound
    return dist_err, distortion, low_bound, high_bound, right_bound

def ae_sample(esses, d1, d2, overlapping_d, p=1, n_samples=10**7, **kwargs):
    pes = np.zeros_like(d1)
    for i, d1_i in enumerate(d1):
        d2_i = d2[i]
        pts = np.random.default_rng().uniform(size=(n_samples*2, overlapping_d))
        xs = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))[::2]
        v2 = sts.norm(xs, np.sqrt(2*d1_i)).cdf(0)
        v3 = sts.norm(xs, np.sqrt(2*d2_i)).cdf(0)
        v = (v2 + v3 - 2*v2*v3)
    pes[i] =  np.mean(v)
    errs = np.zeros((len(esses), len(d1)))
    for i, s in enumerate(esses):
        errs[i] = ss.comb(s, 2)*pes
    return errs        

def ae_integ(esses, d1, d2, p=1, integ_start=0, integ_end=None, dist_pdf=None,
             err_thr=.01, assert_=False):
    pes = np.zeros_like(d1)
    for i, d1_i in enumerate(d1):
        d2_i = d2[i]
        def _f(x):
            v1 = dist_pdf(x)
            # v2 = sts.norm(x, np.sqrt(d1_i + d2_i)).cdf(0)
            v2 = sts.norm(x, np.sqrt(2*d1_i)).cdf(0)
            v3 = sts.norm(x, np.sqrt(2*d2_i)).cdf(0)
            v = v1*(v2 + v3 - 2*v2*v3)
            return v
           
        pes[i], err = sin.quad(_f, integ_start, integ_end)
        if assert_:
            assert err < err_thr
        elif err > err_thr:
            pes[i] = np.nan
    errs = np.zeros((len(esses), len(d1)))
    for i, s in enumerate(esses):
        errs[i] = ss.comb(s, 2)*pes
    return errs

def distortion_func(bits, features, objs, overlaps, d1=None, p=100,
                    d_cand=None):
    if d1 is not None and np.all(d1 > 0):
        d = _distortion_func_numerical(bits, features, objs, overlaps, d1, p=p,
                                       d_cand=d_cand)
    else:
        d = _distortion_func_analytical(bits, features, objs, overlaps, p=p)
    return d

def _distortion_func_analytical(bits, features, objs, overlaps, p=100):
    t1 = (p**2)/(2*np.pi)
    t2 = (1/2)**(2*overlaps/(features + overlaps))
    t3 = np.exp(-2*bits/(objs*(features + overlaps)))
    d = t1*t2*t3
    return d

def _distortion_func_numerical(bits, features, objs, overlaps, d1, p=100,
                               d_cand=None, eps=.001, distortion_eps=1e-10):
    if d_cand is None:
        d_cand = _distortion_func_analytical(bits, features, objs, overlaps,
                                             p=p)
    if np.array(d1).shape[0] > 1:
        d_cand = np.ones(d1.shape)*d_cand

    def _alt_targ_func(d):
        t1 = features*np.log(p/np.sqrt(2*np.pi*d))
        if np.any((d1 - d) < 0):
            print(d1, d)
            print(d1 - d)
        t2 = overlaps*np.log(p*(d1 - d)/(np.sqrt(2*np.pi*d)*d1**2))
        t3 = bits/objs
        out = np.sum(np.abs(t1 + t2 - t3))
        return out                     
        
    def _log_targ_func(d):
        t1 = (overlaps + features)*np.log(p/np.sqrt(2*np.pi))
        t2 = (features/2)*np.log(1/d)
        t3 = (overlaps/2)*np.log((d1 - d)/(d1**2))
        t4 = bits/objs
        out = np.sum(np.abs(t1 + t2 + t3 - t4))
        return out
        
    def _targ_func(d):
        t1 = (p/np.sqrt(2*np.pi))**(overlaps + features)
        t2 = (1/d)**(features/2)
        t3 = ((d1 - d)/(d1**2))**(overlaps/2)
        t4 = np.exp(bits/objs)
        out = np.sum(np.abs(t1*t2*t3 - t4))
        return out
    
    lower = np.ones_like(d_cand)*distortion_eps
    upper = d1
    res = sio.minimize(_log_targ_func, d_cand, bounds=list(zip(lower, upper)))
    d = res.x
    try:
        assert np.all(_log_targ_func(d) < eps)
    except AssertionError:
        print('minimization error')
        print('d', d)
        print('d1', d1)
        print(overlaps, features, bits, objs)
        print('orig', _targ_func(d))
        print('log ', _log_targ_func(d))
        print('alt ', _alt_targ_func(d))
    return d

def basic_d_func(d1, overall_d):
    d2 = d1*overall_d/(d1 - overall_d)
    return d2

def d_func(delta_d, overall_d):
    if delta_d is None:
        d2 = [overall_d*2]
        d1 = [overall_d*2]
    else:
        d1 = overall_d*2 + np.array(delta_d)
        d2 = d1*overall_d/(d1 - overall_d)
    return np.array(d2), np.array(d1)

def noncoded_weighting(k, c, p):
    weight = (k - c)*(p**2)/12
    return np.array(weight)

def weighted_errors_bits_feats(bits_list, features, objs_list, overlaps_list,
                               d1_n=None, p=100, lam_range=None, lam_beg=0,
                               lam_end_mult=100, lam_n=1000, lam_func=None):
    if d1_n is None:
        d1_len = 1
    else:
        d1_len = d1_n
    if lam_func is not None:
        lam_len = len(overlaps_list)
    elif lam_range is not None:
        lam_len = len(lam_range)
    else:
        lam_len = lam_n
    arr_shape = (len(features), len(bits_list), len(objs_list),
                 len(overlaps_list), d1_len)
    totals = np.zeros(arr_shape + (lam_len,))
    ae = np.zeros(arr_shape)
    local_ds = np.zeros_like(ae)
    d1s = np.zeros_like(ae)
    for i, f in enumerate(features):
        if lam_func is not None:
            lam_range = lam_func(f, overlaps_list, p)
        out = weighted_errors_bits(bits_list, f, objs_list, overlaps_list,
                                   d1_n=d1_n, p=p, lam_range=lam_range,
                                   lam_beg=lam_beg, lam_end_mult=lam_end_mult,
                                   lam_n=lam_n)
        totals[i], local_ds[i], ae[i], d1s[i] = out
    return totals, local_ds, ae, d1s    

def weighted_errors_bits(bits_list, features, objs_list, overlaps_list,
                         d1_n=None, p=100, lam_range=None, lam_beg=0,
                         lam_end_mult=100, lam_n=1000):
    if d1_n is None:
        d1_size = 1
    else:
        d1_size = d1_n
    if lam_range is None:
        lam_size = lam_n
    else:
        lam_size = len(lam_range)
    totals = np.zeros((len(bits_list), len(objs_list), len(overlaps_list),
                       d1_size, lam_size))
    assignment_errs = np.zeros((len(bits_list), len(objs_list),
                                len(overlaps_list), d1_size))
    local_ds = np.zeros_like(assignment_errs)
    d1s = np.zeros_like(assignment_errs)
    for i, b in enumerate(bits_list):
        for j, o in enumerate(objs_list):
            out = weighted_errors_lambda(b, features, o, overlaps_list,
                                         d1_n=d1_n, p=p, lam_range=lam_range,
                                         lam_beg=lam_beg,
                                         lam_end_mult=lam_end_mult,
                                         lam_n=lam_n)
            totals[i, j], local_ds[i, j], assignment_errs[i, j], d1s[i, j] = out
    return totals, local_ds, assignment_errs, d1s

def weighted_errors_lambda(bits, features, objs, overlaps_list, d1_n=None,
                           p=100, lam_range=None, lam_beg=0, lam_end_mult=100,
                           lam_n=1000, d1_fact=50):
    if d1_n is None:
        d1_size = 1
        d1_n = 1
    else:
        d1_size = d1_n
    if lam_range is None:
        lam_range = np.linspace(lam_beg, lam_end_mult*p*features*objs, lam_n)
    totals = np.zeros((len(overlaps_list), d1_size, len(lam_range)))
    local_d = np.zeros((len(overlaps_list), d1_size))
    assignment_err = np.zeros_like(local_d)
    d1s = np.zeros_like(local_d)
    trans_matrix = np.ones((len(lam_range), d1_size))
    lam_range = lam_range.reshape((-1, 1))*trans_matrix
    for i, overlaps in enumerate(overlaps_list):
        obj_list = (objs,)
        initial_d = distortion_func(bits, features, objs, overlaps, p=p)
        delta_d = np.linspace(0, d1_fact*initial_d, d1_n)
        d2_i, d1_i = d_func(delta_d, initial_d)
        if d1_n == 1 and np.all(delta_d == 0):
            overall_d = distortion_func(bits, features, objs, overlaps, d1=None,
                                        p=p)
        else:
            overall_d = distortion_func(bits, features, objs, overlaps, d1=d1_i,
                                        p=p, d_cand=initial_d)
        assignment_d = integrate_assignment_error(obj_list, d1_i, d2_i,
                                                  overlaps, p=p)
        local_d[i] = overall_d
        assignment_err[i] = assignment_d
        w_sum = overall_d.reshape((1, -1)) + lam_range*assignment_d
        totals[i] = w_sum.T
        d1s[i] = d1_i
    return totals, local_d, assignment_err, d1s

def minimum_weighted_error_lambda(bits, features, objs, p=100, lam_range=None,
                                  lam_beg=0, lam_end_mult=3, lam_n=5000,
                                  d1_n=5000, d1_mult=100):
    d1_list = np.linspace(0, d1_mult*p, d1_n)
    overlaps = np.arange(1, features + 1, 1)
    if lam_range is None:
        lam_range = np.linspace(lam_beg, lam_end_mult*p*features*objs, lam_n)
    out = weighted_errors_lambda(bits, features, objs, overlaps,
                                 d1_list=d1_list, lam_range=lam_range, p=p)
    total, local_d, ae, d1s = out
    flat_shape = (len(overlaps)*len(d1_list), len(lam_range))
    o_i, d1_i = np.unravel_index(np.argmin(total.reshape(flat_shape), axis=0),
                                 (len(overlaps), len(d1_list)))
    opt_d1 = d1_list[d1_i]
    opt_over = overlaps[o_i]
    min_total = total[o_i, d1_i, range(len(lam_range))]
    return opt_d1, opt_over, min_total, lam_range, local_d, ae, total
