
import numpy as np
import itertools as it
import scipy.stats as sts 
import pickle
import pystan as ps

import general.rf_models as rfm
import mixedselectivity_theory.nms_continuous as nmc


def generate_noisy_samples(stims, cents, wids, scales, cov, n_samps=100):
    func, _ = rfm.make_gaussian_vector_rf(cents, wids, scales, 0)
    resp_set = np.zeros((stims.shape[0], n_samps, cov.shape[0]))
    for i, stim_set in enumerate(stims):
        for j, stim in enumerate(stim_set):
            if j == 0:
                resp = func(stim)
            else:
                resp = resp + func(stim)
        noise = np.random.multivariate_normal(np.zeros(cov.shape[0]),
                                              cov, n_samps)
        noisy_resp = resp + noise
        resp_set[i] = noisy_resp
    return resp_set, func

gt_model = 'assignment/stan_models/gaussian_tuning.pkl'
def estimate_posterior_series(data, dists, rf_params, model_path=gt_model,
                              verbose=False, **stan_params):
    collection = {}
    center_pt = ((rf_params['cents'][-1] - rf_params['cents'][0])/2
                 + rf_params['cents'][0])
    for i, d in enumerate(dists):
        stim = np.zeros((1, 2))
        stim[0, 0] = center_pt - d/2
        stim[0, 1] = center_pt + d/2
        out = generate_noisy_samples(stim, rf_params['cents'],
                                     rf_params['wids'], rf_params['scales'],
                                     rf_params['cov_mat'], n_samps=data['N'])
        resps = out[0][0]
        if verbose:
            print('fitting model {}/{}'.format(i + 1, len(dists)))
        run_dict = {}
        run_dict.update(data)
        run_dict.update(rf_params)
        run_dict['samps'] = resps
        fit = estimate_posterior_stan(run_dict, model_path=model_path,
                                      **stan_params)
        run_dict['stim'] = stim
        diags = ps.diagnostics.check_hmc_diagnostics(fit)
        collection[d] = (fit, run_dict, diags)
    return collection

def estimate_posterior_stan(data, model_path=gt_model, **stan_params):
    sm = pickle.load(open(model_path, 'rb'))
    fit = sm.sampling(data=data, **stan_params)
    return fit

def generate_1d_rfs(space_size, rf_spacing, rf_size, mult=1):
    rf_cents = np.arange(0, space_size + rf_spacing, rf_spacing)
    func, dfunc = rfm.make_gaussian_vector_rf(rf_cents, rf_size, mult, 0)
    return func, dfunc, rf_cents

def generate_2d_rfs(space_sizes, rf_spacings, rf_sizes, mult=1):
    rf_cents1 = np.arange(0, space_sizes[0] + rf_spacings[0], rf_spacings[0])
    rf_cents2 = np.arange(0, space_sizes[1] + rf_spacings[1], rf_spacings[1])
    cents = np.array(list(it.product(rf_cents1, rf_cents2)))
    func, dfunc = rfm.make_gaussian_vector_rf(cents, np.array(rf_sizes),
                                              mult, 0)
    return func, dfunc, cents

def get_local_manifold_rfs(trs, pt, delt=.1):
    diff = trs(pt) - trs(pt + delt)
    norm_diff = diff/delt
    return norm_diff

def estimate_error(local, cov):
    est_var = np.dot(local, np.dot(cov, local))
    return est_var

def estimate_est_cov(local1, local2, cov):
    est_cov = np.dot(local1, np.dot(cov, local2))
    return est_cov

def error_theory(dtrs, n, dists, noise_cov, inv_cov=None, swap_ax=0):
    pairs = _make_pairs(dists, n)
    var = np.zeros(len(dists))
    cov = np.zeros_like(var)
    swap = np.zeros_like(var)
    fim = np.zeros((len(pairs), 2*pairs.shape[-1], 2*pairs.shape[-1]))
    inv_fim = np.zeros_like(fim)
    for i, p in enumerate(pairs):
        p1, p2 = p
        fim[i] = derive_fim(dtrs, p1, p2, noise_cov, inv_cov=inv_cov)
        inv_fim[i] = np.linalg.inv(fim[i])
        var[i] = inv_fim[i, swap_ax, swap_ax]
        cov[i] = inv_fim[i, swap_ax, swap_ax + len(p1)]
        swap[i] = swap_rate(var[i], cov[i], p2[swap_ax] - p1[swap_ax])
    return var, cov, swap, fim, inv_fim

def derive_fim(dtrs, p1, p2, noise_cov, inv_cov=None):
    df1 = dtrs(p1.reshape((1, -1)))
    df2 = dtrs(p2.reshape((1, -1)))
    deriv_matrix = np.concatenate((df1, df2), axis=2)
    if inv_cov is None:
        inv_cov = np.linalg.inv(noise_cov)
    fim = np.zeros((deriv_matrix.shape[2],)*2)
    ax_combos = it.combinations_with_replacement(range(deriv_matrix.shape[2]),
                                                 2)
    for d1, d2 in ax_combos:
        deriv_d1 = deriv_matrix[0, :, d1]
        deriv_d2 = deriv_matrix[0, :, d2]
        fim_d1d2 = np.dot(deriv_d1, np.dot(inv_cov, deriv_d2.T))
        fim[d1, d2] = fim_d1d2
        fim[d2, d1] = fim_d1d2
    return fim

def swap_rate(var, cov, dist):
    distr = sts.norm(dist, np.sqrt(2*var - 2*cov))
    p = distr.cdf(0)
    return p

def distance_var(pt1, pt2, cov, n=5000):
    samps = sts.multivariate_normal([pt1, pt2], cov).rvs(n)
    dists = np.diff(samps, axis=1)
    dist_var = np.var(dists)
    return dist_var

def make_distance_cov_func(tau, axis=0):
    return lambda x, y: np.exp(-((x[axis] - y[axis])**2)/tau)

def make_pair_covs(dtrs, pt, dists, corr_scale, var_scale, distance_cov=None,
                   cents=None, info_lim_scale=20):
    enc_pt1 = dtrs(pt)[0, : ,0]
    n = enc_pt1.shape[0]
    all_covs = np.zeros((len(dists), n, n))
    all_invs = np.zeros_like(all_covs)
    if distance_cov is not None:
        corr_func = make_distance_cov_func(distance_cov)
        cents = np.expand_dims(cents, 1)
        dist_cov = make_covariance_matrix(cents, var_scale, corr_scale,
                                          corr_func=corr_func, inv=False)
        np.fill_diagonal(dist_cov, 0)
    for i, d in enumerate(dists):
        cov = np.identity(n)*var_scale
        enc_pt2 = dtrs(pt + d)[0, :, 0]
        sum_pts = enc_pt1 + enc_pt2
        sum_pts_unit = sum_pts/np.sqrt(np.sum(sum_pts**2))
        outer = np.outer(sum_pts_unit, sum_pts_unit)
        all_covs[i] = cov + var_scale*corr_scale*outer*info_lim_scale
        if distance_cov is not None:
            all_covs[i] = all_covs[i] + dist_cov
        np.fill_diagonal(all_covs[i], var_scale)
        all_invs[i] = np.linalg.inv(all_covs[i])
    return all_covs, all_invs

def make_pair_fims(dtrs, pt, dists, corr_scale, var_scale, **kwargs):
    all_covs, all_invs = make_pair_covs(dtrs, pt, dists, corr_scale, var_scale,
                                        **kwargs)
    pt = np.array([pt])
    all_fims = np.zeros((len(dists), 2*len(pt), 2*len(pt)))
    all_inv_fims = np.zeros_like(all_fims)
    for i, dist in enumerate(dists):
        fim = derive_fim(dtrs, pt, pt + dist, all_covs[i], inv_cov=all_invs[i])
        all_fims[i] = fim
        all_inv_fims[i] = np.linalg.inv(fim)
    return all_fims, all_inv_fims, all_covs

def make_info_limiting_cov(dtrs, pts, corr_scale, var_scale, axis=0,
                           dtrs_eps=0):
    enc_pts1 = dtrs(pts)
    eps_pts = pts
    eps_pts[:, axis] = eps_pts[:, axis] + dtrs_eps
    enc_pts2 = dtrs(eps_pts)
    n = enc_pts1.shape[1]
    cov = np.zeros((n, n))
    for i, ep in enumerate(enc_pts1):
        comp = np.outer(ep[:, axis], enc_pts2[i][:, axis])
        cov = cov + comp
    diag = np.diagonal(cov)
    s = np.mean(diag)
    mult_factor = var_scale*corr_scale/s
    cov = mult_factor*cov
    am = np.identity(n)*(var_scale - np.diagonal(cov))
    cov = cov + am
    inv_cov = np.linalg.inv(cov)
    return cov, inv_cov

def make_covariance_matrix(cents, var, corr_scale, corr_func=None, inv=True):
    size = len(cents)
    cov = np.identity(size)*var
    if corr_func is None:
        cov[cov == 0] = corr_scale*var
    else:
        for c0, c1 in it.combinations(range(size), 2):
            cij = var*corr_scale*corr_func(cents[c0], cents[c1])
            cov[c0, c1] = cij
            cov[c1, c0] = cij
    if inv:
        inv_cov = np.linalg.inv(cov)
        ret = cov, inv_cov
    else:
        ret = cov
    return ret

def _make_pairs(dists, ns):
    dists = np.array(dists)
    ns = np.array(ns)
    pairs = np.ones(dists.shape +  (2,))*(ns/2 - np.max(dists, axis=0)/2)
    pairs[:, 1] = pairs[:, 1] + dists
    return pairs

def estimate_encoding_swap(rf_size, rf_spacing, noise_var, noise_corr, dists,
                           noise_samps=1000, rf_mult=10, space_size=None,
                           buff=None, buff_mult=4, corr_func=None,
                           distortion=nmc.mse_distortion,
                           give_real=True, basin_hop=True, parallel=False,
                           oo=False, n_hops=100, c=1, space_mult=2):
    if buff is None:
        buff = rf_size*buff_mult
    if space_size is None:
        space_size = np.ceil(max(dists)) + space_mult*buff
        space_size = int(space_size)
    trs, _, cents = generate_1d_rfs(space_size, rf_spacing, rf_size, mult=rf_mult)
    d_pairs = _make_pairs(dists, space_size)
    all_pts = d_pairs.reshape((2*len(dists), 1))
    trs_pts = trs(all_pts)
    noise_mean = np.zeros(len(cents))
    noise_cov = make_covariance_matrix(len(cents), noise_var, noise_corr,
                                       corr_func=corr_func)
    noise_distrib = sts.multivariate_normal(noise_mean, noise_cov)
    noise = noise_distrib.rvs((len(dists), noise_samps))
    noise = np.expand_dims(noise, axis=1)
    trs_pts_samp = np.repeat(trs_pts, noise_samps, axis=0)
    trs_pts_struct = trs_pts_samp.reshape((len(dists), 2, noise_samps,
                                           len(cents)))
    noise_pts = trs_pts_struct + noise
    noise_pts_flat = noise_pts.reshape((len(dists)*2*noise_samps, -1))
    all_pts_ns = np.repeat(all_pts, noise_samps, axis=0)
    if give_real:
        give_pts = all_pts_ns
    else:
        give_pts = None
    decoded_pts = nmc.decode_pop_resp(c, noise_pts_flat, trs,
                                      space_size - buff,
                                      buff, real_pts=give_pts,
                                      basin_hop=basin_hop, parallel=parallel,
                                      niter=n_hops)
    dist = distortion(all_pts_ns, decoded_pts, axis=1)
    struct_dist = dist.reshape(d_pairs.shape + (noise_samps,))
    struct_decoded_pts = decoded_pts.reshape(struct_dist.shape)
    struct_all_pts = all_pts_ns.reshape(struct_dist.shape)
    return struct_dist, struct_decoded_pts, struct_all_pts

def analyze_decoding(clean_pts, decoded_pts, distortion):
    mean_dec = np.expand_dims(np.nanmean(decoded_pts, axis=2), axis=2)
    bias = mean_dec[:, :, 0] - clean_pts[:, :, 0]
    mse = np.nanmean((decoded_pts - clean_pts)**2, axis=2)
    var = np.nanvar(decoded_pts, axis=2)
    cov = np.nanmean((decoded_pts[:, 0] - mean_dec[:, 0])
                     *(decoded_pts[:, 1] - mean_dec[:, 1]), axis=1)
    decoded_dists = decoded_pts[:, 1, :] - decoded_pts[:, 0, :]
    switches = np.nansum(decoded_dists < 0, axis=1)/decoded_pts.shape[2]
    return bias, var, cov, switches, decoded_dists, mse

def characterize_encoding(rf_size, rf_spacing, noise_var, noise_corr, dists,
                          noise_samps=1000, space_size=None, buff=None,
                          buff_mult=4, distortion=nmc.mse_distortion,
                          give_real=True, basin_hop=False, parallel=False,
                          oo=False, n_hops=100, rf_mult=10, corr_func=None,
                          space_mult=2):
    n_dists = len(dists)
    n_nc = len(noise_corr)
    bias = np.zeros((n_nc, n_dists, 2))
    var = np.zeros_like(bias)
    mse = np.zeros_like(var)
    cov = np.zeros((n_nc, n_dists))
    switch = np.zeros((n_nc, n_dists))
    d_dists = np.zeros((n_nc, n_dists, noise_samps))
    distorted = np.zeros((n_nc, n_dists, 2, noise_samps))
    decoded = np.zeros_like(distorted)
    orig = np.zeros_like(distorted)
    for i, nc in enumerate(noise_corr):
        out = estimate_encoding_swap(rf_size, rf_spacing, noise_var, nc, dists,
                                     basin_hop=basin_hop, give_real=give_real, 
                                     noise_samps=noise_samps, n_hops=n_hops,
                                     space_size=space_size, corr_func=corr_func,
                                     rf_mult=rf_mult, distortion=distortion,
                                     space_mult=space_mult)
        distorted[i], decoded[i], orig[i] = out
        out = analyze_decoding(orig[i], decoded[i], distorted[i])
        bias[i], var[i], cov[i], switch[i], d_dists[i], mse[i] = out
    return bias, var, cov, switch, d_dists, mse, distorted, decoded, orig
        
