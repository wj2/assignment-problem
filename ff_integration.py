
import math
import numpy as np
import scipy.stats as sts
import itertools as it
import sklearn.linear_model as sklm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import functools as ft
import matplotlib.pyplot as plt

import general.rf_models as rfm
import general.utility as u
import assignment.overlapping_features as am
import superposition_codes.codes as spc

def mse(x, y, axis=2):
    m = np.mean((x - y)**2, axis=axis)
    return m

def broadcast_distance(mat1, mat2, wid):
    dist = np.sum((mat1[:, np.newaxis] - mat2[np.newaxis, :])**2, axis=2)
    w_mat = np.exp(-dist/(2*wid**2))
    return w_mat

def quantify_ae_rate(m, dists=None, n_samps=100, n_stim=2,
                     dist_feat=0):
    if dists is None:
        dists = np.linspace(0, .5, 100)
    out_ae = np.zeros((n_samps, len(dists)))
    out_distortion = np.zeros_like(out_ae)
        
    for i, dist in enumerate(dists):
        for j in range(n_samps):
            out = m.random_example(n_stim, make_others=True,
                                   set_dists=((dist_feat, dist),))
            f1, f2, y, y_hat, inp, ys_all, ds = out
            out_distortion[j, i] = np.sum((y - y_hat)**2)
            corr_ind = np.where(np.all(y == ys_all, axis=-1))[0]
            dists = np.sum((y_hat - ys_all)**2, axis=(1, 2))
            out_ae[j, i] = corr_ind == np.argmin(dists)
    return out_ae, out_distortion

class IntegrationModel:

    def _generate_input(self, n_gen, n_stim=1, set_dists=None):
        inp = np.zeros((len(self.input_distributions), n_stim, n_gen))
        for i, inp_d in enumerate(self.input_distributions):
            inp[i] = inp_d.rvs((n_stim, n_gen))
        if set_dists is not None and n_stim > 1:
            for d_dim, d_dist in set_dists:
                cent = self.input_distributions[d_dim].mean()
                flips = self.rng.choice((-1, 1), size=n_gen)
                inp[d_dim, 0] = cent - flips*d_dist/2
                inp[d_dim, 1] = cent + flips*d_dist/2
        return inp

    def _section_input(self, inp):
        """ 
        input: dim x n_stim x n_trials 
        returns: n_stim x dim x n_trials
        """
        f1_inp = np.swapaxes(inp[self.f1_inds], 0, 1)
        f2_inp = np.swapaxes(inp[self.f2_inds], 0, 1)
        recon_inp = np.swapaxes(inp[self.recon_inds], 0, 1)
        
        return f1_inp, f2_inp, recon_inp

    def _add_noise(self, x, noise_mag, noise_cent=0, noise_distr=sts.norm):
        noise_mag = np.array(noise_mag)
        if len(noise_mag.shape) == 0:
            noise_mag = np.ones(len(x))*noise_mag
        out = []
        for i, nm in enumerate(noise_mag):
            if nm > 0:
                r = noise_distr(noise_cent, nm).rvs(x[i].shape)
            else:
                r = 0
            out.append(x[i] + r)
        return out

    def _add_noise_simple(self, x, noise_mag, noise_distr=sts.norm):
        noise = noise_distr(0, noise_mag).rvs(x.shape)
        x_noisy = x + noise
        return x_noisy

    def multi_stim_func(self, *args):
        return np.sum(args, axis=0)

    def get_resp(self, stim, func, add_noise=True):
        """
        input: n_stim x dim x n_trials
        returns: n_trials x n_neurs
        """
        reps = self.multi_stim_func(*(func(s_i.T) for s_i in stim))
        if add_noise:
            reps = self._add_noise_simple(reps, self.rep_noise)
        return reps
    
    def _generate_input_output_pairs(self, n_gen, n_stim, ret_indiv=False,
                                     inp_noise_mag=0, set_dists=None):
        n_gen = int(n_gen)
        inp = self._generate_input(n_gen, n_stim, set_dists=set_dists)
        f1, f2, r = self._section_input(inp)
        f1, f2 = self._add_noise((f1, f2), inp_noise_mag)

        resp1_i = self.get_resp(f1, self.resp_f1)
        resp2_i = self.get_resp(f2, self.resp_f2)
        inp_int = np.swapaxes(inp, 0, 1)
        integ_targ = self.get_resp(inp_int, self.resp_integ, add_noise=False)
        recon_targ = self.get_resp(r, self.resp_out, add_noise=False)
        total_inp = np.concatenate((resp1_i, resp2_i), axis=1)
        if ret_indiv:
            out = total_inp, integ_targ, recon_targ, inp, resp1_i, resp2_i
        else:
            out = total_inp, integ_targ, recon_targ, inp
        return out

    def random_example(self, n_stim, n_egs=1, make_others=False, dist_func=mse,
                       set_dists=None, noise_mag=0, topogriphy=False):
        out = self._generate_input_output_pairs(n_egs, n_stim, ret_indiv=True,
                                                set_dists=set_dists,
                                                noise_mag=noise_mag)
        x, y, inp, f1, f2 = out
        y_hat = self.model(x)
        if topogriphy:
            f1 = np.reshape(f1, (self.f1_units,)*len(self.f1_inds))
            f2 = np.reshape(f2, (self.f2_units,)*len(self.f2_inds))
            out_shape = (self.out_units,)*len(self.recon_inds)
            y = np.reshape(y, out_shape)
        if make_others:
            ys_all = self._make_alternate_outputs(inp)
            dists = dist_func(np.expand_dims(y_hat, 0), ys_all)
            if topogriphy:
                ysa_shape = (ys_all.shape[0],) + out_shape
                ys_all = np.reshape(ys_all, ysa_shape)
            out = f1, f2, y, y_hat, inp, ys_all, dists
        else:
            out = f1, f2, y, y_hat, inp
        return out

    def _get_min_pair_distances(self, ps, per_feature=True):
        n_stim, n_feats, n_est = ps.shape
        combs = it.combinations(range(n_stim), 2)
        ms_all = np.ones((1, int(n_est)))*np.inf
        for c in combs:
            c = np.array(c)
            diffs = np.diff(ps[c], axis=0)
            if per_feature:
                feat_dists = np.min(np.abs(diffs), axis=1)
            else:
                feat_dists = np.sqrt(np.sum(diffs**2, axis=1))
            m = np.concatenate((ms_all, feat_dists), axis=0)
            ms_all = np.expand_dims(np.min(m, axis=0), 0)
        return ms_all[0]

    def _select_map(self, ds, add_std=.02):
        ds = ds + sts.norm(0, add_std).rvs(ds.shape)
        mes = np.argmin(ds, axis=0)
        return mes
    
    def estimate_ae_rate(self, n_stim, noise_mag=0, set_dists=None, n_est=10e4,
                         dist_func=mse, excl_close=None, add_std=0,
                         boots=False):
        out = self._generate_input_output_pairs(n_est, n_stim,
                                                inp_noise_mag=noise_mag,
                                                set_dists=set_dists)
        x, integ, y, inp = out
        y_hat = self.model(x)
        stim_options, recon_options = self._make_alternate_outputs(inp)
        y_exp = np.expand_dims(y_hat, 1)
        dists = np.sum((y_exp - recon_options)**2, axis=-1)
        ae_errs = np.logical_not(np.argmin(dists, axis=1) == 0)
        
        # if excl_close is not None:
        #     _, _, r = self._section_input(inp)
        #     ms = self._get_min_pair_distances(r)
        #     close_mask = ms > excl_close
        #     ys_all = ys_all[:, close_mask]
        #     y_hat = y_hat[close_mask]
        #     n_est = np.sum(close_mask)
        # ds = dist_func(np.expand_dims(y_hat, 0), ys_all)
        # mes = self._select_map(ds, add_std=add_std)
        
        if boots:
            ae_rate = u.bootstrap_list(ae_errs, np.mean, n=n_est)
        else:
            ae_rate = ae_errs
        return ae_rate

    def get_unit_weights(self, unit=None, layer=0, topogriphy=True,
                         filt_mag=0):
        weights = self.model.weights[layer*2].numpy()
        weight_mask = np.max(np.abs(weights), axis=0) > filt_mag
        weights = weights[:, weight_mask]
        if unit is None:
            unit = np.random.randint(0, weights.shape[1])
        w_vec = weights[:, unit]
        if topogriphy:
            n_f1 = len(self.ms_f1)
            n_f2 = len(self.ms_f2)
            assert n_f1 + n_f2 == len(w_vec)
            n_side_f1 = int(np.round(n_f1**(1/self.ms_f1.shape[1])))
            n_side_f2 = int(np.round(n_f2**(1/self.ms_f2.shape[1])))
            w_f1 = np.reshape(w_vec[:n_f1], (n_side_f1,)*self.ms_f1.shape[1])
            w_f2 = np.reshape(w_vec[-n_f2:], (n_side_f2,)*self.ms_f2.shape[1])
            w_vec = (w_f1, w_f2)
        return w_vec            

    def get_unique_cents(self):
        f1_cents = list(np.unique(self.ms_f1[:, i])
                        for i in range(self.ms_f1.shape[1]))
        f2_cents = list(np.unique(self.ms_f2[:, i])
                        for i in range(self.ms_f2.shape[1]))
        out_cents = list(np.unique(self.ms_out[:, i])
                         for i in range(self.ms_out.shape[1]))
        return np.array(f1_cents), np.array(f2_cents), np.array(out_cents)
        
    def estimate_ae_rate_dists(self,
                               n_stim,
                               dists,
                               noise_mag=0,
                               n_est=10e4,
                               dist_func=mse,
                               common_dims=(0,),
                               unique_dims=(1, 2),
                               unique_dist=.5,
                               excl_close=None,
                               boots=False):
        ae_rates = np.zeros((len(dists), n_est))
        for i, d in enumerate(dists):
            set_dists = (tuple((di, unique_dist) for di in unique_dims)
                         + tuple((di, d) for di in common_dims))
            ae_rates[i] = self.estimate_ae_rate(n_stim, noise_mag=noise_mag,
                                                set_dists=set_dists, n_est=n_est,
                                                dist_func=dist_func, boots=boots,
                                                excl_close=excl_close)
        return ae_rates

    def estimate_ae_rate_noise(self, n_stim, noise_mags, n_est=10e4,
                               dist_func=mse, retrain=False, excl_close=None,
                               boots=1000, **kwargs):
        ae_rates = np.zeros((len(noise_mags), boots))
        for i, nm in enumerate(noise_mags):
            if retrain:
                self.train(n_stim, train_noise_mag=nm, **kwargs)
            ae_rates[i] = self.estimate_ae_rate(n_stim, noise_mag=nm,
                                                n_est=n_est,
                                                dist_func=dist_func,
                                                excl_close=excl_close,
                                                boots=boots)
        return ae_rates

    def _make_alternate_outputs(self, inp):
        n_stim = inp.shape[1]
        n_feats = inp.shape[0]
        n_trials = inp.shape[2]

        n_maps = np.math.factorial(n_stim)
        f1_is = set(self.f1_inds)
        f2_is = np.array(self.f2_inds)
        f1_unique = np.array(list(f1_is.difference(f2_is)))
        all_maps = np.zeros((n_feats, n_stim, n_maps, n_trials))
        all_recons = np.zeros((n_trials, n_maps, self.out_units))
        for i, p in enumerate(it.permutations(range(n_stim))):
            all_maps[:, :, i] = inp[:, p]
            all_maps[f1_unique, :, i] = inp[f1_unique, :]
            use_map_inp = np.swapaxes(all_maps[:, :, i], 0, 1)
            use_map_inp = use_map_inp[..., self.recon_inds, :]
            all_recons[:, i] = self.get_resp(use_map_inp, self.resp_out,
                                             add_noise=False)
        return all_maps, all_recons
        
    def _construct_network(self, inp_units, hu_ns, out_units, act_func='relu',
                           activity_regularizer=None, # regularizers.L2(.001),
                           noise=0, no_output=False, **kwargs):
        ls = []
        ls.append(layers.InputLayer(inp_units))
        for n in hu_ns:
            ls.append(layers.Dense(n, activation=act_func,
                                   activity_regularizer=activity_regularizer,
                                   **kwargs))
            if noise > 0:
                ls.append(layers.GaussianNoise(noise))
        if not no_output:
            ls.append(layers.Dense(out_units, activation=act_func))
        if no_output:
            model_int = keras.Sequential(ls)
            model_out = None
        else:
            model_int = keras.Sequential(ls[:-1])
            model_out = keras.Sequential(ls)
        return model_int, model_out

class RandomPopsModel(IntegrationModel):

    def __init__(self, f1_units, f2_units, out_units, input_distributions,
                 f1_inds, f2_inds, recon_inds, integ_units=1000,
                 hu_units=(500,), noise=1, inp_pwr=20, pop_func='random',
                 connectivity_gen='learn', **kwargs):
        self.rng = np.random.default_rng()
        self.inp_pwr = inp_pwr
        self.f1_inds = np.array(f1_inds, dtype=int)
        self.f2_inds = np.array(f2_inds, dtype=int)
        self.recon_inds = np.array(recon_inds, dtype=int)
        self.input_distributions = np.array(input_distributions, dtype=object)
        self.rep_noise = noise
        self.hu_units = hu_units

        self.f1_units = f1_units
        f1_distributions = self.input_distributions[self.f1_inds]
        f1_code = spc.Code(inp_pwr, f1_units, dims=len(f1_distributions),
                           sigma_n=noise, pop_func=pop_func)
        self.f1_code = f1_code
        self.resp_f1 = f1_code.rf
        self.ms_f1 = f1_code.rf_cents
        self.wid_f1 = f1_code.wid

        self.f2_units = f2_units
        f2_distributions = self.input_distributions[self.f2_inds]
        f2_code = spc.Code(inp_pwr, f2_units, dims=len(f2_distributions),
                           sigma_n=noise, pop_func=pop_func)
        self.f2_code = f2_code
        self.resp_f2 = f2_code.rf
        self.ms_f2 = f2_code.rf_cents
        self.wid_f2 = f2_code.wid

        self.integ_units = integ_units
        integ_code = spc.Code(2*inp_pwr, integ_units,
                            dims=len(input_distributions),
                            sigma_n=noise, pop_func=pop_func)
        self.integ_code = integ_code
        self.resp_integ = integ_code.rf
        self.ms_integ = integ_code.rf_cents
        self.wid_integ = integ_code.wid
        
        recon_distributions = self.input_distributions[self.recon_inds]
        self.out_units = out_units
        out_code = spc.Code(inp_pwr, out_units, dims=len(recon_distributions),
                            sigma_n=noise, pop_func=pop_func)
        self.out_code = out_code
        self.resp_out = out_code.rf
        self.ms_out = out_code.rf_cents
        self.wid_out = out_code.wid

        if connectivity_gen == 'naive':
            out = self.generate_naive_connectivity(**kwargs)
        elif connectivity_gen == 'learn_linear':
            out = self.learn_linear_connectivity(**kwargs)
        elif connectivity_gen == 'learn_nonlinear':
            out = self.learn_nonlinear_connectivity(hu_units=hu_units,
                                                    **kwargs)
        elif connectivity_gen == 'learn_nonlinear_piece':
            out = self.learn_nonlinear_piece_connectivity(hu_units=hu_units,
                                                          **kwargs)
        else:
            s = 'unknown connectivity generator {}'.format(connectivity_gen)
            raise IOError(s)

        self.integ_func, self.out_func = out
        
    def generate_naive_connectivity(self, normalize=True,
                                    use_wid=.05):
        if use_wid is not None:
            wids = (use_wid, use_wid, use_wid)
        else:
            wids = (self.wid_f1, self.wid_f2, self.wid_out)
        w_f1_integ = broadcast_distance(self.ms_integ[:, self.f1_inds],
                                        self.ms_f1,
                                        wids[0])
        
        wid_f2_int = np.sqrt(self.wid_integ**2 + self.wid_f2**2)
        w_f2_integ = broadcast_distance(self.ms_integ[:, self.f2_inds],
                                        self.ms_f2,
                                        wids[1])

        
        wid_int_out = np.sqrt(self.wid_integ**2 + self.wid_out**2)
        w_integ_out = broadcast_distance(self.ms_out,
                                         self.ms_integ[:, self.recon_inds],
                                         wids[2])
        ws = (w_f1_integ, w_f2_integ, w_integ_out)
        if normalize:
            ws = list(normalize_weight(w, axis=1)
                      for w in out)
        w_f1_integ, w_f2_integ, w_integ_out = ws
        w_f12_integ = np.concatenate((w_f1_integ, w_f2_integ),
                                     axis=1)
        
        def integ_func(x, thresh=0):
            integ = np.dot(w_f12_integ, x)
            integ[integ < thresh] = 0
            return integ

        def out_func(x, thresh_int=0, thresh_out=0):
            integ = integ_func(x, thresh=thresh_int)
            out = np.dot(w_integ_out, integ)
            out[out < thresh_out] = 0
            return out

        return integ_func, out_func

    def learn_linear_connectivity(self, n_samples=10000, model=sklm.Ridge,
                                  n_stim=1):
        out = self._generate_input_output_pairs(n_samples, n_stim,
                                                ret_indiv=True)
        inp, integ, recon, stim, f1_r, f2_r = out
        m_inp_int = model()
        m_inp_int.fit(inp, integ)
        w_f1_int = m_inp_int.coef_[:, :f1_r.shape[1]]
        w_f2_int = m_inp_int.coef_[:, f1_r.shape[1]:]
        
        int_bias = m_inp_int.intercept_
        
        m_out_int = model()
        m_out_int.fit(integ, recon)
        w_int_out = m_out_int.coef_

        out_bias = m_out_int.intercept_
        w_fi_full = np.concatenate((w_f1_int, w_f2_int), axis=1)
        
        def integ_func(x, thresh=0):
            integ = m_inp_int.predict(x)
            integ[integ < thresh] = 0
            return integ

        def out_func(x, thresh_int=0, thresh_out=0):
            integ = integ_func(x, thresh=thresh_int)
            out = m_int_out.predict(integ)
            out[out < thresh_out] = 0
            return out
        
        return integ_func, out_func

    def _compile(self, model, loss=keras.losses.MeanSquaredError(), **kwargs):
        self.model.compile(loss=loss, **kwargs)
        
    def learn_nonlinear_connectivity(
            self,
            n_samples=50000,
            n_stim=2,
            loss=keras.losses.MeanSquaredError(),
            epochs=40,
            verbose=False,
            batch_size=200,
            hu_units=None,
            use_early_stopping=True,
            n_val=10000,
            no_integ=False,
            **kwargs
    ):
        out = self._generate_input_output_pairs(n_samples, n_stim,
                                                ret_indiv=True)
        inp, integ, recon, stim, f1_r, f2_r = out

        out = self._generate_input_output_pairs(n_val, n_stim,
                                                ret_indiv=True)        
        val_inp, _, val_recon, _, _, _ = out

        if no_integ:
            layers = ()
        else:
            layers = (integ.shape[1],)
        if hu_units is None:
            hu_units = layers
        else:
            hu_units = hu_units + layers
        m_int, m_out = self._construct_network(inp.shape[1], hu_units,
                                               recon.shape[1], **kwargs)
        
        if use_early_stopping:
            cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             mode='min', patience=2)
            curr_cb = kwargs.get('callbacks', [])
            curr_cb.append(cb)
            kwargs['callbacks'] = curr_cb
            
        m_out.compile(loss=loss)
        m_out.fit(inp, recon, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, validation_data=(val_inp, val_recon))
        
        return m_int, m_out

    def learn_nonlinear_piece_connectivity(
            self,
            n_samples=50000,
            n_stim=2,
            loss=keras.losses.MeanSquaredError(),
            epochs=40,
            verbose=True,
            batch_size=200,
            model=sklm.Ridge,
            hu_units=None,
            use_early_stopping=True,
            n_val=10000,
            no_integ=False,
            **kwargs
    ):
        out = self._generate_input_output_pairs(n_samples, n_stim,
                                                ret_indiv=True)
        inp, integ, recon, stim, f1_r, f2_r = out

        out = self._generate_input_output_pairs(n_val, n_stim,
                                                ret_indiv=True)        
        val_inp, val_integ, _, _, _, _ = out

        if hu_units is None:
            hu_units = (integ.shape[1],)
        else:
            hu_units = hu_units + (integ.shape[1],)        
        m_int, _ = self._construct_network(inp.shape[1], hu_units,
                                           recon.shape[1], no_output=True,
                                           **kwargs)
        m_int.compile(loss=loss)
        if use_early_stopping:
            cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             mode='min', patience=2)
            curr_cb = kwargs.get('callbacks', [])
            curr_cb.append(cb)
            kwargs['callbacks'] = curr_cb

        m_int.fit(inp, integ, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, validation_data=(val_inp, val_integ),
                  **kwargs)

        m_out_int = model()
        m_out_int.fit(m_int(inp), recon)
        def out_func(x, thresh_out=0):
            integ = m_int(x)
            out = m_out_int.predict(integ)
            out[out < thresh_out] = 0
            return out
        
        return m_int, out_func

    def get_integ_rep(self, inps, thresh=None, nonlin=True):
        integ_r = self.integ_func(inps)
        # if thresh is None:
        #     thresh = 0
        # inps = np.expand_dims(inps, 1)
        # integ_r = np.sum(inps*self.w_inp_full, axis=2) + self.int_bias
        # if nonlin:
        #     integ_r[integ_r < thresh] = 0
        return integ_r
        
    def model(self, inps, **kwargs):
        # integ_r = np.expand_dims(self.get_integ_rep(inps, **kwargs), 1)
        # out = np.sum(np.expand_dims(self.w_int_out, 0)*integ_r, axis=2)
        out = self.out_func(inps)
        # return out + self.out_bias
        return out

    def get_theoretical_ae(self, ds, n_stim=2, use_full=True):
        if use_full:
            mse_f1 = self.f1_code.get_predicted_mse()
            mse_f2 = self.f2_code.get_predicted_mse()
        else:
            mse_f1 = 1/self.f1_code.get_predicted_fi()
            mse_f2 = 1/self.f2_code.get_predicted_fi()
        
        ae_rate = am.dist_ae_prob(ds, mse_f1, mse_f2, n_stim=n_stim)
        return ae_rate

class RandomPopsRecurrent(RandomPopsModel):

    def learn_nonlinear_connectivity(
            self,
            n_samples=50000,
            n_stim=2,
            loss=keras.losses.MeanSquaredError(),
            epochs=40,
            verbose=False,
            batch_size=200,
            **kwargs
    ):
        out = self._generate_input_output_pairs(n_samples, n_stim,
                                                ret_indiv=True)
        inp, integ, recon, stim, f1_r, f2_r = out

        m = self._construct_network(inp.shape[1], (integ.shape[1],),
                                    recon.shape[1], **kwargs)
        m.compile(loss=loss)
        m.fit(inp, recon, batch_size=batch_size, epochs=epochs,
              verbose=verbose)
        self.tf_model = m
        out = list(w.numpy().T for w in m.weights)
        w_inp_int, int_bias, w_int_out, out_bias = out
        w_f1_int = w_inp_int[:, :f1_r.shape[1]]
        w_f2_int = w_inp_int[:, f1_r.shape[1]:]
        return w_f1_int, w_f2_int, w_int_out, int_bias, out_bias
            
    
def normalize_weight(w, axis=1):
    w_norm = w/np.sum(w, axis=axis, keepdims=True)
    return w_norm

class MLPIntegrationModel(IntegrationModel):

    def __init__(self, f1_units, f2_units, out_units, input_distributions,
                 f1_inds, f2_inds, recon_inds, integ_units=None, act_func='relu',
                 noise=1, inp_pwr=10,
                 **kwargs):
        self.f1_inds = np.array(f1_inds, dtype=int)
        self.f2_inds = np.array(f2_inds, dtype=int)
        self.recon_inds = np.array(recon_inds, dtype=int)
        self.input_distributions = np.array(input_distributions, dtype=object)
        recon_distributions = self.input_distributions[self.recon_inds]
        self.out_units = out_units
        
        out_code = spc.Code(inp_pwr, out_units, dims=len(recon_distributions),
                            sigma_n=noise)
        self.out_code = out_code
        self.resp_out = ft.partial(out_code.get_rep, add_noise=False)
        self.ms_out = None
        
        f1_distributions = self.input_distributions[self.f1_inds]
        f1_code = spc.Code(inp_pwr, f1_units, dims=len(f1_distributions),
                           sigma_n=noise)
        self.resp_f1 = f1_code.get_rep
        self.ms_f1 = None
        
        f2_distributions = self.input_distributions[self.f2_inds]
        f2_code = spc.Code(inp_pwr, f2_units, dims=len(f2_distributions),
                           sigma_n=noise)
        self.resp_f2 = f2_code.get_rep
        self.ms_f2 = None
        
        if hu_units is None:
            hu_units = (out_units,)
        self.model = self._construct_network(f1_units + f2_units, integ_units,
                                             out_units,
                                             act_func=act_func,
                                             noise=noise, **kwargs)
        self._model_compiled = False

    def compile(self, loss=keras.losses.MeanSquaredError(), **kwargs):
        self.model.compile(loss=loss, **kwargs)
        self._model_compiled = True
        
    def train(self, n_stim, train_size=10e4, batch_size=64, epochs=2,
              val_size=100, train_noise_mag=0, val_noise_mag=0,
              validate_n=None):
        if not self._model_compiled:
            self.compile()
        out = self._generate_input_output_pairs(train_size, n_stim,
                                                noise_mag=train_noise_mag)
        inp_train, out_train, _ = out
        if validate_n is None:
            validate_n = n_stim
        out = self._generate_input_output_pairs(val_size, validate_n,
                                                noise_mag=val_noise_mag)
        val_d = out[:2]
        history = self.model.fit(inp_train, out_train, batch_size=batch_size,
                                 epochs=epochs, validation_data=val_d)
        return history
        
def visualize_integ_layers(m, n_stim=2, dists=None, flat_vis=False):
    if dists is None:
        dists = ((0, .2), (1, .5),
                 (2, .5))
    out = m._generate_input_output_pairs(1, n_stim, set_dists=dists)
    inp, integ, recon, stim = out

    recon_hat = m.model(inp)
    integ_hat = m.get_integ_rep(inp)
    
    shown_stim = stim[..., 0].T
    recon_stim = shown_stim[:, m.recon_inds]

    fwid = 3
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(fwid*2, fwid))
    rfm.visualize_random_rf_responses(recon[0], m.ms_out, ax=ax1,
                                      plot_stim=recon_stim)
    rfm.visualize_random_rf_responses(recon_hat[0], m.ms_out, ax=ax2,
                                      plot_stim=recon_stim)

    f = plt.figure(figsize=(fwid*2, fwid))
    ax1 = f.add_subplot(1, 2, 1, projection='3d')
    ax2 = f.add_subplot(1, 2, 2, projection='3d')
    ax1 = rfm.visualize_random_rf_responses(integ_hat[0], m.ms_integ,
                                            vis_dims=(0, 1, 2), ax=ax1,
                                            plot_stim=shown_stim)
    ax2 = rfm.visualize_random_rf_responses(integ[0], m.ms_integ,
                                            vis_dims=(0, 1, 2), ax=ax2,
                                            plot_stim=shown_stim)
    for ax in (ax1, ax2):
        ax.set_xlabel('shared')
        ax.set_ylabel('r1')
        ax.set_zlabel('r2')

    if flat_vis:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fwid*3, fwid))
        rfm.visualize_random_rf_responses(integ_hat[0], m.ms_integ,
                                          vis_dims=(0, 2),
                                          ax=ax1, plot_stim=shown_stim)
        rfm.visualize_random_rf_responses(integ_hat[0], m.ms_integ,
                                          vis_dims=(0, 1),
                                          ax=ax2, plot_stim=shown_stim)
        rfm.visualize_random_rf_responses(integ_hat[0], m.ms_integ,
                                          vis_dims=(1, 2),
                                          ax=ax3, plot_stim=shown_stim)
