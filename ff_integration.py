
import math
import numpy as np
import scipy.stats as sts
import itertools as it
import general.rf_models as rfm
import general.utility as u
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def mse(x, y, axis=2):
    m = np.mean((x - y)**2, axis=axis)
    return m

class IntegrationModel():

    def __init__(self, f1_units, f2_units, out_units, input_distributions,
                 f1_inds, f2_inds, recon_inds, hu_units=None, act_func='relu',
                 scale=1, baseline=0, **kwargs):
        self.f1_inds = np.array(f1_inds, dtype=int)
        self.f2_inds = np.array(f2_inds, dtype=int)
        self.recon_inds = np.array(recon_inds, dtype=int)
        self.input_distributions = np.array(input_distributions, dtype=object)
        recon_distributions = self.input_distributions[self.recon_inds]
        self.out_units = out_units
        out = rfm.get_distribution_gaussian_resp_func(out_units,
                                                      recon_distributions,
                                                      scale=scale,
                                                      baseline=baseline)
        self.out_resp_f, _, self.ms_out, _ = out
        
        f1_distributions = self.input_distributions[self.f1_inds]
        self.f1_units = f1_units
        out = rfm.get_distribution_gaussian_resp_func(f1_units,
                                                      f1_distributions,
                                                      scale=scale,
                                                      baseline=baseline)
        self.resp_f1, _, self.ms_f1, _ = out
        
        f2_distributions = self.input_distributions[self.f2_inds]
        self.f2_units = f2_units
        out = rfm.get_distribution_gaussian_resp_func(f2_units,
                                                      f2_distributions,
                                                      scale=scale,
                                                      baseline=baseline)
        self.resp_f2, _, self.ms_f2, _ = out
        
        if hu_units is None:
            hu_units = (out_units**len(self.input_distributions),)
        self.model = self._construct_network(hu_units, out_units,
                                             act_func=act_func, **kwargs)
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

    def _generate_input(self, n_gen, n_stim=1, set_dists=None):
        inp = np.zeros((len(self.input_distributions), n_stim, n_gen))
        for i, inp_d in enumerate(self.input_distributions):
            inp[i] = inp_d.rvs((n_stim, n_gen))
        if set_dists is not None and n_stim > 1:
            for d_dim, d_dist in set_dists:
                cent = self.input_distributions[d_dim].mean()
                inp[d_dim, 0] = cent - d_dist/2
                inp[d_dim, 1] = cent + d_dist/2
        return inp

    def _section_input(self, inp):
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

    def multi_stim_func(self, *args):
        return np.sum(args, axis=0)
    
    def _generate_input_output_pairs(self, n_gen, n_stim, ret_indiv=False,
                                     noise_mag=0, set_dists=None):
        n_gen = int(n_gen)
        inp = self._generate_input(n_gen, n_stim, set_dists=set_dists)
        f1, f2, r = self._section_input(inp)
        f1, f2 = self._add_noise((f1, f2), noise_mag)
        resp1_i = self.multi_stim_func(*(self.resp_f1(f1_i.T) for f1_i in f1))
        resp2_i = self.multi_stim_func(*(self.resp_f2(f2_i.T) for f2_i in f2))
        recon_targ = self.multi_stim_func(*(self.out_resp_f(r_i.T) for r_i in r))
        total_inp = np.concatenate((resp1_i, resp2_i), axis=1)
        if ret_indiv:
            out = total_inp, recon_targ, inp, resp1_i, resp2_i
        else:
            out = total_inp, recon_targ, inp
        return out
        
    def _construct_network(self, hu_ns, out_units, act_func='relu',
                           activity_regularizer=regularizers.L2(.05),
                           **kwargs):
        ls = []
        for n in hu_ns:
            ls.append(layers.Dense(n, activation=act_func, **kwargs))
        out_units_n = out_units**len(self.recon_inds)
        ls.append(layers.Dense(out_units_n))
        model = keras.Sequential(ls)
        return model
        
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
                         dist_func=mse, excl_close=None, add_std=0, boots=None):
        out = self._generate_input_output_pairs(n_est, n_stim,
                                                noise_mag=noise_mag,
                                                set_dists=set_dists)
        x, y, inp = out
        y_hat = self.model(x)
        ys_all = self._make_alternate_outputs(inp)
        if excl_close is not None:
            _, _, r = self._section_input(inp)
            ms = self._get_min_pair_distances(r)
            close_mask = ms > excl_close
            ys_all = ys_all[:, close_mask]
            y_hat = y_hat[close_mask]
            n_est = np.sum(close_mask)
        ds = dist_func(np.expand_dims(y_hat, 0), ys_all)
        mes = self._select_map(ds, add_std=add_std)
        if boots is not None:
            f = lambda x: np.sum(x > 0)
            ae_rate = u.bootstrap_list(mes, f, n=boots)/n_est
        else:
            ae_rate = np.sum(mes > 0)/n_est
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
        
    def estimate_ae_rate_dists(self, n_stim, dists, noise_mag, n_est=10e4,
                               dist_func=mse, dim=0, excl_close=None,
                               boots=1):
        ae_rates = np.zeros((len(dists), boots))
        for i, d in enumerate(dists):
            set_dists = ((dim, d),)
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
        f1_is = set(self.f1_inds)
        f2_is = set(self.f2_inds)
        f1_unique = f1_is.difference(f2_is)
        f1_other = set(range(inp.shape[0])).difference(f1_unique)
        f1_unique = list(f1_unique)
        f1_other = list(f1_other)
        other_outs = []
        new_inp = np.zeros_like(inp)
        for p in it.permutations(range(n_stim)):
            new_inp[f1_unique] = inp[f1_unique, p]
            new_inp[f1_other] = inp[f1_other]
            _, _, r = self._section_input(new_inp)
            targ = self.multi_stim_func(*(self.out_resp_f(r_i.T) for r_i in r))
            
            other_outs.append(targ)
        return np.array(other_outs)
        
