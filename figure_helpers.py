import matplotlib.pyplot as plt
import numpy as np
import assignment.overlapping_features as am
import general.plotting as gpl
import scipy.stats as sts
import scipy.integrate as sint

def plot_eg_overlap(ax1, ax2, n_stim=2, eg_s=100):
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    stims = np.random.rand(n_stim, 3)*100

    for stim in stims:
        gpl.plot_trace_werr(stim[1], stim[0], marker='o', ax=ax1)
        gpl.plot_trace_werr(stim[1], stim[2], marker='o', ax=ax2)
    
    ax1.set_xlim([0, eg_s])
    ax1.set_ylim([0, eg_s])
    _ = ax1.set_yticks([0, eg_s/2, eg_s])
    _ = ax1.set_yticklabels(['0', 's/2', 's'])
    _ = ax2.set_yticks([0, eg_s/2, eg_s])
    _ = ax2.set_yticklabels(['0', 's/2', 's'])
    _ = ax1.set_xticks([0, eg_s/2, eg_s])
    _ = ax1.set_xticklabels(['0', 's/2', 's'])
    ax2.set_xlabel(r'$f_{2}$')
    ax1.set_ylabel(r'$f_{1}$')
    ax2.set_ylabel(r'$f_{3}$')

def plot_distance_error_prob(ax, distortions=(1, 10, 20), color_st=None,
                             distance_bounds=(.1, 20), n_dists=1000,
                             label=True, label_num=None):
    distances = np.linspace(distance_bounds[0], distance_bounds[1], n_dists)
    sps = np.zeros((len(distortions), n_dists))
    for i, distort in enumerate(distortions):
        swap_prob = sts.norm(distances, np.sqrt(2*distort)).cdf(0)
        if label and label_num is None:
            label = r'$D_{1/2} = '+'{}$'.format(distort)
        elif label and label_num is not None:
            label = r'$\Delta D = '+'{}$'.format(label_num[i])
        else:
            label = ''
        color = gpl.add_color(color_st, i*.1)
        gpl.plot_trace_werr(distances, swap_prob, ax=ax, label=label,
                            color=color)
        sps[i] = swap_prob
    _ = ax.set_xlabel('distance between stimuli')
    _ = ax.set_ylabel('assignment error\nprobability')
    return sps, distances

def plot_distance_distrib_prob(ax, overlap_dims=(1,),
                               funcs=(am.line_picking_line,), n_dists=1000):
    for i, d in enumerate(overlap_dims):
        dists = np.linspace(0, np.sqrt(d), n_dists)
        func = funcs[i]
        gpl.plot_trace_werr(dists, func(dists), ax=ax, label='C = {}'.format(d))
    ax.set_xlabel('distance between stimuli')
    _ = ax.set_ylabel('probability density')

def plot_ae_rate(a_rate, a_approx, pr, s_counts, ax, colors=None):
    for i, s in enumerate(s_counts):
        l = gpl.plot_trace_werr(pr, a_rate[i], label='N = {}'.format(s), ax=ax,
                                log_x=True, log_y=True,
                                error_func=gpl.conf95_interval,
                                color=colors[i])
        gpl.plot_trace_werr(pr, a_approx[i], ax=ax, linestyle='dashed',
                            color=colors[i])
    
    ax.set_xlabel(r'precision ratio ($s/D_{1/2}$)')
    ax.set_ylabel('assignment error rate')

def plot_distance_dim_means(ax, dims, funcs, size, curve, dists, colors=None):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    for i, f in enumerate(funcs):
        if colors is not None:
            color = colors[i]
        else:
            color = None
        mean_pdf = lambda x: x*f(x)
        m_f = sint.quad(mean_pdf, 0, np.sqrt(dims[i]))[0]*size
        curve_ind = np.argmin((m_f - dists)**2)
        curve_val = curve[curve_ind]
        
        ax.vlines(m_f, ylim[0], curve_val, color=color)
        ax.hlines(curve_val, xlim[0], m_f, color=color)
    ax.set_ylim(ylim)

def compute_ae_red(dims, stim_count, pr, size, delta_d=(0,)):
    ds = (size/pr)**2
    aes = np.zeros((len(dims), len(delta_d), len(ds)))
    redund = np.zeros_like(aes)
    for i, od in enumerate(dims):
        for j, dd in enumerate(delta_d):
            dx, dy = am.dxdy_from_dsdelt(ds, dd)
            aes[i, j] = am.integrate_assignment_error(stim_count, dx, dy, od)
            redund[i, j] = am.feature_redundant_info(stim_count, dx, dy, od)
    return aes, redund

def plot_ae_error(ax1, ax2, aes, redund, ols, pr, cut_end=10, delta_d=None):

    redund[redund < 0] = np.nan

    if delta_d is None:
        for i, ae in enumerate(aes):
            gpl.plot_trace_werr(pr[:-cut_end], ae[:-cut_end], ax=ax1,
                                log_x=True, log_y=True,
                                label='C = {}'.format(ols[i]))
            gpl.plot_trace_werr(pr[:-cut_end], redund[i, :-cut_end], ax=ax2,
                                log_x=True, log_y=False)
    else:
        for i, ae in enumerate(aes):
            color = None
            for j, dd in enumerate(delta_d):
                l_c = 'C = {}'.format(ols[i])
                l_d = r'$\Delta D = '+'{}$'.format(dd)
                l = ' , '.join((l_c, l_d))
                if j > 0:
                    color = gpl.add_color(color, j*.1)
                l = gpl.plot_trace_werr(pr[:-cut_end], ae[j, :-cut_end], ax=ax1,
                                        log_x=True, log_y=True, label=l,
                                        color=color)
                if j == 0:
                    color = l[0].get_color()
                gpl.plot_trace_werr(pr[:-cut_end], redund[i, j, :-cut_end],
                                    ax=ax2, log_x=True, log_y=False,
                                    color=color)
    ax1.set_xlabel('precision ratio ($s/D_{S}$)')
    ax2.set_xlabel('precision ratio ($s/D_{S}$)')
    ax1.set_ylabel('assignment error rate')
    ax2.set_ylabel('redundancy (nats)')

def plot_rdb(ax, bits_range=(0, 20), n_bits=1000, size=100):
    bits = np.linspace(bits_range[0], bits_range[1], n_bits)
    p = size

    dist = ((p**2)/(2*np.pi))*np.exp(-bits)

    gpl.plot_trace_werr(bits, dist, ax=ax, log_y=True, color='r')
    
    poly_pts = [[bits[0], dist[0]], [bits[-1], dist[-1]], [0, 0]]
    imp_region = plt.Polygon(poly_pts, color='r', alpha=.2)
    ax.add_patch(imp_region)

    ax.set_xlabel(r'information rate $I(X; \hat{X})$ (bits)')
    ax.set_ylabel('estimator variance (MSE)')
    _ = ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim((bits[0], bits[-1]))
    ax.set_ylim((dist[-1], dist[0]))

def compute_bit_alloc(bits, k, c, delta_d, size, n_stim=2):
    ds = am.compute_ds(bits, k, c, delta_d, n_stim, s=size)
    unique_b = am.rdb_gaussian(ds, size)
    dx, dy = am.dxdy_from_dsdelt(ds, delta_d)
    common_b_x = am.rdb_gaussian(dx, size)
    common_b_y = am.rdb_gaussian(dy, size)
    return unique_b, common_b_x, common_b_y, ds, dx, dy
