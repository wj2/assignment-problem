import matplotlib.pyplot as plt
import numpy as np
import assignment.overlapping_features as am
import general.plotting as gpl
import general.utility as u
import scipy.stats as sts
import scipy.integrate as sint
import assignment.data_analysis as da

def plot_eg_overlap(ax1, ax2, n_stim=2, eg_s=100, color1=(.6, .6, .6),
                    color2=(.6, .6, .6)):
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    stims = np.random.rand(n_stim, 3)*100

    for stim in stims:
        gpl.plot_trace_werr(stim[1], stim[0], marker='o', ax=ax1,
                            color=color1)
        gpl.plot_trace_werr(stim[1], stim[2], marker='o', ax=ax2,
                            color=color2)
    
    ax1.set_xlim([0, eg_s])
    ax1.set_ylim([0, eg_s])
    _ = ax1.set_yticks([0, eg_s/2, eg_s])
    _ = ax1.set_yticklabels(['0', 's/2', 's'])
    _ = ax2.set_yticks([0, eg_s/2, eg_s])
    _ = ax2.set_yticklabels(['0', 's/2', 's'])
    _ = ax1.set_xticks([0, eg_s/2, eg_s])
    _ = ax1.set_xticklabels(['0', 's/2', 's'])
    ax2.set_xlabel('position')
    ax1.set_ylabel('audition')
    ax2.set_ylabel('vision')
    
def plot_noise_eg(stim_locs, dx, dy, axs, x_max=100, y_max=6, noise_delts=None,
                  n_pts=1000, d_height=5, d_alpha=.6, d_color=(.85, .85, .85),
                  txt_offset=.2, r_c1=(0, 0, 0), r_c2=(0, 0, 0)):
    ts, x_ax, y_ax = axs

    x_pts = np.linspace(-dx, dx, n_pts)
    y_pts = np.linspace(-dy, dy, n_pts)

    ts.vlines(stim_locs, 0, 1, zorder=10)
    if noise_delts is None:
        noise_delts = np.zeros((2, len(stim_locs)))
        noise_delts[0] = sts.norm(0, np.sqrt(dx)).rvs(len(stim_locs))
        noise_delts[1] = sts.norm(0, np.sqrt(dy)).rvs(len(stim_locs))
    x_ax.vlines(stim_locs + noise_delts[0], 0, 1, zorder=10, color=r_c2)
    y_ax.vlines(stim_locs + noise_delts[1], 0, 1, zorder=10, color=r_c1)

    x_distr_pts = sts.norm(0, np.sqrt(dx)).pdf(x_pts)
    x_distr_pts = d_height*x_distr_pts/max(x_distr_pts)
    y_distr_pts = sts.norm(0, np.sqrt(dy)).pdf(y_pts)
    y_distr_pts = d_height*y_distr_pts/max(y_distr_pts)
    mo_x = np.sort(stim_locs + noise_delts[0])
    mo_y = np.sort(stim_locs + noise_delts[1])
    for i, sl in enumerate(stim_locs):
        x_ax.fill_between(x_pts + sl, x_distr_pts, alpha=d_alpha, color=d_color)
        y_ax.fill_between(y_pts + sl, y_distr_pts, alpha=d_alpha, color=d_color)
        label_x = r'$\hat{X}_{' + str(i + 1) + '}$'
        label_y = r'$\hat{Y}_{' + str(i + 1) + '}$'
        x_ax.text(sl + noise_delts[0, i], 0 - txt_offset, label_x, ha='center',
                  va='top')
        y_ax.text(sl + noise_delts[1, i], 1 + txt_offset, label_y, ha='center',
                  va='bottom')
                
    ts.set_xlim([0, x_max])    
    ts.set_ylim([0, y_max])    
    ts.set_xticks([])
    x_ax.set_xlim([0, x_max])    
    x_ax.set_ylim([0, y_max])
    x_ax.set_xticks([])
    y_ax.set_xlim([0, x_max])    
    y_ax.set_ylim([0, y_max])
    y_ax.set_xticks([])
    ts.set_xlabel('common feature')
    
    gpl.clean_plot(y_ax, 1)
    gpl.clean_plot(x_ax, 1)
    gpl.clean_plot(ts, 1)
    
    
def plot_distance_error_prob(ax, distortions=(1, 10, 20), color=None,
                             distance_bounds=(.1, 20), n_dists=1000,
                             label=True, label_num=None, line_label=False,
                             ll_buff=.1, color_incr=.1, emp_ders=None,
                             emp_xs=None):
    distances = np.linspace(distance_bounds[0], distance_bounds[1], n_dists)
    sps = np.zeros((len(distortions), n_dists))
    for i, distort in enumerate(distortions):
        swap_prob = sts.norm(distances, np.sqrt(2*distort)).cdf(0)
        sp = sts.norm(distances, np.sqrt(2*distort)).cdf(0)
        swap_prob = 2*sp - 2*sp**2
        if label and label_num is None:
            label = r'$D_{X/Y} = '+'{}$'.format(distort)
        elif label and label_num is not None:
            label = r'$\Delta D = '+'{}$'.format(label_num[i])
        else:
            label = ''
        if label and line_label:
            legend_label = ''
        else:
            legend_label = label
        
        if i > 0:
            color = gpl.add_color(color, i*color_incr)
        gpl.plot_trace_werr(distances, swap_prob, ax=ax, label=legend_label,
                            color=color)
        if line_label:
            ms = np.array([int(np.floor(n_dists/2 - ll_buff*n_dists)),
                           int(np.ceil(n_dists/2 + ll_buff*n_dists))])
            gpl.label_plot_line(distances, swap_prob, label, ax, mid_samp=ms)
        sps[i] = swap_prob
        if emp_ders is not None and emp_xs is not None:
            gpl.plot_trace_werr(emp_xs, emp_ders[:, i].T, color=color,
                                error_func=gpl.conf95_interval, ax=ax)
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
    
    ax.set_xlabel(r'precision ratio ($s/D_{X/Y}$)')
    ax.set_ylabel('assignment error rate')

def plot_stan_model(model, ae_ax, dist_ax, uni_ax, n=4, spacing=np.pi/4, sz=8):
    m = model[0]
    rb_means = np.mean(m.samples['report_bits'], axis=0)
    db_means = np.mean(m.samples['dist_bits'], axis=0)
    sm_means = np.mean(m.samples['stim_mem'], axis=0)
    ae_prob, _ = da.ae_var_discrete(db_means, n, spacing=spacing,
                                    sz=sz)
    unif_prob = da.uniform_prob(sm_means, n)
    dist = da.dr_gaussian(rb_means, n)
    subj_xs = np.random.randn(len(dist))
    x_pos = np.array([0])
    ae_prob_arr = np.expand_dims(ae_prob, 1)
    p = ae_ax.violinplot(ae_prob, positions=x_pos,
                               showextrema=False)
    gpl.plot_trace_werr(x_pos, ae_prob_arr, points=True,
                        ax=ae_ax,
                        error_func=gpl.conf95_interval)

    dist_arr = np.expand_dims(dist, 1)
    p = dist_ax.violinplot(dist, positions=x_pos,
                         showextrema=False)
    gpl.plot_trace_werr(x_pos, dist_arr, points=True,
                        ax=dist_ax,
                        error_func=gpl.conf95_interval)

    up_arr = np.expand_dims(unif_prob, 1)
    p = uni_ax.violinplot(up_arr, positions=x_pos,
                         showextrema=False)
    gpl.plot_trace_werr(x_pos, up_arr, points=True,
                        ax=uni_ax,
                        error_func=gpl.conf95_interval)

    gpl.clean_plot(ae_ax, 0)
    gpl.clean_plot_bottom(ae_ax)
    gpl.clean_plot(dist_ax, 0)
    gpl.clean_plot_bottom(dist_ax)
    gpl.clean_plot(uni_ax, 0)
    gpl.clean_plot_bottom(uni_ax)
    gpl.make_yaxis_scale_bar(ae_ax, anchor=0, magnitude=.2, double=False,
                             label='assignment\nerror rate', text_buff=.95)
    gpl.make_yaxis_scale_bar(dist_ax, anchor=0, magnitude=.5, double=False,
                             label='local distortion\n(MSE)', text_buff=.8)
    gpl.make_yaxis_scale_bar(uni_ax, anchor=0, magnitude=.2, double=False,
                             label='guess rate', text_buff=.7)
    
def plot_multidim_schem(p1, p2, ax, non_alpha=.5):
    p1 = np.array(p1)
    p2 = np.array(p2)
    sp = np.stack((p1, p2), 0)
    ax.grid(False)
    ax.view_init(30, -75)
    for i in range(p1.shape[0]):
        xs = sp[:, 0]
        if i > 0:
            ys = sp[:, 1]
        else:
            ys = np.zeros(2)
        if i > 1:
            zs = sp[:, 2]
        else:
            zs = np.zeros(2)
        if i < 2:
            alpha = non_alpha
        else:
            alpha = 1
        l = ax.plot(xs, ys, zs, alpha=alpha)
        c = l[0].get_color()
        ax.plot(xs, ys, zs, 'o', color=c, alpha=alpha)
    ax.set_xticklabels(['', '0', 'd/2', 'd'])
    ax.set_yticklabels(['', '0', 'd/2', 'd'])
    ax.set_zticklabels(['', '0', 'd/2', 'd'])
    
    
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
        
        ax.vlines(m_f, ylim[0], curve_val, color=color, linestyle='dashed')
        ax.hlines(curve_val, xlim[0], m_f, color=color, linestyle='dashed')
    ax.set_ylim(ylim)

def plot_delta_xy(ds, circ_pt_delts, ax, color_st=None, line_color=(.8, .8, .8),
                  color_incr=.1):
    delta_eg = np.linspace(-.9, .9, 1000)
    dx, dy = am.dxdy_from_dsdelt(ds, delta_eg)

    gpl.plot_trace_werr(dx, dy, ax=ax, log_y=True, log_x=True, color=line_color)

    color = color_st
    for i, cpd in enumerate(circ_pt_delts):
        dx, dy = am.dxdy_from_dsdelt(ds, cpd)
        l = ax.plot(dx, dy, 'o', color=color)
        color = gpl.add_color(l[0].get_color(), color_incr)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.hlines(ds, *xl, color='k', linestyle='dashed')
    ax.vlines(ds, *yl, color='k', linestyle='dashed')
    ax.set_xlabel(r'$D_{X}$')
    ax.set_ylabel(r'$D_{Y}$')

def plot_empirical_distance(ax, dims, n_samps=10000, fact=100, **kwargs):
    for d in dims:
        samps = fact*sts.uniform(0, 1).rvs((2, d, n_samps))
        dists = np.sqrt(np.sum(np.diff(samps, axis=0)[0]**2, axis=0))
        ax.hist(dists, density=True, histtype='step', bins=30, **kwargs)
    return ax
    
def estimate_empirical_aes(dims, stim_count, pr, size, delta_d=(0,),
                           n_ests=5000, boot=True):
    ds = (size/pr)**2
    aes = np.zeros((len(dims), len(delta_d), len(ds), n_ests))
    for i, od in enumerate(dims):
        for j, dd in enumerate(delta_d):
            for k, ds_i in enumerate(ds):
                out = am.estimate_ae_full(size, stim_count, dd, ds_i, od,
                                          n_ests=n_ests)
                if boot:
                    out = u.bootstrap_on_axis(out, u.mean_axis0, axis=0,
                                              n=n_ests)
                aes[i, j, k] = out
    return aes 
    
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

def plot_ae_error(ax1, ax2, aes, redund, ols, pr, cut_end=10, delta_d=None,
                  aes_est=None, col_mult=.1):

    redund[redund < 0] = np.nan

    if delta_d is None:
        for i, ae in enumerate(aes):
            l = gpl.plot_trace_werr(pr[:-cut_end], ae[:-cut_end], ax=ax1,
                                    log_x=True, log_y=True,
                                    label='C = {}'.format(ols[i]))
            gpl.plot_trace_werr(pr[:-cut_end], redund[i, :-cut_end], ax=ax2,
                                log_x=True, log_y=False)
            if aes_est is not None:
                ae_est_plot = aes_est[i, 0, :-cut_end].T
                ae_col = l[0].get_color()
                gpl.plot_trace_werr(pr[:-cut_end], ae_est_plot,
                                    ax=ax1, color=ae_col,
                                    error_func=gpl.conf95_interval)
                                    
    else:
        for i, ae in enumerate(aes):
            color = None
            for j, dd in enumerate(delta_d):
                l_c = 'C = {}'.format(ols[i])
                l_d = r'$\Delta D = '+'{}$'.format(dd)
                l = ' , '.join((l_c, l_d))
                if j > 0:
                    color = gpl.add_color(color, j*col_mult)
                l = gpl.plot_trace_werr(pr[:-cut_end], ae[j, :-cut_end], ax=ax1,
                                        log_x=True, log_y=True, label=l,
                                        color=color)
                if aes_est is not None:
                    ae_est_plot = aes_est[i, j, :-cut_end].T
                    ae_col = l[0].get_color()
                    gpl.plot_trace_werr(pr[:-cut_end], ae_est_plot,
                                        ax=ax1, color=ae_col,
                                        error_func=gpl.conf95_interval)

                if j == 0:
                    color = l[0].get_color()
                gpl.plot_trace_werr(pr[:-cut_end], redund[i, j, :-cut_end],
                                    ax=ax2, log_x=True, log_y=False,
                                    color=color)
    ax1.set_xlabel('precision ratio ($s/D_{S}$)')
    ax2.set_xlabel('precision ratio ($s/D_{S}$)')
    ax1.set_ylabel('assignment error rate')
    ax2.set_ylabel('redundancy (nats)')

def plot_asymmetric_eg(ds, delt, integ_ax, indiv_ax, colors, n_pts=1000,
                       sd_mult=3):
    dx, dy = am.dxdy_from_dsdelt(ds, delt)
    pdf_pts = np.linspace(-sd_mult*np.sqrt(dy), sd_mult*np.sqrt(dy),
                          n_pts)
    c_x, c_y, c_s = colors 
    g_x = sts.norm(0, np.sqrt(dx)).pdf(pdf_pts)
    g_y = sts.norm(0, np.sqrt(dy)).pdf(pdf_pts)
    g_s = sts.norm(0, np.sqrt(ds)).pdf(pdf_pts)
    gpl.plot_trace_werr(pdf_pts, g_s, ax=integ_ax, color=c_s,
                        label=r'$s | \hat{x}, \hat{y}$')
    gpl.plot_trace_werr(pdf_pts, g_x, ax=indiv_ax, color=c_x,
                        label=r'$s | \hat{y}$')
    gpl.plot_trace_werr(pdf_pts, g_y, ax=indiv_ax, color=c_y,
                        label=r'$s | \hat{x}$')
    indiv_ax.set_xlabel(r'stimulus value $s$')
    indiv_ax.set_ylabel(r'posterior')
    integ_ax.set_ylabel(r'posterior')
    
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
