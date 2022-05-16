
import assignment.overlapping_features as am
import assignment.data_analysis as da
import general.plotting as gpl
import general.plotting_styles as gps
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import arviz as av
import pickle
import scipy.stats as sts
import os
import assignment.ff_integration as ff
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import general.rf_models as rfm

from assignment.figure_helpers import *

bf = ('/Users/wjj/Dropbox/research/uc/freedman/analysis/'
      'assignment/figs/')
colors = gps.nms_colors
n_colors = gps.assignment_num_colors
r_colors = np.array([[186, 143, 149],
                     [167, 148, 183],
                     [214, 142, 214]])/255

def setup():
    gps.set_paper_style(colors)
    
def figure1(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'e', 'f')
    if data is None:
        data = {}
    fsize = (5, 3.5)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(10, 10)

    schem_grid = gs[0:6, 0:5]
    overlap_eg_1_grid = gs[0:3, 5:7]
    overlap_eg_2_grid = gs[3:6, 5:7]
    noise_eg_grid = gs[0:6, 7:]
    noise_eg_1_grid = gs[0:2, 7:]
    noise_eg_2_grid = gs[2:4, 7:]
    noise_eg_3_grid = gs[4:6, 7:]
    distance_error_grid = gs[6:, 0:3]
    distance_distrib_grid = gs[6:, 3:6]
    assignment_error_grid = gs[6:, 6:]

    if 'a' in gen_panels:
        schem_ax = f.add_subplot(schem_grid)

    if 'b' in gen_panels:
        ol_ax_1 = f.add_subplot(overlap_eg_1_grid)
        ol_ax_2 = f.add_subplot(overlap_eg_2_grid, sharex=ol_ax_1)
        plot_eg_overlap(ol_ax_1, ol_ax_2, color1=r_colors[0],
                        color2=r_colors[1])

    stim_locs = (20, 40, 80)
    dx = 10**2
    dy = 10**2
    noise_delts = np.array([[6, 4, -5],
                            [12, -21, 3]])
    if 'c' in gen_panels:
        noise1_ax = f.add_subplot(noise_eg_1_grid)
        noise2_ax = f.add_subplot(noise_eg_2_grid)
        noise3_ax = f.add_subplot(noise_eg_3_grid)
        plot_noise_eg(stim_locs, dx, dy, (noise3_ax, noise2_ax, noise1_ax),
                      noise_delts=noise_delts, r_c1=r_colors[0],
                      r_c2=r_colors[1])

    dists = np.array((1, 10, 20))
    dist_bounds = (.1, 20)
    emp_xs = np.linspace(dist_bounds[0], dist_bounds[1], 50)
    n_emps = 1000
    if 'd' in gen_panels:
        de_ax = f.add_subplot(distance_error_grid)
        if 'd' in data.keys():
            emp_ders, emp_xs = data['d']
        else:
            emp_ders = am.fixed_distance_ds_dxdy(emp_xs, dists, dists)
            data['d'] = (emp_ders, emp_xs)
        plot_distance_error_prob(de_ax, distortions=dists,
                                 emp_ders=emp_ders, emp_xs=emp_xs,
                                 color=n_colors[0])

    if 'e' in gen_panels:
        dd_ax = f.add_subplot(distance_distrib_grid)
        plot_distance_distrib_prob(dd_ax)

    if 'f' in gen_panels:
        pr = np.logspace(0, 4, 50)
        stim_ns = np.arange(2, 6, 1)
        n_samps = 5000
        
        if data is not None and 'f' in data.keys():
            a_rate, a_approx, pr, stim_ns = data['f']
        else:
            a_rate = am.estimate_ae_sr_s_ranges(stim_ns, pr, n=n_samps)
            a_approx = am.error_approx_further(stim_ns, pr)
            data['f'] = (a_rate, a_approx, pr, stim_ns)
        ae_ax = f.add_subplot(assignment_error_grid)
        plot_ae_rate(a_rate, a_approx, pr, stim_ns, ae_ax, colors=n_colors)

    fname = os.path.join(bf, 'fig1-py.pdf')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure2(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'de')
    if data is None:
        data = {}

    fsize = (5, 3)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(12, 12)

    v_mid = 5
    schem_grid = gs[0:v_mid, 0:4]
    distance_distrib_grid = gs[0:v_mid, 4:8]
    distance_prob_grid = gs[0:v_mid, 8:]
    ae_c_grid = gs[v_mid:, 0:6]
    red_c_grid = gs[v_mid:, 6:]

    pt1 = (0, 0, 0)
    pt2 = (4, 4, 4)
    if 'a' in gen_panels:
        schem_ax = f.add_subplot(schem_grid, projection='3d')
        plot_multidim_schem(pt1, pt2, schem_ax)

    ol_dims = (1, 2, 3)
    ol_funcs = [am.line_picking_line, am.line_picking_square,
                am.line_picking_cube]
    if 'b' in gen_panels:
        distrib_ax = f.add_subplot(distance_distrib_grid)
        plot_distance_distrib_prob(distrib_ax, overlap_dims=ol_dims,
                                   funcs=ol_funcs)

    ds = (10,)
    size = 100
    bound_color = np.array((.3, .3, .3))
    if 'c' in gen_panels:
        ep_ax = f.add_subplot(distance_prob_grid)
        ep_ax.set_yscale('log')
        out = plot_distance_error_prob(ep_ax, distortions=ds,
                                       color=bound_color,
                                       distance_bounds=(20, size - 20),
                                       line_label=True)
        curve, dists = out
        plot_distance_dim_means(ep_ax, ol_dims, ol_funcs, size, curve[0],
                                dists, colors=colors)

    stim_count = (2,)
    n_ests = 10000
    pr = np.logspace(0, 3, 50)
    if 'de' in data.keys():
        aes, redund, pr, stim_count, ol_dims, aes_est = data['de']
    elif 'de' in gen_panels:        
        aes_est = estimate_empirical_aes(ol_dims, stim_count[0], pr, size,
                                         n_ests=n_ests)
        aes, redund = compute_ae_red(ol_dims, stim_count, pr, size)
        data['de'] = (aes, redund, pr, stim_count, ol_dims, aes_est)
    if 'de' in gen_panels:
        ae_ax = f.add_subplot(ae_c_grid)
        red_ax = f.add_subplot(red_c_grid)
        plot_ae_error(ae_ax, red_ax, aes[:, 0], redund[:, 0], ol_dims, pr,
                      aes_est=aes_est)

    fname = os.path.join(bf, 'fig2-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure3(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'ef')
    if data is None:
        data = {}

    fsize = (3.5, 5.5)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(12, 12)

    schem_1_grid = gs[0:2, 0:6]
    schem_2_grid = gs[2:4, 0:6]
    dxdeltad_grid = gs[0:2, 6:]
    distae_grid = gs[2:4, 6:]
    ae_delt_grid = gs[4:8, :]
    red_delt_grid = gs[8:, :]

    ds = 10
    delt = .45
    if 'a' in gen_panels:
        integ_ax = f.add_subplot(schem_1_grid)
        indiv_ax = f.add_subplot(schem_2_grid, sharex=integ_ax, sharey=integ_ax)
        plot_asymmetric_eg(ds, delt, integ_ax, indiv_ax, colors=r_colors)

    delta_x = np.linspace(0, .9, 1000)
    ds = 10
    dx, dy = am.dxdy_from_dsdelt(ds, delta_x)

    if 'c' in gen_panels:
        dd_ax = f.add_subplot(dxdeltad_grid)
        
        gpl.plot_trace_werr(delta_x, dx, ax=dd_ax, label=r'$D_{X}$',
                            color=r_colors[0])
        gpl.plot_trace_werr(delta_x, dy, ax=dd_ax, label=r'$D_{Y}$',
                            log_y=True, color=r_colors[1])
        xl = dd_ax.get_xlim()
        dd_ax.hlines(ds, *xl, color='k', linestyle='dashed')
        dd_ax.set_xlabel(r'$\Delta D$')
        dd_ax.set_ylabel('MSE')

    delta_d_inds = np.array((0, 550, -1))
    if 'd' in gen_panels:
        dae_ax = f.add_subplot(distae_grid)
        sum_d = (dx + dy)/2
        distort = sum_d[delta_d_inds]
        delts = np.round(delta_x[delta_d_inds], 2)
        plot_distance_error_prob(dae_ax, distortions=distort, label_num=delts,
                                 color=n_colors[0])

    size = 100
    stim_count = (2,)
    pr = np.logspace(0, 4, 50)
    delta_d = np.array([0, .5, .9])
    ol_dims = (1,)
    n_ests = 5000
    if 'ef' in data.keys():
        aes, redund, pr, stim_count, ol_dims, delta_d, aes_est = data['ef']
    else:
        aes_est = estimate_empirical_aes(ol_dims, stim_count[0], pr, size,
                                         delta_d=delta_d,
                                         n_ests=n_ests)
        aes, redund = compute_ae_red(ol_dims, stim_count, pr, size,
                                     delta_d=delta_d)
        data['ef'] = (aes, redund, pr, stim_count, ol_dims, delta_d, aes_est)
    if 'ef' in gen_panels:
        ae_ax = f.add_subplot(ae_delt_grid)
        red_ax = f.add_subplot(red_delt_grid)
        
        ae_i = plt.axes([0, 0, 1, 1], aspect='equal')
        ip1 = InsetPosition(ae_ax, [0.65, 0.65, 0.35, 0.35]) 
        ae_i.set_axes_locator(ip1)
        ds = size/(pr[40]**2)
        plot_delta_xy(ds, delta_d, ae_i)
        plot_ae_error(ae_ax, red_ax, aes, redund, ol_dims, pr, delta_d=delta_d,
                      aes_est=aes_est)

    fname = os.path.join(bf, 'fig3-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure4(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'bc')
    if data is None:
        data = {}

    fsize = (4, 2)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(12, 12)

    rdb_grid = gs[:, 0:6]
    alloc_grid = gs[:, 6:10]
    alloc_sum_grid = gs[:, 10:12]

    size = 100
    if 'a' in gen_panels:
        rdb_ax = f.add_subplot(rdb_grid)
        plot_rdb(rdb_ax, size=size)

    k = 5
    c = 1
    bits = 20
    delta_d = .5
    unique_col = (.4, .1, .1)
    common_col = (.1, .4, .4)
    if 'bc' in gen_panels:
        ub, cbx, cby, ds, dx, dy = compute_bit_alloc(bits, k, c, delta_d,
                                                     size)
        unique_heights = (ub,)*(k - c)
        common_heights = (cbx, cby)
        unique_labels = tuple(r'$f_'+'{}$'.format(i + 1) for i in range(k - c))
        c_labels = ((r'$f^{X}_'+'{}$'.format(i + 1),
                     r'$f^{Y}_'+'{}$'.format(i + 1))
                    for i in range(k - c, k))
        common_labels = ()
        for com in c_labels:
            common_labels = common_labels + com
            
        alloc_ax = f.add_subplot(alloc_grid)
        alloc_ax.bar(range(k - c), unique_heights, color=unique_col)
        alloc_ax.bar(range(k - c, k + c), common_heights,
                     color=common_col)
        alloc_ax.set_xticks(range(k + c))
        alloc_ax.set_xticklabels(unique_labels + common_labels)
        alloc_ax.set_xlabel('feature bit allocation')
        alloc_ax.set_ylabel('information (bits)')
        gpl.clean_plot(alloc_ax, 0)

        as_ax = f.add_subplot(alloc_sum_grid, sharey=alloc_ax)
        as_ax.bar([0], [ub], color=unique_col)
        as_ax.bar([1], [cbx + cby], color=common_col)
        as_ax.set_xticks([0, 1])
        as_ax.set_xticklabels(['bits/unique', 'bits/common'], rotation=90)
        gpl.clean_plot(as_ax, 1)
        
    fname = os.path.join(bf, 'fig4-py.pdf')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure5(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('abcde', 'f')
    if data is None:
        data = {}

    fsize = (5, 4)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    ae_c_grid = gs[:25, 0:50]
    local_c_grid = gs[32:50, 0:50]
    ae_delt_grid = gs[:25, 55:]
    local_delt_grid = gs[32:50, 55:]
    
    info_by_local_grid = gs[70:, :40]
    
    ae_red_grid = gs[70:, 55:70]
    ld_red_grid = gs[70:, 85:]
    # total_grid = gs[56:, 50:]

    source_distrib = 'uniform'
    k = 5
    s = 100
    n_stim = 2
    bits = np.linspace(40, 60, 25)
    c_xys = np.array([1, 2, 3])
    delts = .99 - np.logspace(-4, 0, 20)
    if 'abcdef' in data.keys():
        out = data['abcdef']
        aes, evs, red, bits, k, s, n_stim, c_xys, delts, source_distrib = out
    else:
        out = am.ae_ev_bits(bits, k, s, n_stim, c_xys, delts,
                            source_distrib=source_distrib, compute_redund=True)
        aes, evs, red = out
        data['abcdef'] = (aes, evs, red, bits, k, s, n_stim, c_xys, delts,
                          source_distrib)
    """ c x delt x bits """
    if 'abcde' in gen_panels:
        aec_ax = f.add_subplot(ae_c_grid)
        lc_ax = f.add_subplot(local_c_grid, sharex=aec_ax)
        aed_ax = f.add_subplot(ae_delt_grid, sharey=aec_ax)
        ld_ax = f.add_subplot(local_delt_grid, sharex=aed_ax,
                              sharey=lc_ax)

        lae_ax = f.add_subplot(info_by_local_grid)
        
        delt_ind = -1
        bits_ind = 12
        color_incr = .0015

        lae_i = plt.axes([0, 0, 1, 1])
        ip1 = InsetPosition(lae_ax, [0.7, 0.7, 0.3, 0.3]) 
        lae_i.set_axes_locator(ip1)
        for i, c in enumerate(c_xys):
            gpl.plot_trace_werr(bits, aes[i, delt_ind], ax=aec_ax,
                                label='C = {}'.format(c), log_y=True)
            gpl.plot_trace_werr(bits, evs[i, delt_ind], ax=lc_ax,
                                log_y=True)
            gpl.plot_trace_werr(delts, aes[i, :, bits_ind], ax=aed_ax,
                                linestyle='dashed', log_y=True)
            gpl.plot_trace_werr(delts, evs[i, :, bits_ind], ax=ld_ax,
                                linestyle='dashed')

            aec_ax.set_ylabel('assignment\nerror rate')
            ld_ax.set_xlabel(r'asymmetry ($\Delta D$)')
            lc_ax.set_ylabel('estimator\nvariance '+r'$D_{S}$')
            lc_ax.set_xlabel('information (nats)')

            lae_ax.set_ylabel(r'estimator variance $D_{S}$')
            lae_ax.set_xlabel('assignment error rate')
            
            l = gpl.plot_trace_werr(aes[i, delt_ind], evs[i, delt_ind],
                                    ax=lae_ax, log_y=True, log_x=True)
            bi_color = l[0].get_color()
            for j, bi in enumerate(bits):
                bi_color = gpl.add_color(bi_color, color_incr*j)
                gpl.plot_trace_werr(aes[i, :, j], evs[i, :, j],
                                    ax=lae_ax, linestyle='dashed',
                                    color=bi_color)
            gpl.plot_trace_werr(aes[i, :, -1], evs[i, :, -1],
                                ax=lae_i, log_x=True, log_y=True)
            if c == max(c_xys):
                gpl.label_plot_line(aes[i, delt_ind], evs[i, delt_ind],
                                    r'$\leftarrow$ info', ax=lae_ax)
                gpl.label_plot_line(aes[i, :, 0][::-1], evs[i, :, 0][::-1],
                                    r'asymmetry $\rightarrow$',
                                    ax=lae_ax, buff=5)

    bit_ind = -10
    if 'f' in gen_panels:
        ae_red_ax = f.add_subplot(ae_red_grid)
        ld_red_ax = f.add_subplot(ld_red_grid)
        ae_red_ax.set_xlabel('redundancy (nats)')
        ae_red_ax.set_ylabel('assignment\nerror rate')
        ld_red_ax.set_ylabel('estimator\nvariance '+r'$D_{S}$')
        for i, c in enumerate(c_xys):
            r = red[i, :, bit_ind]
            ae = aes[i, :, bit_ind]
            ld = evs[i, :, bit_ind]
            gpl.plot_trace_werr(r, ae, log_y=True, ax=ae_red_ax)
            gpl.plot_trace_werr(r, ld, log_y=True, ax=ld_red_ax)

    # if 'f' in gen_panels:
    #     tot_ax = f.add_subplot(total_grid)
    #     tot_ax.set_xlabel('information (nats)')
    #     tot_ax.set_ylabel('total distortion')
    #     for i, c in enumerate(c_xys):
    #         lam = am.mse_weighting(k, c, s, source_distrib=source_distrib)
    #         total_err = evs[i] + aes[i]*lam
    #         gpl.plot_trace_werr(bits, total_err[0], ax=tot_ax, log_y=True)

    gpl.clean_plot(ld_ax, 1)
    gpl.clean_plot(aed_ax, 1)
    fname = os.path.join(bf, 'fig5-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def _plot_rfs(rf_cents, rf_wids, ax, scale=(0, 1), thin=5, color=None,
              plot_dots=False, make_scales=True):
    cps = u.get_circle_pts(100, 2)

    for i, rfc in enumerate(rf_cents[::thin]):
        rfw = 10*np.sqrt(rf_wids[i])
        l = ax.plot(cps[:, 0]*rfw[0] + rfc[0],
                    cps[:, 1]*rfw[1] + rfc[1],
                    color=color,
                    linewidth=1)
        if plot_dots:
            ax.plot(rfc[0], rfc[1], 'o',
                    color=l[0].get_color())
    if make_scales:
        gpl.make_xaxis_scale_bar(ax, 1, label='dimension 1', double=False)
        gpl.make_yaxis_scale_bar(ax, 1, label='dimension 2', double=False)
        gpl.clean_plot(ax, 0)

def figure_fi(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'bc', 'de')
    if data is None:
        data = {}

    fsize = (5, 4)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    schem_grid = gs[:33, :33]
    mse_nu_grid = gs[:33, 40:65]
    ae_nu_grid = gs[:33, 75:]

    iso_grid = gs[40:, :33]
    iso_inset_grid = gs[42:70, 20:33]
    map_grid = gs[40:, 50:90]
    map_cb_grid = gs[60:80, 95:]

    if data.get('a') is None and 'a' in gen_panels:
        rf_distrs = (sts.uniform(0, 1),)*2
        n_units = 1000
        rf_cents, rf_wids = rfm.get_random_uniform_fill(n_units, rf_distrs)
        data['a'] = (rf_cents, rf_wids)

    schem_ax = f.add_subplot(schem_grid)
    if 'a' in gen_panels:
        rf_cents, rf_wids = data['a']
        _plot_rfs(rf_cents, rf_wids, schem_ax)

    overlaps = (1, 2)
    n_n_units = 20
    n_units = np.logspace(2, 3, n_n_units, dtype=int)
    if data.get('bc') is None and 'bc' in gen_panels:
        total_feats = 2
        pwr = 10
        fi_theor = np.zeros((len(overlaps), len(n_units)))
        fi_emp = np.zeros_like(fi_theor)
        ae_theor = np.zeros_like(fi_theor)
        for i, ov in enumerate(overlaps):
            n_feats = total_feats + ov
            rf_distrs = (sts.uniform(0, 1),)*n_feats
            stim_distr = u.MultivariateUniform(n_feats, (0, 1))
            for j, nu in enumerate(n_units):
                fi, fi_var, _, w, _ = rfm.max_fi_power(pwr, nu, n_feats)
                fi_theor[i, j] = fi[0, 0]
            
                rf_cents, rf_wids = rfm.get_random_uniform_fill(nu, rf_distrs)
                use_wids = np.ones_like(rf_wids)*w**2
                fi, pwr, distortion = am.compute_fi(rf_cents, use_wids, pwr,
                                                    stim_distr)
                fi_emp[i, j] = fi

                r_distort = np.array([1/fi_theor[i, j]])
                ae_theor[i, j] = am.integrate_assignment_error(
                    (2,), r_distort, r_distort, ov, p=1)

        data['bc'] = (fi_emp, fi_theor, ae_theor)

    mse_nu_ax = f.add_subplot(mse_nu_grid)
    ae_nu_ax = f.add_subplot(ae_nu_grid)
    if 'bc' in gen_panels:
        fi_emp, fi_theor, ae_theor = data['bc']
        for i, ov in enumerate(overlaps):
            l = gpl.plot_trace_werr(n_units, 1/fi_emp[i], ax=mse_nu_ax,
                                    log_x=True, log_y=True)
            gpl.plot_trace_werr(n_units, 1/fi_theor[i], ax=mse_nu_ax, log_x=True,
                                log_y=True, color=l[0].get_color(),
                                linestyle='dashed')
            gpl.plot_trace_werr(n_units, ae_theor[i], ax=ae_nu_ax, log_x=True,
                                log_y=True)

    data_path = 'assignment/many_mse_tradeoffs-nr3.pkl'
    if data.get('de') is None and 'de' in gen_panels:
        out = pickle.load(open(data_path, 'rb'))
        data['de'] = out

    iso_ax = f.add_subplot(iso_grid)
    iso_inset_ax = f.add_subplot(iso_inset_grid)
    map_ax = f.add_subplot(map_grid)
    map_cb_ax = f.add_subplot(map_cb_grid)
    if 'de' in gen_panels:
        ax_vals, total_mse, mse_dist, ae_rate = data['de'][:4]
        total_units, n_feats, overlaps, total_pwrs = ax_vals
        # indices are (units, feats, overlaps, pwrs)
        pwr_ind = 40
        feat_ind = 3
        print('nf', n_feats[feat_ind])
        var = 1/6
        use_regions = (1, 2)
        td_thresh = .01
        region_list = []
        min_maps = []

        linestyles = ('dashed', 'solid')
        
        for k, mse_r in mse_dist.items():            
            ae_r = ae_rate[k]
            td_maps = []
            rep_feats = []
            for i, ov in enumerate(overlaps):
                total_rep_feats = n_feats[feat_ind] + (k - 1)*ov
                region_feats = am._split_integer(total_rep_feats, k)
                print(region_feats)
                smaller_feat = min(region_feats)
                rep_feats.append(smaller_feat)
                mse_i = 2*mse_r*n_feats[feat_ind]
                ae_distortion = (2*smaller_feat*var
                                 + mse_i)
                spec_mse = mse_i*(1 - ae_r)
                spec_ae_dist = ae_distortion*ae_r
                if k in use_regions:
                    ls = linestyles[use_regions.index(k)]
                    l = gpl.plot_trace_werr(spec_mse[:, feat_ind, i, pwr_ind],
                                            spec_ae_dist[:, feat_ind, i, pwr_ind],
                                            ax=iso_ax, label='', # 'C = {}'.format(ov),
                                            points=False, markersize=3, ls=ls)
                    col = l[0].get_color()
                    td_i = (spec_mse[:, feat_ind, i, pwr_ind]
                            + spec_ae_dist[:, feat_ind, i, pwr_ind])
                    td_mask = td_i < td_thresh
                    if np.any(td_mask):
                        td_pt = total_units[td_mask][0]
                    else:
                        td_pt = np.nan
                    iso_inset_ax.plot([ov], [td_pt], 'o', color=col)

                td_map = total_mse[k][:, feat_ind, i, :]
                td_maps.append(td_map)
            full_map_k = np.stack(td_maps, axis=0)
            min_map_k = np.min(full_map_k, axis=0)
            min_maps.append(min_map_k)
            region_list.append(k)
        min_regions = np.stack(min_maps, axis=0)
        num_regions = np.argmin(min_regions, axis=0)
        plot_map = np.array(region_list)[num_regions]
        cmap_name = None
        cmap = plt.get_cmap(cmap_name)
        cm = gpl.pcolormesh(total_pwrs, total_units, plot_map, ax=map_ax,
                            cmap=cmap)
        possibles = np.unique(plot_map)
        if len(possibles) > 1:
            delta = np.diff(possibles)[0]
            bounds = np.concatenate((possibles - delta/2,
                                     [possibles[-1] + delta/2]))
        else:
            bounds = (possibles[0] - .5, possibles[0] + .5)
        colors = cmap(possibles/np.max(possibles))
        mappable = mpl.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(colors))
        f.colorbar(mappable, cax=map_cb_ax,
                   boundaries=bounds, ticks=possibles,
                   label='number of regions')
        
        
        map_ax.set_xscale('log')
        map_ax.set_yscale('log')
        map_ax.set_xlabel('total power')
        map_ax.set_ylabel('total units')
        map_ax.set_title('K = {}'.format(n_feats[feat_ind]))
        
        # iso_ax.set_aspect('equal')
        iso_ax.set_xscale('log')
        rads = [.01, .02]
        labels = ['constant\ntotal distortion', '']
        for i, rad in enumerate(rads):
            pts = np.linspace(0, np.pi/2, 100)
            y = rad*np.sin(pts)
            x = rad*np.cos(pts)
            # iso_ax.plot(x, y, linestyle='dashed', color='k',
            #             label=labels[i])
        # iso_ax.legend(frameon=False)
        iso_ax.set_xlabel('local MSE')
        iso_ax.set_ylabel('assignment MSE')

        gpl.clean_plot(iso_inset_ax, 0)
        iso_inset_ax.set_yscale('log')
        iso_inset_ax.set_xlabel('C')
        iso_inset_ax.set_ylabel('units required for\n'
                                'total distortion < {}'.format(td_thresh))


    # plot transition map between two and three regions
    # for final piece
    
    return data

def figure6_alt(basefolder=bf, mp1=None, mp2=None, mp3=None, gen_panels=None,
                data=None, experiment_key='Zhang & Luck 2008'): 
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'e', 'f', 'comp')
    if data is None:
        data = {}

    fsize = (7.5, 6)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    schem_grid = gs[:20, 0:65]
    mod_dist_grid = gs[:26, 70:75]
    mod_ae_grid = gs[:26, 82:87]
    mod_uni_grid = gs[:26, 94:]    
    n_participants = 8
    pred_loads = []
    pred_colors = []
    pred_ppc = []
    spacing = np.linspace(0, 100, n_participants + 1)
    buff = 1
    for i in range(n_participants):
        beg = int(np.round(spacing[i] + buff))
        end = int(np.round(spacing[i+1] - buff))
        pred_ppc.append(gs[26:44, beg:end])
        pred_loads.append(gs[54:72, beg:end])
        pred_colors.append(gs[82:, beg:end])

    abort = False
    if mp1 is None or mp2 is None:
        print('-------------------------------------------------')
        print('this figure relies on model fits that are missing')
        print('the code for fitting the models is included here,')
        print('however it is computationally expensive and so is')
        print('not run as part of this function')
        abort = True
        
    if 'a' in gen_panels:
        schem_ax = f.add_subplot(schem_grid)

    if not abort:
        if 'def' not in data.keys():
            m1, p1, diag1 = da.load_model(mp1)
            m2, p2, diag2 = da.load_model(mp2)
            m1_dict = {experiment_key:m1}
            d1_dict = {experiment_key:p1}
            data['def'] = (d1_dict, m1_dict)
            data['c'] = [m1]
            comp = ((m1, diag1), (m2, diag2))
            if mp3 is not None:
                m3, _, diag3 = da.load_model(mp3)
                comp = comp + ((m3, diag3),)
            data['comp'] = comp

        if 'comp' in gen_panels:
            m_dict = {}
            if mp3 is not None:
                ((m1, diag1), (m2, diag2), (m3, diag3)) = data['comp']
            else:
                ((m1, diag1), (m2, diag2)) = data['comp']
            m_dict = {'full':m1.arviz, 'uni':m2.arviz}
            if mp3 is not None:
                m_dict['assign'] = m3.arviz
            out = av.compare(m_dict)
            print(out)
        
        data_color = (.7, .7, .7)
        ct = np.nanmean
        ef = gpl.conf95_interval
        boots = 200
        
        dist_ax = f.add_subplot(mod_dist_grid)
        ae_ax = f.add_subplot(mod_ae_grid)
        uni_ax = f.add_subplot(mod_uni_grid, sharey=ae_ax)
        if 'c' in gen_panels:
            m1_list = data['c']
            plot_stan_model(m1_list, ae_ax, dist_ax, uni_ax)

        set_xs = [2, 5]
        ax1 = f.add_subplot(pred_loads[0])
        axs_load = list(f.add_subplot(pl, sharex=ax1, sharey=ax1)
                        for pl in pred_loads[1:])
        axs_load = [ax1] + axs_load
        if 'd' in gen_panels:
            d1_dict, m1_dict = data['def']
            mse_dict = da.model_subj_org(d1_dict)
            mod_dict = da.model_subj_org(d1_dict, model=m1_dict)
            up = (f, axs_load)
            _ = da.plot_experiment_func(mse_dict, log_y=False,
                                        data_color=data_color, 
                                        central_tendency=ct, legend=False,
                                        error_func=ef,  plot_fit=False,
                                        use_plot=up, use_same_ax=False,
                                        boots=boots, sep_subj=True,
                                        model_data=mod_dict,
                                        set_xs=set_xs)  

            gpl.clean_plot(axs_load[0], 0)
            [gpl.clean_plot(al, 1) for al in axs_load[1:]]

        ax1 = f.add_subplot(pred_colors[0])
        axs_col = list(f.add_subplot(pl, sharex=ax1, sharey=ax1)
                        for pl in pred_colors[1:])
        axs_col = [ax1] + axs_col
        up = (f, axs_col)
        if 'e' in gen_panels:
            d1_dict, m1_dict = data['def']
            mld_dict = da.model_subj_org(d1_dict, org_func=da.mse_by_dist)
            modd_dict = da.model_subj_org(d1_dict, org_func=da.mse_by_dist,
                                          model=m1_dict)

            _ = da.plot_experiment_func(mld_dict,
                                        plot_func=da.plot_dist_dependence,
                                        x_ax='color distance (radians)',
                                        log_y=False,
                                        data_color=data_color,
                                        central_tendency=ct, legend=False,
                                        n_bins=5, need_trials=20,
                                        error_func=ef, use_plot=up,
                                        plot_fit=True, use_same_ax=False,
                                        boots=boots, sep_subj=True,
                                        model_data=modd_dict,
                                        set_xs=set_xs)
            gpl.clean_plot(axs_col[0], 0)
            [gpl.clean_plot(al, 1) for al in axs_col[1:]]
            
        ax1 = f.add_subplot(pred_ppc[0])
        axs_ppc = list(f.add_subplot(pl, sharex=ax1, sharey=ax1)
                        for pl in pred_ppc[1:])
        axs_ppc = [ax1] + axs_ppc
        up = (f, axs_ppc)
        if 'f' in gen_panels:
            d1_dict, m1_dict = data['def']
            mld_dict = da.model_subj_org(d1_dict)
            modd_dict = da.model_subj_org(d1_dict, model=m1_dict)

            _ = da.plot_experiment_func(mld_dict,
                                        plot_func=da.plot_subj_ppc,
                                        x_ax='error (radians)',
                                        log_y=False,
                                        data_color=data_color,
                                        central_tendency=ct, legend=False,
                                        n_bins=5, need_trials=20,
                                        error_func=ef, use_plot=up,
                                        plot_fit=True, use_same_ax=False,
                                        boots=boots, sep_subj=True,
                                        model_data=modd_dict,
                                        y_ax='density',
                                        set_xs=set_xs)
            gpl.clean_plot(axs_ppc[0], 0)
            [gpl.clean_plot(al, 1) for al in axs_ppc[1:]]

    fname = os.path.join(bf, 'fig6-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data


def figure6(basefolder=bf, datapath1=None, modelpath1=None,
            modelpath2=None, m_pattern1=None, m_pattern2=None,
            gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'de', 'f')
    if data is None:
        data = {}

    fsize = (7.5, 5)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    schem_grid = gs[:28, 0:65]
    mod_dist_grid = gs[:35, 78:83]
    mod_ae_grid = gs[:35, 90:95]
    n_participants = 8
    pred_loads = []
    pred_colors = []
    spacing = np.linspace(0, 100, n_participants + 1)
    buff = 1
    for i in range(n_participants):
        beg = int(np.round(spacing[i] + buff))
        end = int(np.round(spacing[i+1] - buff))
        pred_loads.append(gs[35:65, beg:end])
        pred_colors.append(gs[75:, beg:end])

    abort = False
    if datapath1 is None:
        print('-------------------------------------------------')
        print('this figures relies on data that is missing')
        abort = True
        print('missing data can be downloaded from: ')

    if modelpath1 is None or modelpath2 is None:
        print('-------------------------------------------------')
        print('this figure relies on model fits that are missing')
        print('the code for fitting the models is included here,')
        print('however it is computationally expensive and so is')
        print('not run as part of this function')
        abort = True
    if modelpath1 is None:
        print('missing model fits can be generated by running: ')
    if modelpath2 is None:
        print('missing model fits can be generated by running: ')
        
    if 'a' in gen_panels:
        schem_ax = f.add_subplot(schem_grid)

    if not abort:
        models1, funcs1 = da.load_models(modelpath1, m_pattern1)
        
        data_color = (.7, .7, .7)
        ct = np.nanmean
        ef = gpl.conf95_interval
        boots = 1000
        n_samples = 100
        experiment_keys = ('Zhang & Luck 2008',)

        dist_ax = f.add_subplot(mod_dist_grid)
        ae_ax = f.add_subplot(mod_ae_grid)
        if 'c' in gen_panels:
            plot_stan_model(models1[experiment_keys[0]], ae_ax, dist_ax)

        if 'de' in gen_panels:
            if 'de' not in data.keys():
                data1 = da.load_data(datapath1, sort_pos=True,
                                     collapse_subs=False,
                                     dict_convert=True,
                                     keep_experiments=experiment_keys)
                m_data1 = da.simulate_data(data1, funcs1, n_samples=n_samples)
                data['de'] = (data1, m_data1)
            else:
                data1, m_data1 = data['de']

            ml_dict = da.experiment_subj_org(data1)
            mod_dict = da.experiment_subj_org(m_data1)
            ax1 = f.add_subplot(pred_loads[0])
            axs_load = list(f.add_subplot(pl, sharex=ax1, sharey=ax1)
                            for pl in pred_loads[1:])
            axs_load = [ax1] + axs_load
            up = (f, axs_load)
            _ = da.plot_experiment_func(ml_dict, log_y=False,
                                        data_color=data_color, 
                                        central_tendency=ct, legend=False,
                                        error_func=ef,  plot_fit=False,
                                        use_plot=up, use_same_ax=False,
                                        boots=boots, sep_subj=True,
                                        model_data=mod_dict)  

            gpl.clean_plot(axs_load[0], 0)
            [gpl.clean_plot(al, 1) for al in axs_load[1:]]
            
            mld_dict = da.experiment_subj_org(data1, org_func=da.mse_by_dist)
            modd_dict = da.experiment_subj_org(m_data1, org_func=da.mse_by_dist)

            ax1 = f.add_subplot(pred_colors[0])
            axs_load = list(f.add_subplot(pl, sharex=ax1, sharey=ax1)
                            for pl in pred_colors[1:])
            axs_col = [ax1] + axs_load
            up = (f, axs_col)
            _ = da.plot_experiment_func(mld_dict,
                                        plot_func=da.plot_dist_dependence,
                                        x_ax='color distance (radians)',
                                        log_y=False,
                                        data_color=data_color,
                                        central_tendency=ct, legend=False,
                                        n_bins=5, need_trials=20,
                                        error_func=ef, use_plot=up,
                                        plot_fit=True, use_same_ax=False,
                                        boots=boots, sep_subj=True,
                                        model_data=modd_dict)
            gpl.clean_plot(axs_col[0], 0)
            [gpl.clean_plot(al, 1) for al in axs_col[1:]]
            
    fname = os.path.join(bf, 'fig6-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure7(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('ab', 'c', 'def', 'd', 'ef', 'e', 'f')
    if data is None:
        data = {}

    fsize = (7.5, 5)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    eg_inp1 = gs[:20, :20]
    eg_inp2 = gs[25:45, :20]
    eg_hidden = gs[:45, 25:55]
    eg_integ_corr = gs[:20, 65:85]
    eg_integ_err = gs[25:45, 65:85]

    train_12 = gs[:20, 90:]
    train_23 = gs[25:45, 90:]

    dist_match = gs[55:, :30]
    asym_match = gs[55:, 36:63]
    over_match = gs[55:, 69:]

    p = 10
    input_dists = (sts.uniform(0, p), sts.uniform(0, p), sts.uniform(0, p))
    f1_inds = (1, 0)
    f2_inds = (2, 0)
    recon_inds = (1, 2)

    f1_units = 30
    f2_units = 30
    recon_units = 30
    hu_units = (900, 900)
    train_size = 10e4

    model_args = (f1_units, f2_units, recon_units, input_dists, 
                  f1_inds, f2_inds, recon_inds)
    model_kwargs = {'hu_units':hu_units}
    
    if 'ab' not in data.keys() and 'ab' in gen_panels:
        m = ff.IntegrationModel(*model_args, **model_kwargs)
        data['ab'] = m

    eg_dists = ((0, 4), (1, 2), (2, 7))
    eg_stim = 2

    eg_i1_ax = f.add_subplot(eg_inp1, aspect='equal')
    eg_i2_ax = f.add_subplot(eg_inp2, aspect='equal')
    eg_h_ax = f.add_subplot(eg_hidden)
    eg_corr_ax = f.add_subplot(eg_integ_corr, aspect='equal')
    eg_err_ax = f.add_subplot(eg_integ_err, aspect='equal')
    if 'ab' in gen_panels:
        m = data['ab']
        out = m.random_example(eg_stim, make_others=True, set_dists=eg_dists,
                               noise_mag=0, topogriphy=True)
        f1, f2, y, y_hat, inp, ys_all, ds = out
        f1_c, f2_c, o_c = m.get_unique_cents()
        f1_x = gpl.pcolormesh_axes(f1_c[0], len(f1_c[0]))
        f1_y = gpl.pcolormesh_axes(f1_c[1], len(f1_c[1]))
        eg_i1_ax.pcolormesh(f1_x, f1_y, f1)

        f2_x = gpl.pcolormesh_axes(f2_c[0], len(f2_c[0]))
        f2_y = gpl.pcolormesh_axes(f2_c[1], len(f2_c[1]))
        eg_i2_ax.pcolormesh(f2_x, f2_y, f2)

        o_x = gpl.pcolormesh_axes(o_c[0], len(o_c[0]))
        o_y = gpl.pcolormesh_axes(o_c[1], len(o_c[1]))
        eg_corr_ax.pcolormesh(o_x, o_y, ys_all[0])
        eg_err_ax.pcolormesh(o_x, o_y, ys_all[1])

    n_epochs = 20
    train_size_th = 10e3
    if 'c' not in data.keys() and 'c' in gen_panels:
        m12 = ff.IntegrationModel(*model_args, **model_kwargs)
        h12 = m12.train(1, validate_n=2, train_size=train_size_th,
                        epochs=n_epochs)
        
        m23 = ff.IntegrationModel(*model_args, **model_kwargs)
        h23 = m23.train(2, validate_n=3, train_size=train_size_th,
                        epochs=n_epochs)
        data['c'] = (h12, h23)

    t12_ax = f.add_subplot(train_12)
    t23_ax = f.add_subplot(train_23, sharex=t12_ax, sharey=t12_ax)
    if 'c' in gen_panels:
        h12, h23 = data['c']

        x_epochs = np.arange(1, n_epochs + 1)
        gpl.plot_trace_werr(x_epochs, h12.history['loss'], ax=t12_ax)
        gpl.plot_trace_werr(x_epochs, h12.history['val_loss'], ax=t12_ax)
        gpl.plot_trace_werr(x_epochs, h23.history['loss'], ax=t23_ax)
        gpl.plot_trace_werr(x_epochs, h23.history['val_loss'], ax=t23_ax)


    n_stim = 2
    train_size = 10e4
    if 'def' not in data.keys() and 'def' in gen_panels:
        m = ff.IntegrationModel(*model_args, **model_kwargs)
        m.train(n_stim, train_size=train_size)
        data['def'] = m

    n_stim = 2
    n_est = 10e3
    noise_mags = np.linspace(.1, 1, 3)
    dists = np.linspace(.2, 3, 20)
    use_dim = 0
    excl_close = 2
    boots = 1000
    if 'd' not in data.keys() and 'd' in gen_panels:
        m = data['def']
        pairs = []
        for nm in noise_mags:
            dist_rates = m.estimate_ae_rate_dists(n_stim, dists, nm,
                                                  n_est=n_est, dim=use_dim,
                                                  excl_close=excl_close,
                                                  boots=boots)
            dist_theor = am.distance_error_rate(dists, None, .5*(nm**2))
            pairs.append((dist_rates, dist_theor))
        data['d'] = pairs

    dist_m_ax = f.add_subplot(dist_match)
    if 'd' in gen_panels:
        pairs = data['d']
        for i, (dist_rates, dist_theor) in enumerate(pairs):
            nm = noise_mags[i]
            l = gpl.plot_trace_werr(dists, dist_rates.T, ax=dist_m_ax,
                                    error_func=gpl.conf95_interval,
                                    label='D = {}'.format(nm))
            col = l[0].get_color()
            gpl.plot_trace_werr(dists, dist_theor[0], ax=dist_m_ax,
                                color=col, linestyle='dashed')

    # if 'ef' not in data.keys() and 'ef' in gen_panels:
    #     input_dists = (sts.uniform(0, p),)*4
    #     f1_inds = (3, 1, 0)
    #     f2_inds = (2, 1, 0)
    #     recon_inds = (3, 2)
        
    #     f1_units = 20
    #     f2_units = 20
    #     recon_units = 30
    #     hu_units = (900, 900)
    #     train_size = 10e4

    #     model_args = (f1_units, f2_units, recon_units, input_dists, 
    #                   f1_inds, f2_inds, recon_inds)
    #     model_kwargs = {'hu_units':hu_units}
    #     m = ff.IntegrationModel(*model_args, **model_kwargs)
    #     m.train(n_stim, train_size=train_size)
    #     data['ef'] = (data['def'], m), (1, 2)

    asyms = np.linspace(0, .9, 3)
    overall_d = .5
    if 'e' not in data.keys() and 'e' in gen_panels:
        m = data['def']
        pairs = []
        for a in asyms:
            d1, d2 = am.d_func(a, overall_d)
            nm = (np.sqrt(d1), np.sqrt(d2))
            dist_rates = m.estimate_ae_rate_dists(n_stim, dists, nm,
                                                  n_est=n_est, dim=use_dim,
                                                  excl_close=excl_close,
                                                  boots=boots)
            dist_theor = am.distance_error_rate(dists, np.array([a]),
                                                np.array([overall_d]))
            pairs.append((dist_rates, dist_theor))
        data['e'] = pairs

    dist_m_ax = f.add_subplot(asym_match)
    if 'e' in gen_panels:
        pairs = data['e']
        for i, (dist_rates, dist_theor) in enumerate(pairs):
            nm = noise_mags[i]
            l = gpl.plot_trace_werr(dists, dist_rates.T, ax=dist_m_ax,
                                    error_func=gpl.conf95_interval,
                                    label='D = {}'.format(nm))
            col = l[0].get_color()
            gpl.plot_trace_werr(dists, dist_theor[0], ax=dist_m_ax,
                                color=col, linestyle='dashed')    
            
    noise_mags = np.linspace(.1, 1, 50)
    n_est = 10e4
    ns = (2, 3, 4)
    if 'f' not in data.keys() and 'f' in gen_panels:
        m = data['def']
        pairs = []
        srs = p/noise_mags
        for i, n in enumerate(ns):
            noise_rates = m.estimate_ae_rate_noise(n, noise_mags, n_est=n_est,
                                                   excl_close=excl_close)
            ds = noise_mags**2
            theor_rates = am.integrate_assignment_error((n,), ds, ds, 1,
                                                        p=p)
            pairs.append((noise_rates, theor_rates, n))
        data['f'] = srs, pairs

    ae_rate_ax = f.add_subplot(over_match)
    if 'f' in gen_panels:
        srs, pairs = data['f']
        for (nr, tr, n) in pairs:
            l = gpl.plot_trace_werr(srs, nr.T, ax=ae_rate_ax,
                                    error_func=gpl.conf95_interval,
                                    log_x=True, log_y=True)
            col = l[0].get_color()
            gpl.plot_trace_werr(srs, tr, ax=ae_rate_ax, color=col)
            
    fname = os.path.join(bf, 'fig7-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data
