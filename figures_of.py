
import assignment.overlapping_features as am
import assignment.data_analysis as da
import general.plotting as gpl
import general.plotting_styles as gps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import scipy.stats as sts

from assignment.figure_helpers import *

bf = ('/Users/wjj/Dropbox/research/uc/freedman/analysis/'
              'mixedselectivity_theory/figs/')
colors = gps.nms_colors
n_colors = gps.assignment_num_colors

def setup():
    gps.set_paper_style(colors)
    
def figure1(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'e', 'f')
    if data is None:
        data = {}
    fsize = (6, 3.5)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(10, 10)

    schem_grid = gs[0:6, 0:5]
    overlap_eg_1_grid = gs[0:3, 5:7]
    overlap_eg_2_grid = gs[3:6, 5:7]
    noise_eg_grid = gs[0:6, 7:]
    distance_error_grid = gs[6:, 0:3]
    distance_distrib_grid = gs[6:, 3:6]
    assignment_error_grid = gs[6:, 6:]

    if 'a' in gen_panels:
        schem_ax = f.add_subplot(schem_grid)

    if 'b' in gen_panels:
        ol_ax_1 = f.add_subplot(overlap_eg_1_grid)
        ol_ax_2 = f.add_subplot(overlap_eg_2_grid, sharex=ol_ax_1)
        plot_eg_overlap(ol_ax_1, ol_ax_2)

    if 'c' in gen_panels:
        noise_ax = f.add_subplot(noise_eg_grid)

    if 'd' in gen_panels:
        de_ax = f.add_subplot(distance_error_grid)
        plot_distance_error_prob(de_ax, color_st=n_colors[0])

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

    return data

def figure2(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'de')
    if data is None:
        data = {}

    fsize = (6, 3.5)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(12, 12)

    v_mid = 5
    schem_grid = gs[0:v_mid, 0:4]
    distance_distrib_grid = gs[0:v_mid, 4:8]
    distance_prob_grid = gs[0:v_mid, 8:]
    ae_c_grid = gs[v_mid:, 0:6]
    red_c_grid = gs[v_mid:, 6:]

    if 'a' in gen_panels:
        schem_ax = f.add_subplot(schem_grid)

    ol_dims = (1, 2, 3)
    ol_funcs = [am.line_picking_line, am.line_picking_square,
                am.line_picking_cube]
    if 'b' in gen_panels:
        distrib_ax = f.add_subplot(distance_distrib_grid)
        plot_distance_distrib_prob(distrib_ax, overlap_dims=ol_dims,
                                   funcs=ol_funcs)

    ds = (10,)
    size = 100
    if 'c' in gen_panels:
        ep_ax = f.add_subplot(distance_prob_grid)
        out = plot_distance_error_prob(ep_ax, distortions=ds,
                                       color_st=n_colors[0],
                                       distance_bounds=(20, size - 20),
                                       label=False)
        ep_ax.set_yscale('log')
        curve, dists = out
        plot_distance_dim_means(ep_ax, ol_dims, ol_funcs, size, curve[0],
                                dists, colors=colors)

    stim_count = (2,)
    pr = np.logspace(0, 4, 50)
    if 'de' in data.keys():
        aes, redund, pr, stim_count, ol_dims = data['de']
    else:
        aes, redund = compute_ae_red(ol_dims, stim_count, pr, size)
        data['de'] = (aes, redund, pr, stim_count, ol_dims)
    if 'de' in gen_panels:
        ae_ax = f.add_subplot(ae_c_grid)
        red_ax = f.add_subplot(red_c_grid)
        plot_ae_error(ae_ax, red_ax, aes, redund, ol_dims, pr)
    return data

def figure3(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'ef')
    if data is None:
        data = {}

    fsize = (7.5, 2.5)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(12, 12)

    schem_grid = gs[0:6, 0:2]
    dxdy_grid = gs[6:, 0:2]
    dxdeltad_grid = gs[0:6, 2:4]
    distae_grid = gs[6:, 2:4]
    ae_delt_grid = gs[:, 4:8]
    red_delt_grid = gs[:, 8:]

    if 'a' in gen_panels:
        schem_ax = f.add_subplot(schem_grid)

    if 'b' in gen_panels:
        delta_eg = np.linspace(-.9, .9, 1000)
        ds = 10
        dx, dy = am.dxdy_from_dsdelt(ds, delta_eg)
        
        dxdy_ax = f.add_subplot(dxdy_grid, aspect='equal')

        gpl.plot_trace_werr(dx, dy, ax=dxdy_ax, log_y=True, log_x=True)
        dxdy_ax.plot(2*ds, 2*ds, 'o')
        xl = dxdy_ax.get_xlim()
        yl = dxdy_ax.get_ylim()
        dxdy_ax.hlines(ds, *xl, color='k', linestyle='dashed')
        dxdy_ax.vlines(ds, *yl, color='k', linestyle='dashed')
        # dxdy_ax.set_xlim(xl)
        # dxdy_ax.set_ylim(yl)
        dxdy_ax.set_xlabel(r'$D_{X}$')
        dxdy_ax.set_ylabel(r'$D_{Y}$')

    delta_x = np.linspace(0, .9, 1000)
    ds = 10
    dx, dy = am.dxdy_from_dsdelt(ds, delta_x)

    if 'c' in gen_panels:
        dd_ax = f.add_subplot(dxdeltad_grid)
        
        gpl.plot_trace_werr(delta_x, dx, ax=dd_ax, label=r'$D_{X}$')
        gpl.plot_trace_werr(delta_x, dy, ax=dd_ax, label=r'$D_{Y}$',
                            log_y=True)
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
                                 color_st=n_colors[0])

    size = 100
    stim_count = (2,)
    pr = np.logspace(0, 4, 50)
    delta_d = np.array([0, .5, .9])
    ol_dims = (1,)
    if 'ef' in data.keys():
        aes, redund, pr, stim_count, ol_dims, delta_d = data['ef']
    else:
        aes, redund = compute_ae_red(ol_dims, stim_count, pr, size,
                                     delta_d=delta_d)
        data['ef'] = (aes, redund, pr, stim_count, ol_dims, delta_d)
    if 'ef' in gen_panels:
        ae_ax = f.add_subplot(ae_delt_grid)
        red_ax = f.add_subplot(red_delt_grid)
        plot_ae_error(ae_ax, red_ax, aes, redund, ol_dims, pr, delta_d=delta_d)
        
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
    return data

def figure5(basefolder=bf, gen_panels=None, data=None):
    setup()
    if gen_panels is None:
        gen_panels = ('abcde', 'f')
    if data is None:
        data = {}

    fsize = (6, 5)
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(12, 12)

    ae_c_grid = gs[:3, 0:7]
    local_c_grid = gs[3:6, 0:7]
    ae_delt_grid = gs[:3, 7:]
    local_delt_grid = gs[3:6, 7:]
    info_by_local_grid = gs[6:, :6]
    total_grid = gs[6:, 6:]

    source_distrib = 'uniform'
    k = 5
    s = 100
    n_stim = 2
    bits = np.linspace(40, 50, 25)
    c_xys = np.array([1, 2, 3])
    # delts = np.linspace(0, .95, 20)
    delts = .99 - np.logspace(-4, 0, 20)
    if 'abcdef' in data.keys():
        out = data['abcdef']
        aes, evs, bits, k, s, n_stim, c_xys, delts, source_distrib = out
    else:
        out = am.ae_ev_bits(bits, k, s, n_stim, c_xys, delts,
                            source_distrib=source_distrib)
        aes, evs = out
        data['abcdef'] = (aes, evs, bits, k, s, n_stim, c_xys, delts,
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
        for i, c in enumerate(c_xys):
            gpl.plot_trace_werr(bits, aes[i, delt_ind], ax=aec_ax,
                                label='C = {}'.format(c), log_y=True)
            gpl.plot_trace_werr(bits, evs[i, delt_ind], ax=lc_ax,
                                log_y=True)
            gpl.plot_trace_werr(delts, aes[i, :, bits_ind], ax=aed_ax,
                                linestyle='dashed', log_y=True)
            gpl.plot_trace_werr(delts, evs[i, :, bits_ind], ax=ld_ax,
                                linestyle='dashed')

            aec_ax.set_ylabel('assignment error\nrate')
            ld_ax.set_xlabel('information (nats)')
            lc_ax.set_ylabel('MSE')
            lc_ax.set_xlabel('information (nats)')
            
            l = gpl.plot_trace_werr(aes[i, delt_ind], evs[i, delt_ind],
                                    ax=lae_ax, log_y=True, log_x=True)
            bi_color = l[0].get_color()
            for j, bi in enumerate(bits):
                bi_color = gpl.add_color(bi_color, color_incr*j)
                gpl.plot_trace_werr(aes[i, :, j], evs[i, :, j],
                                    ax=lae_ax, linestyle='dashed',
                                    color=bi_color)
            

    if 'f' in gen_panels:
        tot_ax = f.add_subplot(total_grid)
        for i, c in enumerate(c_xys):
            lam = am.mse_weighting(k, c, s, source_distrib=source_distrib)
            total_err = evs[i] + aes[i]*lam
            gpl.plot_trace_werr(bits, total_err[0], ax=tot_ax, log_y=True)
            
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
    f = plt.figure(figsize=fsize, constrained_layout=True)
    gs = f.add_gridspec(12, 12)

    schem_grid = gs[:6, 0:4]
    mc1_grid = gs[6:9, 0:4]
    mc2_grid = gs[9:, :4]
    n_participants = 8
    pred_loads = []
    pred_colors = []
    pred_dists = []
    for i in range(n_participants):
        pred_loads.append(gs[0:4, 4+i:4+i+1])
        pred_colors.append(gs[4:8, 4+i:4+i+1])
        pred_dists.append(gs[8:12, 4+i:4+i+1])

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
        models2, funcs2, data2 = da.load_spatial_models(modelpath2,
                                                        m_pattern2)

        if 'b' in gen_panels:
            mc1_ax = f.add_subplot(mc1_grid)

        if 'c' in gen_panels:
            mc2_ax = f.add_subplot(mc2_grid)

        data_color = (.7, .7, .7)
        ct = np.nanmean
        ef = gpl.conf95_interval
        boots = 5000
        experiment_keys = ('Zhang & Luck 2008',)
        if 'de' in gen_panels:
            if 'de' not in data.keys():
                data1 = da.load_data(datapath1, sort_pos=True,
                                     collapse_subs=False,
                                     dict_convert=True,
                                     keep_experiments=experiment_keys)
                m_data1 = da.simulate_data(data1, funcs1)
                data['de'] = (data1, m_data1)
            else:
                data1, m_data1 = data['de']

            ml_dict = da.experiment_subj_org(data1)
            mod_dict = da.experiment_subj_org(m_data1)
            axs_load = [f.add_subplot(pl) for pl in pred_loads]
            up = (f, axs_load)
            _ = da.plot_experiment_func(ml_dict, log_y=False, 
                                        data_color=data_color, 
                                        central_tendency=ct, legend=False,
                                        error_func=ef,  plot_fit=False,
                                        use_plot=up, use_same_ax=False,
                                        boots=boots, sep_subj=True,
                                        model_data=mod_dict)  

            mld_dict = da.experiment_subj_org(data1, org_func=da.mse_by_dist)
            modd_dict = da.experiment_subj_org(m_data1, org_func=da.mse_by_dist)
            
            axs_col = [f.add_subplot(pc) for pc in pred_colors]
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
            
        if 'f' in gen_panels:
            n_samples = 100
            if 'f' not in data.keys():
                m_data2 = da.simulate_data(data2, funcs2, spatial=True,
                                           n_samples=n_samples)
                data['f'] = m_data2
            else:
                m_data2 = data['f']
            df = 'stim_poss'
            spat_dict = da.experiment_subj_org(data2, dist_field=df, 
                                               org_func=da.mse_by_dist)
            spat_mod_dict = da.experiment_subj_org(m_data2, dist_field=df, 
                                                   org_func=da.mse_by_dist)

            axs_dist = [f.add_subplot(pd) for pd in pred_dists]
            up = (f, axs_dist)
            _ = da.plot_experiment_func(spat_dict,
                                        plot_func=da.plot_dist_dependence,
                                        x_ax='angular distance (radians)',
                                        log_y=False,
                                        data_color=data_color,
                                        central_tendency=ct, legend=False,
                                        n_bins=5, need_trials=20,
                                        error_func=ef, use_plot=up,
                                        plot_fit=True, use_same_ax=False,
                                        boots=boots, sep_subj=True,
                                        model_data=spat_mod_dict,
                                        model_boots=boots)
    return data
