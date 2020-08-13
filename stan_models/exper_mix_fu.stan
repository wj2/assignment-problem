functions {  
  real get_distortion(real bits, int n_stim) {
    real distortion;
    distortion = 2*pi()*exp(-2*bits/n_stim);
    return distortion;	  
  }

  real get_ae_probability(real bits, int n_stim, int max_stim, real spacing) {
    real distortion;
    real prob;
    real all_comb;
    real both;
    real either;
    distortion = get_distortion(bits, n_stim);
    prob = normal_lcdf(-spacing | 0, sqrt(2*distortion));
    all_comb = choose(max_stim - 1, n_stim - 1);
    if (n_stim >= 3) {
      both = choose(max_stim - 3, n_stim - 3)/all_comb;
    } else {
      both = 0;
    }
    if (n_stim >= 2) {
      either = 2*choose(max_stim - 3, n_stim - 2)/all_comb;
    } else {
      either = 0;
    }
    return prob + log_sum_exp(log(2*both), log(either));
  }

  real get_ae_lps(real err, vector stim_errs, real ld, real aep, int n_stim) {
    vector[n_stim] laps;
    laps[1] = log1m_exp(aep) + normal_lpdf(err | 0, ld);
    for (i in 2:n_stim) {
      laps[i] = (aep - log(n_stim - 1)
		 + normal_lpdf(stim_errs[i] | 0, ld));
    }
    return log_sum_exp(laps);
  }
  
  real get_lps(real err, vector stim_errs, real report_bits, real dist_bits,
	       real mech_dist, real mem_stim, int n_stim, int max_stim,
	       real spacing) {
    vector[n_stim + 1] lps;
    vector[2] per_mem;
    real cp;
    real plf;
    real ae_prob;
    real local_d;
    for (i in 0:n_stim) {
      plf = poisson_lpmf(i | mem_stim);
      if (i == n_stim) {
	plf = log_sum_exp(plf, poisson_lccdf(i | mem_stim));
      }
      cp = log(1.*i/n_stim);
      if (i == 0) {
	per_mem[1] = log(0);
      } else if (i == 1) {
	local_d = sqrt(mech_dist + get_distortion(report_bits, i));
	per_mem[1] = cp + normal_lpdf(err | 0, local_d);
      } else {
	local_d = sqrt(mech_dist + get_distortion(report_bits, i));
	ae_prob = get_ae_probability(dist_bits, i, max_stim, spacing);
	per_mem[1] = cp + get_ae_lps(err, stim_errs, local_d, ae_prob,
				     n_stim);
      }
      per_mem[2] = log(1 - exp(cp)) + uniform_lpdf(err | -pi(), pi());
      lps[i+1] = plf + log_sum_exp(per_mem);
    }
    return log_sum_exp(lps);
  }
}

data {
//sizes 
  int<lower=0> T; // number of trials
  int<lower=2> N; // max number of stimuli
  int<lower=1> S; // number of subjects
  real<lower=0> stim_spacing; // minimum spacing between stimuli
  
  // prior data
  real<lower=0> report_bits_mean_mean;
  real<lower=0> report_bits_mean_var;
  real<lower=0> report_bits_var_mean;
  real<lower=0> report_bits_var_var;

  real<lower=0> dist_bits_mean_mean;
  real<lower=0> dist_bits_mean_var;
  real<lower=0> dist_bits_var_mean;
  real<lower=0> dist_bits_var_var;

  real<lower=0> mech_dist_mean_mean;
  real<lower=0> mech_dist_mean_var;
  real<lower=0> mech_dist_var_mean;
  real<lower=0> mech_dist_var_var;

  real<lower=0> stim_mem_mean_mean;
  real<lower=0> stim_mem_mean_var;
  real<lower=0> stim_mem_var_mean;
  real<lower=0> stim_mem_var_var;

  // main data
  vector[T] report_err; // dist from target
  int<lower=1, upper=N> num_stim[T]; // number of stimuli on each trial
  matrix[T, N] stim_locs; // relative locations of stimuli in report space
  int<lower=1, upper=S> subj_id[T];
  matrix[T, N] stim_errs; // errors relative to each stimulus 
}

parameters {
  // prior-related
  real<lower=0> report_bits_mean;
  real<lower=0> report_bits_var;

  real<lower=0> dist_bits_mean;
  real<lower=0> dist_bits_var;

  real<lower=0> mech_dist_mean;
  real<lower=0> mech_dist_var;

  real<lower=0> stim_mem_mean;
  real<lower=0> stim_mem_var;


  // data-related
  vector[S] report_bits_raw;
  vector[S] dist_bits_raw;
  vector[S] mech_dist_raw;
  vector[S] stim_mem_raw;
}

transformed parameters {
  vector<lower=0>[S] report_bits;
  vector<lower=0>[S] dist_bits;
  vector<lower=0>[S] mech_dist;
  vector<lower=0>[S] stim_mem;

  report_bits = report_bits_mean + report_bits_raw*report_bits_var;
  dist_bits = dist_bits_mean + dist_bits_raw*dist_bits_var;
  mech_dist = mech_dist_mean + mech_dist_raw*mech_dist_var; 
  stim_mem = stim_mem_mean + stim_mem_raw*stim_mem_var; 
}

model {
  // var declarations
  int subj;
  int n_stim;

  // priors
  report_bits_var ~ normal(report_bits_var_mean, report_bits_var_var);
  report_bits_mean ~ normal(report_bits_mean_mean, report_bits_mean_var);

  dist_bits_var ~ normal(dist_bits_var_mean, dist_bits_var_var);
  dist_bits_mean ~ normal(dist_bits_mean_mean, dist_bits_mean_var);

  mech_dist_var ~ normal(mech_dist_var_mean, mech_dist_var_var);
  mech_dist_mean ~ normal(mech_dist_mean_mean, mech_dist_mean_var);

  stim_mem_var ~ normal(stim_mem_var_mean, stim_mem_var_var);
  stim_mem_mean ~ normal(stim_mem_mean_mean, stim_mem_mean_var);

  report_bits_raw ~ normal(0, 1);
  dist_bits_raw ~ normal(0, 1);
  mech_dist_raw ~ normal(0, 1);
  stim_mem_raw ~ normal(0, 1);
  
  // model
  for (t in 1:T) {
    subj = subj_id[t];
    n_stim = num_stim[t];
    target += get_lps(report_err[t], stim_errs[t]', report_bits[subj],
		      dist_bits[subj], mech_dist[subj], stim_mem[subj],
		      n_stim, N, stim_spacing);
  }
}

generated quantities {
  vector[T] log_lik;
  vector[T] err_hat;
  
  for (t in 1:T) {
    int subj;
    int n_stim;
    real ae_prob;
    real local_d;
    real err;
    vector[N] lps;
    vector[N] aep_per;
    int draw_ind;
    real eh;
    int corr;
    real corr_prob;
    int stim_enc;
    
    subj = subj_id[t];
    n_stim = num_stim[t];

    log_lik[t] = get_lps(report_err[t], stim_errs[t]', report_bits[subj],
			 dist_bits[subj], mech_dist[subj], stim_mem[subj],
			 n_stim, N, stim_spacing);

    stim_enc = min(poisson_rng(stim_mem[subj]), n_stim);
    corr_prob = min([1.*stim_enc/n_stim, 1.]);
    corr = bernoulli_rng(corr_prob);
    if (corr == 1) {
      local_d = sqrt(mech_dist[subj]
		     + get_distortion(report_bits[subj], stim_enc));
      if (stim_enc == 1) {
	eh = normal_rng(stim_locs[t, 1], local_d);
      } else {
	ae_prob = exp(get_ae_probability(dist_bits[subj], stim_enc, N,
					 stim_spacing));
      
	aep_per[1] = 1 - ae_prob;
	aep_per[2:n_stim] = rep_vector(ae_prob/(n_stim - 1), n_stim - 1);
	draw_ind = categorical_rng(aep_per[:n_stim]);
	eh = normal_rng(stim_locs[t, draw_ind], local_d);
      }
      if (eh > pi()) {
      err_hat[t] = eh - 2*pi();
      } else if (eh < -pi()) {
	err_hat[t] = eh + 2*pi();
      } else {
	err_hat[t] = eh;
      }
    } else {
      err_hat[t] = uniform_rng(-pi(), pi());
    }
  }
}
