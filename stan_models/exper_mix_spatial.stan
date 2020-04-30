functions {  
  real get_distortion(real bits, int n_stim) {
    real distortion;
    distortion = 2*pi()*exp(-2*bits/n_stim);
    return distortion;	  
  }

  vector get_bits(vector distortion, int n_stim) {
    vector[dims(distortion)[1]] bits;
    bits = n_stim*.5*log(2*pi()./distortion);
    return bits;
  }
  
  vector get_ae_probability(real bits, vector dist, int n_stim) {
    real distortion;
    int n = dims(dist)[1]; 
    vector[n] ae_prob;
    distortion = get_distortion(bits, n_stim);
    for (i in 1:n) {
      ae_prob[i] = normal_cdf(-dist[i], 0, sqrt(distortion));
    }
    return ae_prob;
  }

  real report_err_rng(real db, real rb, real md, vector poss, int n_stim,
		      vector stim_locs) {
    vector[n_stim - 1] ae_prob;
    real local_d;
    real ae_ep;
    int draw_ind;
    real s_mean;
    vector[n_stim] out_prob;
    real eh;
    real err;

    ae_prob = get_ae_probability(db, poss[2:n_stim], n_stim);
    local_d = sqrt(md + get_distortion(rb, n_stim));

    ae_ep = sum(ae_prob);
    if (ae_ep >= 1) {
      out_prob[1] = 0;
      out_prob[2:] = ae_prob/ae_ep;
    } else {
      out_prob[1] = 1 - ae_ep;
      out_prob[2:] = ae_prob;
    }
    draw_ind = categorical_rng(out_prob);
    s_mean = stim_locs[draw_ind];
    eh = normal_rng(s_mean, local_d);
    if (eh > pi()) {
      err = eh - 2*pi();
    } else if (eh < -pi()) {
      err = eh + 2*pi();
    } else {
      err = eh;
    }
    return err;
  }
  
  real compute_log_prob(real err, real db, real rb, real md, vector poss,
			int n_stim, vector alt_err) {
    vector[n_stim - 1] ae_prob;
    real local_d;
    vector[n_stim] lps;
    real ae_ep;
    int lps_start_ind;
    real sum_lps;
    vector[n_stim] out_prob;

    ae_prob = get_ae_probability(db, poss[2:n_stim], n_stim);
    local_d = sqrt(md + get_distortion(rb, n_stim));

    ae_ep = sum(ae_prob);
    if (ae_ep >= 1) {
      lps_start_ind = 2;
      out_prob[2:] = ae_prob/ae_ep;
    } else {
      lps_start_ind = 1;
      lps[1] = log(1 - ae_ep) + normal_lpdf(err | 0, local_d);
      out_prob[2:] = ae_prob;
    }
    for (i in 2:n_stim) {
      lps[i] = (log(out_prob[i])
		+ normal_lpdf(alt_err[i] | 0, local_d));
    }
    sum_lps = log_sum_exp(lps[lps_start_ind:]);
    return sum_lps;
  }
}

data {
//sizes 
  int<lower=0> T; // number of trials
  int<lower=2> N; // max number of stimuli
  int<lower=1> S; // number of subjects
  
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

  // main data
  vector[T] report_err; // dist from target
  int<lower=1, upper=N> num_stim[T]; // number of stimuli on each trial
  matrix[T, N] stim_locs; // relative locations of stimuli in report space
  matrix[T, N] stim_poss; // relative locations of stimuli in cue space
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

  // data-related
  vector[S] report_bits_raw;
  vector[S] dist_bits_raw;
  vector[S] mech_dist_raw;
}

transformed parameters {
  vector<lower=0>[S] report_bits;
  vector<lower=0>[S] dist_bits;
  vector<lower=0>[S] mech_dist;

  report_bits = report_bits_mean + report_bits_raw*report_bits_var;
  dist_bits = dist_bits_mean + dist_bits_raw*dist_bits_var;
  mech_dist = mech_dist_mean + mech_dist_raw*mech_dist_var; 
}

model {
  // var declarations
  int subj;
  real sum_lps;
  
  // priors
  report_bits_var ~ normal(report_bits_var_mean, report_bits_var_var);
  report_bits_mean ~ normal(report_bits_mean_mean, report_bits_mean_var);

  dist_bits_var ~ normal(dist_bits_var_mean, dist_bits_var_var);
  dist_bits_mean ~ normal(dist_bits_mean_mean, dist_bits_mean_var);

  mech_dist_var ~ normal(mech_dist_var_mean, mech_dist_var_var);
  mech_dist_mean ~ normal(mech_dist_mean_mean, mech_dist_mean_var);
  
  report_bits_raw ~ normal(0, 1);
  dist_bits_raw ~ normal(0, 1);
  mech_dist_raw ~ normal(0, 1);

  // model  
  for (t in 1:T) {
    subj = subj_id[t];
    sum_lps = compute_log_prob(report_err[t], dist_bits[subj],
			       report_bits[subj], mech_dist[subj],
			       stim_poss[t]', num_stim[t], stim_errs[t]');
    target += sum_lps;
  }
}

generated quantities {
  vector[T] log_lik;
  vector[T] err_hat;
  for (t in 1:T) {
    int subj = subj_id[t];
    log_lik[t] = compute_log_prob(report_err[t], dist_bits[subj],
				  report_bits[subj], mech_dist[subj],
				  stim_poss[t]', num_stim[t], stim_errs[t]');
    err_hat[t] = report_err_rng(dist_bits[subj], report_bits[subj],
				mech_dist[subj], stim_poss[t]', num_stim[t],
				stim_locs[t]');
  }
}
