functions {  
  real get_distortion(real bits, int n_stim) {
    real distortion;
    distortion = 2*pi()*exp(-2*bits/n_stim);
    return distortion;	  
  }
  
  real get_lps(real err, vector stim_errs, real report_bits,
	       real mem_stim, real mech_dist, int n_stim, int max_stim,
	       real spacing) {
    vector[2] lps;
    real cp;
    real local_d;
    cp = min([mem_stim/n_stim, 1.]);
    local_d = sqrt(mech_dist + get_distortion(report_bits, n_stim));
    
    lps[1] = log(cp) + normal_lpdf(err | 0, local_d);
    lps[2] = log1m(cp) + uniform_lpdf(err | -pi(), pi());
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

  // data-related
  vector[S] report_bits_raw;
  vector[S] dist_bits_raw;
  vector[S] mech_dist_raw;
}

transformed parameters {
  vector<lower=0>[S] report_bits;
  vector<lower=0>[S] stim_mem;
  vector<lower=0, upper=N>[S] mech_dist;

  report_bits = report_bits_mean + report_bits_raw*report_bits_var;
  stim_mem = dist_bits_mean + dist_bits_raw*dist_bits_var;
  mech_dist = mech_dist_mean + mech_dist_raw*mech_dist_var; 
}

model {
  // var declarations
  int subj;
  int n_stim;
  real ae_prob;
  real local_d;
  real err;
  vector[N] lps;
  
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
    n_stim = num_stim[t];

    target += get_lps(report_err[t], stim_errs[t]', report_bits[subj],
		      stim_mem[subj], mech_dist[subj], n_stim, N,
		      stim_spacing);
  }
}

generated quantities {
  vector[T] log_lik;
  vector[T] err_hat;
  
  for (t in 1:T) {
    int subj;
    int n_stim;
    real corr_prob;
    real local_d;
    int corr;
    real eh;
    
    subj = subj_id[t];
    n_stim = num_stim[t];

    log_lik[t] = get_lps(report_err[t], stim_errs[t]', report_bits[subj],
			 stim_mem[subj], mech_dist[subj], n_stim, N,
			 stim_spacing);

    local_d = sqrt(mech_dist[subj]
		   + get_distortion(report_bits[subj], n_stim));
    corr_prob = min([stim_mem[subj]/n_stim, 1.]);
    corr = bernoulli_rng(corr_prob);
    if (corr == 1) {
      eh = normal_rng(0, local_d);
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
