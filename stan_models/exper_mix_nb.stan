functions {  
  real get_distortion(real bits, int n_stim) {
    real distortion;
    distortion = 2*pi()*exp(-2*bits/n_stim);
    return distortion;	  
  }

  vector get_bits(vector distortion, int n_stim) {
    vector[dims(distortion)[0]] bits;
    bits = n_stim*.5*log(2*pi()./distortion);
    return bits;
  }
  
  real get_ae_probability(real bits, int n_stim, int max_stim, real spacing) {
    real distortion;
    real prob;
    real all_comb;
    real both;
    real either;
    distortion = get_distortion(bits, n_stim);
    prob = normal_cdf(-spacing, 0, sqrt(distortion));
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
    return prob*both*2 + prob*either;
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
  real<lower=0> report_mse1_mean;
  real<lower=0> report_mse1_var;

  real<lower=0> dist_mse1_mean;
  real<lower=0> dist_mse1_var;

  real<lower=0> mech_mse_mean;
  real<lower=0> mech_mse_var;

  // data-related
  vector<lower=0>[S] report_mse1;
  vector<lower=0>[S] dist_mse1;
  vector<lower=0>[S] mech_mse;
}

transformed parameters {
  
  vector<lower=0>[S] report_bits;
  vector<lower=0>[S] dist_bits;

  report_bits = get_bits(report_mse1, 1);
  dist_bits = get_bits(dist_mse1, 1);
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
  report_mse1_var ~ normal(report_bits_var_mean, report_bits_var_var);
  report_mse1_mean ~ normal(report_bits_mean_mean, report_bits_mean_var);

  dist_mse1_var ~ normal(dist_bits_var_mean, dist_bits_var_var);
  dist_mse1_mean ~ normal(dist_bits_mean_mean, dist_bits_mean_var);

  mech_mse_var ~ normal(mech_dist_var_mean, mech_dist_var_var);
  mech_mse_mean ~ normal(mech_dist_mean_mean, mech_dist_mean_var);

  report_mse1 ~ normal(report_mse1_mean, report_mse1_var);
  dist_mse1 ~ normal(dist_mse1_mean, dist_mse1_var);
  mech_mse ~ normal(mech_mse_mean, mech_mse_var);

  // model  
  for (t in 1:T) {
    subj = subj_id[t];
    n_stim = num_stim[t];

    ae_prob = get_ae_probability(dist_bits[subj], n_stim, N, stim_spacing);
    local_d = sqrt(mech_mse[subj] + get_distortion(report_bits[subj], n_stim));

    err = report_err[t];

    lps[1] = log(1 - ae_prob) + normal_lpdf(err | 0, local_d);
    for (i in 2:n_stim) {
      lps[i] = (log(ae_prob/(n_stim - 1))
		+ normal_lpdf(stim_errs[t, i] | 0, local_d));
    }
    target += log_sum_exp(lps[:n_stim]);
  }
}
