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

  real<lower=0> encoding_rate_mean_mean;
  real<lower=0> encoding_rate_mean_var;
  real<lower=0> encoding_rate_var_mean;
  real<lower=0> encoding_rate_var_var;

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

  real<lower=0> encoding_rate_mean;
  real<lower=0> encoding_rate_var;

  // data-related
  vector[S] report_mse1_raw;
  vector[S] dist_mse1_raw;
  vector[S] mech_mse_raw;
  vector[S] encoding_rate_raw;
}

transformed parameters {
  
  vector<lower=0>[S] report_bits;
  vector<lower=0>[S] dist_bits;
  vector<lower=0>[S] report_mse1;
  vector<lower=0>[S] dist_mse1;
  vector<lower=0>[S] mech_dist;
  vector<lower=0>[S] encoding_rate;

  report_mse1 = report_mse1_mean + report_mse1_var*report_mse1_raw;
  dist_mse1 = dist_mse1_mean + dist_mse1_var*dist_mse1_raw;
  mech_dist = mech_mse_mean + mech_mse_var*mech_mse_raw;
  encoding_rate = encoding_rate_mean + encoding_rate_raw*encoding_rate_var;
  
  report_bits = get_bits(report_mse1, 1);
  dist_bits = get_bits(dist_mse1, 1);
}

model {
  // var declarations
  int subj;
  int n_stim;
  int tr_encoded;
  real ae_prob;
  real local_d;
  real err;
  vector[N+1] lps;
  int lps_start_ind;
  vector[N+1] enc_lps;
  real enc_lprob;
  real unif_prob;
  real ne_prob;
  real ninf_prob;
  int lps_end;
  
  // priors
  report_mse1_var ~ normal(report_bits_var_mean, report_bits_var_var);
  report_mse1_mean ~ normal(report_bits_mean_mean, report_bits_mean_var);

  dist_mse1_var ~ normal(dist_bits_var_mean, dist_bits_var_var);
  dist_mse1_mean ~ normal(dist_bits_mean_mean, dist_bits_mean_var);

  mech_mse_var ~ normal(mech_dist_var_mean, mech_dist_var_var);
  mech_mse_mean ~ normal(mech_dist_mean_mean, mech_dist_mean_var);

  encoding_rate_var ~ normal(encoding_rate_var_mean, encoding_rate_var_var);
  encoding_rate_mean ~ normal(encoding_rate_mean_mean, encoding_rate_mean_var);
  
  report_mse1_raw ~ normal(0, 1);
  dist_mse1_raw ~ normal(0, 1);
  mech_mse_raw ~ normal(0, 1);
  encoding_rate_raw ~ normal(0, 1);

  
  // model  
  for (t in 1:T) {
    subj = subj_id[t];
    n_stim = num_stim[t];

    err = report_err[t];

    // probability of zero encoding, total lapse
    enc_lps[1] = (poisson_lpmf(0 | encoding_rate[subj])
		  + uniform_lpdf(err | -pi(), pi()));

    // probability of 1 to n_stim encoded
    for (n_enc in 1:n_stim) {

      // probability that this many stimuli were encoded
      ne_prob = poisson_lpmf(n_enc | encoding_rate[subj]);
      if (n_enc == n_stim) {
	ninf_prob = poisson_lccdf(n_enc + 1 | encoding_rate[subj]);
	enc_lprob = log_sum_exp(ne_prob, ninf_prob);
      } else {
	enc_lprob = ne_prob;
      }

      // ae_prob and distortion for that many stimuli
      ae_prob = get_ae_probability(dist_bits[subj], n_enc, N, stim_spacing);
      local_d = sqrt(mech_dist[subj] + get_distortion(report_bits[subj],
						      n_enc));

      // probability that non-encoded stimulus is target
      unif_prob = n_enc; // to convert to real
      unif_prob = 1 - unif_prob/n_stim;

      if (unif_prob > 0) { // if non-zero prob, account for it!
	lps[1] = log(unif_prob) + uniform_lpdf(err | -pi(), pi());
	lps_start_ind = 1;
      } else { // else, don't
	lps_start_ind = 2;
      }

      // probability that target was encoded and no AE
      lps[2] = (log(1 - unif_prob) + log(1 - ae_prob)
		+ normal_lpdf(err | 0, local_d));

      // probability that target was encoded, but made AE
      // not representing randomness in stim choice...
      if (n_enc > 1) {
	for (i in 3:n_stim + 1) {
	  lps[i] = (log(1 - unif_prob) + log(ae_prob/(n_stim - 1))
		    + normal_lpdf(stim_errs[t, i - 1] | 0, local_d));
	}
	lps_end = n_stim + 1;
      } else {
	lps_end = 2;
      }
      // totalling up
      enc_lps[n_enc+1] = (enc_lprob
			  + log_sum_exp(lps[lps_start_ind:lps_end]));
    }
    target += log_sum_exp(enc_lps[:n_stim+1]);
  }
}

generated quantities {
  vector[T] log_lik;
  vector[T] err_hat; 
  // var declarations
  
  for (t in 1:T) {
    int subj;
    int n_stim;
    int tr_encoded;
    real ae_prob;
    real local_d;
    real err;
    vector[N+1] lps;
    int lps_start_ind;
    vector[N+1] enc_lps;
    real enc_lprob;
    real unif_prob;
    real ne_prob;
    real ninf_prob;
    int lps_end;
    int ec;
    real eh;
    int draw_ind;
    vector[N - 1] vp;

    subj = subj_id[t];
    n_stim = num_stim[t];

	
    err = report_err[t];

    ec = min(poisson_rng(encoding_rate[subj]), n_stim);
    if (ec == 0) {
      eh = uniform_rng(-pi(), pi());
    }
    // probability of zero encoding, total lapse
    enc_lps[1] = (poisson_lpmf(0 | encoding_rate[subj])
		  + uniform_lpdf(err | -pi(), pi()));

    // probability of 1 to n_stim encoded
    for (n_enc in 1:n_stim) {

      // probability that this many stimuli were encoded
      ne_prob = poisson_lpmf(n_enc | encoding_rate[subj]);
      if (n_enc == n_stim) {
	ninf_prob = poisson_lccdf(n_enc + 1 | encoding_rate[subj]);
	enc_lprob = log_sum_exp(ne_prob, ninf_prob);
      } else {
	enc_lprob = ne_prob;
      }

      // ae_prob and distortion for that many stimuli
      ae_prob = get_ae_probability(dist_bits[subj], n_enc, N, stim_spacing);
      local_d = sqrt(mech_dist[subj] + get_distortion(report_bits[subj],
						      n_enc));

      // probability that non-encoded stimulus is target
      unif_prob = n_enc;
      unif_prob = 1 - unif_prob/n_stim;

      if (ec == n_enc) {
	if (bernoulli_rng(unif_prob) == 1) {
	  eh = uniform_rng(-pi(), pi());
	} else if (bernoulli_rng(ae_prob) == 0 || n_enc == 1) {
	  eh = normal_rng(0, local_d);
	} else {
	  vp[:n_stim - 1] = rep_vector(1.0/(n_stim - 1), n_stim - 1);
	  draw_ind = categorical_rng(vp[:n_stim - 1]);
	  eh = normal_rng(stim_locs[t, draw_ind + 1], local_d);
	}
      }
      
      if (unif_prob > 0) { // if non-zero prob, account for it!
	lps[1] = log(unif_prob) + uniform_lpdf(err | -pi(), pi());
	lps_start_ind = 1;
      } else { // else, don't
	lps_start_ind = 2;
      }

      // probability that target was encoded and no AE
      lps[2] = (log(1 - unif_prob) + log(1 - ae_prob)
		+ normal_lpdf(err | 0, local_d));

      // probability that target was encoded, but made AE
      if (n_enc > 1) {
	for (i in 3:n_stim + 1) {
	  lps[i] = (log(1 - unif_prob) + log(ae_prob/(n_stim - 1))
		    + normal_lpdf(stim_errs[t, i - 1] | 0, local_d));
	}
	lps_end = n_stim + 1;
      } else {
	lps_end = 2;
      }
      // totalling up
      enc_lps[n_enc+1] = (enc_lprob
			  + log_sum_exp(lps[lps_start_ind:lps_end]));
    }
    if (eh > pi()) {
      err_hat[t] = eh - 2*pi();
    } else if (eh < -pi()) {
      err_hat[t] = eh + 2*pi();
    } else {
      err_hat[t] = eh;
    }
    log_lik[t] = log_sum_exp(enc_lps[:n_stim+1]);
  }
}
