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
  
  vector get_ae_probability(real bits, vector dist, int mem_stim) {
    real distortion;
    real se_lprob;
    int n = dims(dist)[1]; 
    vector[n] ae_prob;
    real neg_min;
    real pos_min;
    neg_min = negative_infinity();
    pos_min = positive_infinity();
    distortion = get_distortion(bits, mem_stim);
    se_lprob = log((mem_stim - 1.)/n);
    for (j in 1:n) {
      if (pos_min > dist[j] && dist[j] > 0) {
	pos_min = dist[j];
      } else if (neg_min < dist[j] && dist[j] < 0) {
	neg_min = dist[j];
      }
    }
    for (i in 1:n) {
      if (dist[i] == neg_min || dist[i] == pos_min) {
	if (-fabs(dist[i])/sqrt(2*distortion) < -8.5) {
	  ae_prob[i] = log(0);
	} else {
	  ae_prob[i] = se_lprob + normal_lcdf(-fabs(dist[i])
					      | 0, sqrt(2*distortion));
	}
      } else {
	ae_prob[i] = log(0);
      }
    }
    return ae_prob;
  }

  real report_err_nenc_rng(real db, real rb, vector poss, int n_stim,
			   vector stim_locs, int mem_stim) {
    vector[n_stim - 1] ae_prob;
    real local_d;
    real ae_ep;
    int draw_ind;
    real s_mean;
    vector[n_stim] out_prob;
    real eh;
    real err;

    local_d = sqrt(get_distortion(rb, mem_stim));

    if (mem_stim == 1) {
      out_prob = rep_vector(1./n_stim, n_stim);
    } else {
      ae_prob = get_ae_probability(db, poss[2:n_stim], mem_stim);
      ae_ep = log_sum_exp(ae_prob);
      out_prob[1] = log1m_exp(ae_ep);
      out_prob[2:] = ae_prob;
      if (is_nan(out_prob[1])) {
	out_prob[1] = 0.;
      }
      out_prob = exp(out_prob);
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
  
  real report_err_rng(real db, real rb, vector poss, int n_stim,
		      vector stim_locs, real enc_rate) {
    int mem_stim;
    real err;
    mem_stim = min(n_stim, poisson_rng(enc_rate));
    if (mem_stim == 0) {
      err = uniform_rng(-pi(), pi());
    } else {
      err = report_err_nenc_rng(db, rb, poss, n_stim, stim_locs, mem_stim);
    }
    return err;
  }

  real compute_lp_enc(real err, real db, real rb, vector poss,
		      int n_stim, vector alt_err, int mem_stim) {
    vector[n_stim - 1] ae_prob;
    real local_d;
    vector[n_stim] lps;
    real ae_ep;
    int lps_start_ind;
    real t_enc_lps;
    real nt_enc_lps;
    real sum_lps;
    vector[n_stim] out_prob;

    local_d = sqrt(get_distortion(rb, mem_stim));

    if (mem_stim == 1) {
      t_enc_lps = log(1./n_stim) + normal_lpdf(err | 0, local_d);
      nt_enc_lps = log(1 - 1./n_stim) + uniform_lpdf(err | -pi(), pi());
    } else {
      ae_prob = get_ae_probability(db, poss[2:n_stim], mem_stim);
      ae_ep = log_sum_exp(ae_prob);
      if (is_nan(ae_ep)) {
	ae_ep = log(0);
      }
      lps[1] = log1m_exp(ae_ep) + normal_lpdf(err | 0, local_d);
      out_prob[2:] = ae_prob;
      for (i in 2:n_stim) {
	lps[i] = (out_prob[i]
		  + normal_lpdf(alt_err[i] | 0, local_d));
      }
      t_enc_lps = log(1.*mem_stim/n_stim) + log_sum_exp(lps);
      nt_enc_lps = (log(1 - 1.*mem_stim/n_stim)
		    + uniform_lpdf(err | -pi(), pi()));     
    }
    sum_lps = log_sum_exp(t_enc_lps, nt_enc_lps);
    return sum_lps;
  }

    vector compute_lp_enc_vec(real err, real db, real rb, vector poss,
			      int n_stim, vector alt_err, int mem_stim) {
    vector[n_stim - 1] ae_prob;
    real local_d;
    vector[n_stim] lps;
    real ae_ep;
    int lps_start_ind;
    real t_enc_lps;
    real nt_enc_lps;
    real sum_lps;
    vector[n_stim] out_prob;
    vector[2] out_vec;

    local_d = sqrt(get_distortion(rb, mem_stim));

    if (mem_stim == 1) {
      t_enc_lps = log(1./n_stim) + normal_lpdf(err | 0, local_d);
      nt_enc_lps = log(1 - 1./n_stim) + uniform_lpdf(err | -pi(), pi());
    } else {
      ae_prob = get_ae_probability(db, poss[2:n_stim], mem_stim);
      ae_ep = sum(exp(ae_prob));
      lps[1] = log(1 - ae_ep) + normal_lpdf(err | 0, local_d);
      out_prob[2:] = ae_prob;
      for (i in 2:n_stim) {
	lps[i] = (out_prob[i]
		  + normal_lpdf(alt_err[i] | 0, local_d));
      }
      t_enc_lps = log(1.*mem_stim/n_stim) + log_sum_exp(lps);
      nt_enc_lps = (log(1 - 1.*mem_stim/n_stim)
		    + uniform_lpdf(err | -pi(), pi()));
    }
    out_vec[1] = t_enc_lps;
    out_vec[2] = nt_enc_lps;
    return out_vec;
  }
  
  real compute_log_prob(real err, real db, real rb,  vector poss,
			int n_stim, vector alt_err, real enc_rate) {
    vector[n_stim + 1] enc_lps;
    real num_prob;
    vector[2] lpe;
    
    // no stim remembered
    enc_lps[1] = poisson_lpmf(0 | enc_rate) + uniform_lpdf(err | -pi(), pi());
    
    // stim remembered
    for (i in 1:n_stim) {
      lpe = compute_lp_enc_vec(err, db, rb, poss, n_stim, alt_err, i);
      if (i < n_stim) {
	num_prob = poisson_lpmf(i | enc_rate);
      } else {
	num_prob = poisson_lccdf(n_stim - 1 | enc_rate);
      }
      enc_lps[i+1] = log_sum_exp(num_prob + lpe[1], num_prob + lpe[2]);
    }
    return log_sum_exp(enc_lps);
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

  real<lower=0> enc_rate_mean_mean;
  real<lower=0> enc_rate_mean_var;
  real<lower=0> enc_rate_var_mean;
  real<lower=0> enc_rate_var_var;

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

  real<lower=0> enc_rate_mean;
  real<lower=0> enc_rate_var;

  // data-related
  vector[S] report_bits_raw;
  vector[S] dist_bits_raw;
  vector[S] enc_rate_raw;
}

transformed parameters {
  vector<lower=0>[S] report_bits;
  vector<lower=0>[S] dist_bits;
  vector<lower=0>[S] enc_rate;

  report_bits = report_bits_mean + report_bits_raw*report_bits_var;
  dist_bits = dist_bits_mean + dist_bits_raw*dist_bits_var;
  enc_rate = enc_rate_mean + enc_rate_raw*enc_rate_var;
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

  enc_rate_var ~ normal(enc_rate_var_mean, enc_rate_var_var);
  enc_rate_mean ~ normal(enc_rate_mean_mean, enc_rate_mean_var);
  
  report_bits_raw ~ normal(0, 1);
  dist_bits_raw ~ normal(0, 1);
  enc_rate_raw ~ normal(0, 1);

  // model
  for (t in 1:T) {
    subj = subj_id[t];
    sum_lps = compute_log_prob(report_err[t], dist_bits[subj],
			       report_bits[subj], stim_poss[t]',
			       num_stim[t], stim_errs[t]',
			       enc_rate[subj]);
    target += sum_lps;
  }
}

generated quantities {
  vector[T] log_lik;
  vector[T] err_hat;

  for (t in 1:T) {
    int subj = subj_id[t];
    log_lik[t] = compute_log_prob(report_err[t], dist_bits[subj],
				  report_bits[subj], stim_poss[t]',
				  num_stim[t], stim_errs[t]',
				  enc_rate[subj]);
    err_hat[t] = report_err_rng(dist_bits[subj], report_bits[subj],
				stim_poss[t]', num_stim[t],
				stim_locs[t]', enc_rate[subj]);
  }
}
