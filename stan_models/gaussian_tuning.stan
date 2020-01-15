
functions {
  vector resp_func(real inp, vector scales, vector cents, vector wids) {
    return scales .* exp(-(cents - inp).*(cents - inp)./(2*wids));
  }
  
  vector encoding_func(vector all_inp, vector scales, vector cents,
		       vector wids) {
    vector[dims(scales)[1]] resp;
    for (i in 1:dims(all_inp)[1]) {
      if (i == 1) {
	resp = resp_func(all_inp[i], scales, cents, wids);
      } else {
	resp += resp_func(all_inp[i], scales, cents, wids);
      }
    }
    return resp;	  
  }

}

data {
  //sizes 
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of neurons
  int<lower=0> C; // number of stimuli

  // rf params
  vector[K] cents;
  vector[K] wids;
  vector[K] scales;

  // noise params;
  matrix[K, K] cov_mat;

  // fit params
  real buffer;
  
  // main data
  matrix[N, K] samps; // samples from rf model
}

parameters {
  vector<lower=cents[1] + buffer, upper=cents[K] - buffer>[C] stims;
}

transformed parameters {
  ordered[C] stims_ordered;
  stims_ordered = stims;
}

model {
  vector[K] encoded;
  
  stims ~ uniform(cents[1] + buffer, cents[K] - buffer);
  encoded = encoding_func(stims_ordered, scales, cents, wids);
  for (n in 1:N) {
    samps[n] ~ multi_normal(encoded, cov_mat);
  }
}

