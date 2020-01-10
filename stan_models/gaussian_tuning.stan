
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
	resp = resp + resp_func(all_inp[i], scales, cents, wids);
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
  int<lower=0> T; // stimulus conditions

  // rf params
  vector[K] cents;
  vector[K] wids;
  vector[K] scales;

  // noise params;
  matrix[K, K] cov_mat;

  // fit params
  real buffer;
  
  // main data
  real samps[T, N, K]; // samples from rf model
}

parameters {
  matrix<lower=cents[1] + buffer, upper=cents[K] - buffer>[T, C] stims;
}

transformed parameters {
  matrix[T, C] stims_constrained;
  ordered[C] stims_row;
  for (i in 1:T) {
    stims_row = stims[i]';
    stims_constrained[i] = stims_row';
  }
}

model {
  vector[K] encoded;  
  for (i in 1:T) {
    stims[i] ~ uniform(cents[1] + buffer, cents[K] - buffer);
    encoded = encoding_func(stims_constrained[i]', scales, cents, wids);
    for (n in 1:N) {
      to_vector(samps[i, n]) ~ multi_normal(encoded, cov_mat);
    }
  }
}

