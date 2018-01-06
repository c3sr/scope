

__device__ float lhtan_activate_kernel(float x) {
  if (x < 0) {
    return .001f * x;
  }
  if (x > 1) {
    return .001f * (x - 1.f) + 1.f;
  }
  return x;
}
__device__ float lhtan_gradient_kernel(float x) {
  if (x > 0 && x < 1) {
    return 1;
  }
  return .001;
}

__device__ float hardtan_activate_kernel(float x) {
  if (x < -1) {
    return -1;
  }
  if (x > 1) {
    return 1;
  }
  return x;
}
__device__ float linear_activate_kernel(float x) {
  return x;
}
__device__ float logistic_activate_kernel(float x) {
  return 1.f / (1.f + expf(-x));
}
__device__ float loggy_activate_kernel(float x) {
  return 2.f / (1.f + expf(-x)) - 1;
}
__device__ float relu_activate_kernel(float x) {
  return x * (x > 0);
}
__device__ float elu_activate_kernel(float x) {
  return (x >= 0) * x + (x < 0) * (expf(x) - 1);
}
__device__ float relie_activate_kernel(float x) {
  return (x > 0) ? x : .01f * x;
}
__device__ float ramp_activate_kernel(float x) {
  return x * (x > 0) + .1f * x;
}
__device__ float leaky_activate_kernel(float x) {
  return (x > 0) ? x : .1f * x;
}
__device__ float tanh_activate_kernel(float x) {
  return (2.f / (1 + expf(-2 * x)) - 1);
}
