

template <typename accum_t, typename scalar_t, typename output_t>
struct fused_bias_relu_epilogue {

  // Data members pass additional arguments to epilogue
  scalar_t const *Bias;
  accum_t threshold;

  /// Constructor callable on host and device initializes data members
  inline __device__ __host__ fused_bias_relu_epilogue(scalar_t const *Bias, accum_t threshold)
      : Bias(Bias), threshold(threshold) {
  }

  /// Applies bias + ReLu operation
  inline __device__ __host__ output_t operator()(accum_t accumulator, /// element of matrix product result
                                                 output_t c,          /// element of source accumulator matrix C
                                                 size_t idx           /// index of c element; may be used to load
                                                                      /// elements from other identically-
                                                                      /// structured matrices
                                                 ) const {

    // Compute the result by scaling the matrix product, adding bias,
    // and adding the scaled accumulator element.

    accum_t result = output_t(alpha * scalar_t(accumulator) + Bias[i] + // load and add the bias
                              beta * scalar_t(c));

    // apply clamping function
    return max(threshold, result);
  }
};
