

template <int BLUR_SIZE, int TILE_WIDTH>
__global__ void blurKernel(float *out, float *in, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int pixVal = 0;
    int pixels = 0;

    // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
    for (int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE + 1; ++blurrow) {
      for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1; ++blurcol) {

        int currow = row + blurrow;
        int curcol = col + blurcol;
        // Verify we have a valid image pixel
        if (currow > -1 && currow < height && curcol > -1 && curcol < width) {
          pixVal += in[currow * width + curcol];
          pixels++; // Keep track of number of pixels in the avg
        }
      }
    }

    // Write our new pixel value out
    out[row * width + col] = (unsigned char) (pixVal / pixels);
  }
}
