#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"

void test_normalize_mnist() {
  MNISTDataset src;
  src.n_images = 2;
  src.n_rows = 2;
  src.n_cols = 2;

  uint8_t pixels[] = {0, 255, 128, 64, 32, 16, 200, 100};
  uint8_t labels[] = {5, 3};
  src.pixels = pixels;
  src.labels = labels;

  NormalizedMNIST dst;
  int result = normalize_mnist(&src, &dst);

  assert(result == 0);

  assert(dst.n_images == src.n_images);
  assert(dst.n_rows == src.n_rows);
  assert(dst.n_cols == src.n_cols);

  assert(dst.pixels[0] == 0.0f);
  assert(dst.pixels[1] == 1.0f);
  assert(fabsf(dst.pixels[2] - 128 / 255.0f) < 1e-6f);
  assert(fabsf(dst.pixels[3] - 64 / 255.0f) < 1e-6f);
  assert(fabsf(dst.pixels[4] - 32 / 255.0f) < 1e-6f);
  assert(fabsf(dst.pixels[5] - 16 / 255.0f) < 1e-6f);
  assert(fabsf(dst.pixels[6] - 200 / 255.0f) < 1e-6f);
  assert(fabsf(dst.pixels[7] - 100 / 255.0f) < 1e-6f);

  assert(dst.labels == src.labels);

  free_normalized_mnist(&dst);

  printf("normalize_mnist tests passed\n");
}

int main() {
  test_normalize_mnist();
  return 0;
}
