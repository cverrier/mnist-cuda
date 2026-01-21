#include "display.h"
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int display_mnist_samples(const MNISTDataset* dataset, uint32_t n_samples) {
  if (n_samples > dataset->n_images) {
    printf("Error: Requested %d samples, but dataset only has %d images.\n",
           n_samples, dataset->n_images);
    return -1;
  }

  // ASCII grayscale ramp from dark to light
  const char* grayscale = " .:-=+*#%@";
  const size_t levels = strlen(grayscale);
  assert(levels == 10);

  for (uint32_t i = 0; i < n_samples; i++) {
    printf("\n=== Sample %" PRIu32 " | Label: %d ===\n", i, dataset->labels[i]);

    uint32_t image_size = dataset->n_rows * dataset->n_cols;
    uint8_t* image = &dataset->pixels[i * image_size];

    for (uint32_t row = 0; row < dataset->n_rows; row++) {
      for (uint32_t col = 0; col < dataset->n_cols; col++) {
        uint8_t pixel = image[row * dataset->n_cols + col];
        size_t level = pixel * levels / 256;
        // Bounds check for defense-in-depth (should always be in range)
        if (level >= levels) {
          level = levels - 1;
        }
        printf("%c", grayscale[level]);
      }
      printf("\n");
    }
  }
  return 0;
}
