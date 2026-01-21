#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "display.h"
#include "mnist.h"

#define DEFAULT_SAMPLES 5

int main(int argc, char* argv[]) {
  uint32_t n_samples = DEFAULT_SAMPLES;

  if (argc > 1) {
    char* endptr;
    errno = 0;
    long arg = strtol(argv[1], &endptr, 10);

    // Check for conversion errors
    if (endptr == argv[1] || *endptr != '\0' || errno == ERANGE ||
        arg <= 0 || arg > UINT32_MAX) {
      fprintf(stderr, "Usage: %s [n_samples]\n", argv[0]);
      fprintf(stderr,
              "  n_samples: number of digits to display (default: %d)\n",
              DEFAULT_SAMPLES);
      return -1;
    }
    n_samples = (uint32_t)arg;
  }

  MNISTDataset dataset;
  if (load_mnist_images("data/mnist/train-images.idx3-ubyte",
                        "data/mnist/train-labels.idx1-ubyte", &dataset) != 0) {
    fprintf(stderr, "Failed to load MNIST data\n");
    return -1;
  }

  printf("Loaded %d images (%dx%d) with labels\n", dataset.n_images,
         dataset.n_rows, dataset.n_cols);

  if (display_mnist_samples(&dataset, n_samples) != 0) {
    free(dataset.pixels);
    free(dataset.labels);
    return -1;
  }

  free(dataset.pixels);
  free(dataset.labels);
  return 0;
}
