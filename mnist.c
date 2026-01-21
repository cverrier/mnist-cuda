#include "mnist.h"

#include <inttypes.h>
#include <stdlib.h>

#include "bswap.h"

enum {
  MNIST_IMAGE_MAGIC = 0x00000803,
  MNIST_LABEL_MAGIC = 0x00000801,
  MNIST_N_ROWS = 0x0000001C,
  MNIST_N_COLS = 0x0000001C,
  // Maximum allowed images to prevent memory exhaustion attacks
  MNIST_MAX_IMAGES = 1000000
};

int read_uint32_be(FILE* f, uint32_t* out, const char* field_name) {
  if (fread(out, sizeof(uint32_t), 1, f) != 1) {
    fprintf(stderr, "Error: Failed to read %s\n", field_name);
    return -1;
  }
  *out = bswap32(*out);
  return 0;
}

int read_and_validate_uint32_be(FILE* f, uint32_t* out, uint32_t expected,
                                const char* field_name) {
  if (read_uint32_be(f, out, field_name) != 0) {
    return -1;
  }
  if (*out != expected) {
    fprintf(stderr,
            "Error: Invalid %s 0x%08" PRIx32 " (expected 0x%08" PRIx32 ")\n",
            field_name, *out, expected);
    return -1;
  }
  return 0;
}

int load_mnist_images(const char* images_path, const char* labels_path,
                      MNISTDataset* dataset) {
  FILE* f = fopen(images_path, "rb");
  if (!f) {
    fprintf(stderr, "Error: Failed to open images file: %s\n", images_path);
    return -1;
  }

  uint32_t magic_number, n_images, n_rows, n_cols;
  if (read_and_validate_uint32_be(f, &magic_number, MNIST_IMAGE_MAGIC,
                                  "magic number") != 0 ||
      read_uint32_be(f, &n_images, "number of images") != 0 ||
      read_and_validate_uint32_be(f, &n_rows, MNIST_N_ROWS, "number of rows") !=
          0 ||
      read_and_validate_uint32_be(f, &n_cols, MNIST_N_COLS,
                                  "number of columns") != 0) {
    fclose(f);
    return -1;
  }

  // Bounds check to prevent memory exhaustion attacks
  if (n_images > MNIST_MAX_IMAGES) {
    fprintf(stderr, "Error: n_images %" PRIu32 " exceeds maximum %d\n",
            n_images, MNIST_MAX_IMAGES);
    fclose(f);
    return -1;
  }

  dataset->n_images = n_images;
  dataset->n_rows = n_rows;
  dataset->n_cols = n_cols;

  // Cast all operands to size_t before multiplication to prevent overflow
  size_t n_pixels = (size_t)n_images * (size_t)n_rows * (size_t)n_cols;
  dataset->pixels = malloc(n_pixels);
  if (!dataset->pixels) {
    fprintf(stderr, "Failed to allocate memory for pixels\n");
    fclose(f);
    return -1;
  }

  if (fread(dataset->pixels, 1, n_pixels, f) != n_pixels) {
    fprintf(stderr, "Failed to read pixel data\n");
    free(dataset->pixels);
    fclose(f);
    return -1;
  }

  fclose(f);

  f = fopen(labels_path, "rb");
  if (!f) {
    fprintf(stderr, "Error: Failed to open labels file: %s\n", labels_path);
    free(dataset->pixels);
    return -1;
  }

  uint32_t n_labels;
  if (read_and_validate_uint32_be(f, &magic_number, MNIST_LABEL_MAGIC,
                                  "label magic number") != 0 ||
      read_and_validate_uint32_be(f, &n_labels, n_images, "number of labels") !=
          0) {
    fclose(f);
    free(dataset->pixels);
    return -1;
  }

  dataset->labels = malloc(n_labels);
  if (!dataset->labels) {
    fprintf(stderr, "Failed to allocate memory for labels\n");
    fclose(f);
    free(dataset->pixels);
    return -1;
  }

  if (fread(dataset->labels, 1, n_labels, f) != n_labels) {
    fprintf(stderr, "Failed to read label data\n");
    free(dataset->labels);
    free(dataset->pixels);
    fclose(f);
    return -1;
  }

  fclose(f);
  return 0;
}

int normalize_mnist(const MNISTDataset* src, NormalizedMNIST* dst) {
  dst->n_images = src->n_images;
  dst->n_rows = src->n_rows;
  dst->n_cols = src->n_cols;

  // Cast all operands to size_t before multiplication to prevent overflow
  size_t n_pixels =
      (size_t)src->n_images * (size_t)src->n_rows * (size_t)src->n_cols;
  dst->pixels = malloc(n_pixels * sizeof(float));
  if (!dst->pixels)
    return -1;
  for (size_t i = 0; i < n_pixels; i++) {
    dst->pixels[i] = src->pixels[i] / 255.0f;
  }

  // NOTE: Labels pointer is shared with source dataset (not copied).
  // Caller must ensure src->labels outlives dst and is freed only once.
  dst->labels = src->labels;
  return 0;
}

void free_normalized_mnist(NormalizedMNIST* dataset) { free(dataset->pixels); }
