#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint32_t n_images;
  uint32_t n_rows;
  uint32_t n_cols;
  uint8_t* pixels; // Raw pixel data (0-255)
  uint8_t* labels; // Labels (0-9)
} MNISTDataset;

typedef struct {
  uint32_t n_images;
  uint32_t n_rows;
  uint32_t n_cols;
  float* pixels;   // Normalized pixel data (0.0-1.0)
  uint8_t* labels; // Labels (0-9)
} NormalizedMNIST;

/**
 * Read a big-endian uint32_t from file, byte-swap it, and validate against
 * expected value.
 *
 * @param f          File handle to read from
 * @param out        Pointer to store the read value
 * @param expected   Expected value after byte-swap
 * @param field_name Name of the field (for error messages)
 * @return 0 on success, -1 on read error or validation failure
 */
int read_and_validate_uint32_be(FILE* f, uint32_t* out, uint32_t expected,
                                const char* field_name);

/**
 * Load MNIST images and labels from IDX files.
 *
 * @param images_path Path to the IDX3 images file (e.g.,
 * "train-images.idx3-ubyte")
 * @param labels_path Path to the IDX1 labels file (e.g.,
 * "train-labels.idx1-ubyte")
 * @param dataset     Pointer to MNISTDataset to populate
 * @return 0 on success, -1 on error
 */
int load_mnist_images(const char* images_path, const char* labels_path,
                      MNISTDataset* dataset);

/**
 * Normalize MNIST pixel values from uint8 (0-255) to float (0-1).
 *
 * NOTE: The labels pointer is shared between src and dst (not copied).
 * The caller must ensure src->labels outlives dst and is freed only once
 * (typically by freeing src->labels after both datasets are no longer needed).
 *
 * @param src  Source MNISTDataset with raw pixel values
 * @param dst  Destination NormalizedMNIST to populate
 * @return 0 on success, -1 on memory allocation failure
 */
int normalize_mnist(const MNISTDataset* src, NormalizedMNIST* dst);

/**
 * Free resources allocated by normalize_mnist.
 *
 * @param dataset  Pointer to NormalizedMNIST to free
 */
void free_normalized_mnist(NormalizedMNIST* dataset);

#endif
