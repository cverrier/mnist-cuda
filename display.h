#ifndef DISPLAY_H
#define DISPLAY_H

#include "mnist.h"
#include <stdint.h>

/**
 * Display MNIST digit samples as ASCII art with their labels.
 *
 * @param dataset   Pointer to loaded MNISTDataset
 * @param n_samples Number of samples to display
 * @return 0 on success, -1 if n_samples exceeds dataset size
 */
int display_mnist_samples(const MNISTDataset* dataset, uint32_t n_samples);

#endif
