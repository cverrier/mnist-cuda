#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "mnist.h"
#include "model.h"
}

// =============================================================================
// CUDA Error Checking
// =============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// =============================================================================
// Training Configuration
// =============================================================================

#define BATCH_SIZE 64
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.01f
#define INPUT_DIM 784
#define HIDDEN_DIM 128
#define OUTPUT_DIM 10

// =============================================================================
// Helper Functions
// =============================================================================

// Compute accuracy over a dataset using batches
// Returns accuracy as a float in [0, 1]
float compute_accuracy(Model* model, TrainState* state, float* d_pixels,
                       uint8_t* d_labels, int num_samples) {
  int num_batches = num_samples / BATCH_SIZE;
  int total_correct = 0;

  for (int b = 0; b < num_batches; b++) {
    float* d_batch_input = d_pixels + b * BATCH_SIZE * INPUT_DIM;
    uint8_t* d_batch_labels = d_labels + b * BATCH_SIZE;

    forward(model, state, d_batch_input);
    int correct = count_correct(state, d_batch_labels);
    if (correct < 0) {
      fprintf(stderr, "Error computing accuracy\n");
      return -1.0f;
    }
    total_correct += correct;
  }

  int evaluated = num_batches * BATCH_SIZE;
  return (float)total_correct / (float)evaluated;
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main(void) {
  printf("=== MNIST CUDA Training ===\n\n");

  // Declare variables here to avoid goto-over-initialization errors in C++
  int num_train_batches;
  const char* model_path = "model.bin";

  // ---------------------------------------------------------------------------
  // Load Training Data
  // ---------------------------------------------------------------------------
  printf("Loading training data...\n");
  MNISTDataset train_raw;
  if (load_mnist_images("data/mnist/train-images.idx3-ubyte",
                        "data/mnist/train-labels.idx1-ubyte",
                        &train_raw) != 0) {
    fprintf(stderr, "Failed to load training images\n");
    return -1;
  }
  printf("  Loaded %u training images (%ux%u)\n", train_raw.n_images,
         train_raw.n_rows, train_raw.n_cols);

  NormalizedMNIST train_data;
  if (normalize_mnist(&train_raw, &train_data) != 0) {
    fprintf(stderr, "Failed to normalize training data\n");
    free(train_raw.pixels);
    free(train_raw.labels);
    return -1;
  }

  // ---------------------------------------------------------------------------
  // Load Test Data
  // ---------------------------------------------------------------------------
  printf("Loading test data...\n");
  MNISTDataset test_raw;
  if (load_mnist_images("data/mnist/t10k-images.idx3-ubyte",
                        "data/mnist/t10k-labels.idx1-ubyte", &test_raw) != 0) {
    fprintf(stderr, "Failed to load test images\n");
    free(train_data.pixels);
    free(train_raw.pixels);
    free(train_raw.labels);
    return -1;
  }
  printf("  Loaded %u test images (%ux%u)\n", test_raw.n_images,
         test_raw.n_rows, test_raw.n_cols);

  NormalizedMNIST test_data;
  if (normalize_mnist(&test_raw, &test_data) != 0) {
    fprintf(stderr, "Failed to normalize test data\n");
    free(train_data.pixels);
    free(train_raw.pixels);
    free(train_raw.labels);
    free(test_raw.pixels);
    free(test_raw.labels);
    return -1;
  }

  // ---------------------------------------------------------------------------
  // Allocate GPU Memory for Dataset
  // ---------------------------------------------------------------------------
  printf("Copying data to GPU...\n");

  // Training data
  size_t train_pixels_size =
      (size_t)train_data.n_images * INPUT_DIM * sizeof(float);
  size_t train_labels_size = (size_t)train_data.n_images * sizeof(uint8_t);
  float* d_train_pixels;
  uint8_t* d_train_labels;

  CUDA_CHECK(cudaMalloc(&d_train_pixels, train_pixels_size));
  CUDA_CHECK(cudaMalloc(&d_train_labels, train_labels_size));
  CUDA_CHECK(cudaMemcpy(d_train_pixels, train_data.pixels, train_pixels_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_train_labels, train_data.labels, train_labels_size,
                        cudaMemcpyHostToDevice));

  // Test data
  size_t test_pixels_size =
      (size_t)test_data.n_images * INPUT_DIM * sizeof(float);
  size_t test_labels_size = (size_t)test_data.n_images * sizeof(uint8_t);
  float* d_test_pixels;
  uint8_t* d_test_labels;

  CUDA_CHECK(cudaMalloc(&d_test_pixels, test_pixels_size));
  CUDA_CHECK(cudaMalloc(&d_test_labels, test_labels_size));
  CUDA_CHECK(cudaMemcpy(d_test_pixels, test_data.pixels, test_pixels_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_test_labels, test_data.labels, test_labels_size,
                        cudaMemcpyHostToDevice));

  printf("  Training: %.2f MB, Test: %.2f MB\n",
         (train_pixels_size + train_labels_size) / (1024.0 * 1024.0),
         (test_pixels_size + test_labels_size) / (1024.0 * 1024.0));

  // ---------------------------------------------------------------------------
  // Create Model and Training State
  // ---------------------------------------------------------------------------
  printf("Creating model...\n");
  Model model;
  if (create_model(&model, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM) != 0) {
    fprintf(stderr, "Failed to create model\n");
    goto cleanup_gpu_data;
  }
  printf("  Architecture: %d -> %d -> %d\n", INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);

  TrainState state;
  if (create_train_state(&state, &model, BATCH_SIZE) != 0) {
    fprintf(stderr, "Failed to create training state\n");
    free_model(&model);
    goto cleanup_gpu_data;
  }
  printf("  Batch size: %d\n", BATCH_SIZE);

  // ---------------------------------------------------------------------------
  // Training Loop
  // ---------------------------------------------------------------------------
  num_train_batches = train_data.n_images / BATCH_SIZE;
  printf("\nStarting training: %d epochs, %d batches/epoch, lr=%.4f\n\n",
         NUM_EPOCHS, num_train_batches, LEARNING_RATE);

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    float epoch_loss = 0.0f;

    for (int batch = 0; batch < num_train_batches; batch++) {
      // Get pointers to current batch in device memory
      float* d_batch_input = d_train_pixels + batch * BATCH_SIZE * INPUT_DIM;
      uint8_t* d_batch_labels = d_train_labels + batch * BATCH_SIZE;

      // Forward pass
      forward(&model, &state, d_batch_input);

      // Compute loss and gradient
      compute_loss(&model, &state, d_batch_labels);

      // Backward pass
      backward(&model, &state, d_batch_input);

      // Update parameters
      update_params(&model, &state, LEARNING_RATE);

      // Accumulate loss (copy from device every batch for monitoring)
      float batch_loss;
      cudaMemcpy(&batch_loss, state.d_loss, sizeof(float),
                 cudaMemcpyDeviceToHost);
      epoch_loss += batch_loss;
    }

    // Compute average loss for epoch
    float avg_loss = epoch_loss / num_train_batches;

    // Compute train and test accuracy
    float train_acc =
        compute_accuracy(&model, &state, d_train_pixels, d_train_labels,
                         (int)train_data.n_images);
    float test_acc = compute_accuracy(&model, &state, d_test_pixels,
                                      d_test_labels, (int)test_data.n_images);

    printf("Epoch %2d/%d: loss=%.4f, train_acc=%.2f%%, test_acc=%.2f%%\n",
           epoch + 1, NUM_EPOCHS, avg_loss, train_acc * 100.0f,
           test_acc * 100.0f);
  }

  printf("\nTraining complete!\n");

  // ---------------------------------------------------------------------------
  // Save Model Weights
  // ---------------------------------------------------------------------------
  printf("Saving model to %s...\n", model_path);
  if (save_model(&model, model_path) != 0) {
    fprintf(stderr, "Failed to save model\n");
  } else {
    printf("Model saved successfully\n");
  }

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  free_train_state(&state);
  free_model(&model);

cleanup_gpu_data:
  cudaFree(d_train_pixels);
  cudaFree(d_train_labels);
  cudaFree(d_test_pixels);
  cudaFree(d_test_labels);

  // Free host memory
  free(train_data.pixels);
  free(train_raw.pixels);
  free(train_raw.labels);
  free(test_data.pixels);
  free(test_raw.pixels);
  free(test_raw.labels);

  return 0;
}
