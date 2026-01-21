#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// C linkage for functions callable from C code
extern "C" {
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
// CUDA Kernels
// =============================================================================

// Linear layer with ReLU activation
// out[row][col] = max(0, sum_k(input[row][k] * W[k][col]) + b[col])
__global__ void linear_relu_kernel(float* out, float* input, float* W, float* b,
                                   int batch_size, int in_dim, int out_dim) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col >= out_dim || row >= batch_size)
    return;

  float sum = b[col];
  for (int k = 0; k < in_dim; ++k) {
    sum += input[row * in_dim + k] * W[k * out_dim + col];
  }
  out[row * out_dim + col] = fmaxf(0.0f, sum); // ReLU
}

// Linear layer without activation (for output layer)
// out[row][col] = sum_k(input[row][k] * W[k][col]) + b[col]
__global__ void linear_kernel(float* out, float* input, float* W, float* b,
                              int batch_size, int in_dim, int out_dim) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col >= out_dim || row >= batch_size)
    return;

  float sum = b[col];
  for (int k = 0; k < in_dim; ++k) {
    sum += input[row * in_dim + k] * W[k * out_dim + col];
  }
  out[row * out_dim + col] = sum;
}

// Softmax kernel: one thread per row (sample)
// probs[row] = softmax(logits[row]) with numerical stability
// Uses the max trick: softmax(x) = softmax(x - max(x))
__global__ void softmax_kernel(float* probs, float* logits, int batch_size) {
  int row = threadIdx.x + blockIdx.x * blockDim.x;

  if (row >= batch_size)
    return;

  // Step 1: Find max for numerical stability
  float max_val = logits[row * 10];
  for (int j = 1; j < 10; ++j) {
    max_val = fmaxf(max_val, logits[row * 10 + j]);
  }

  // Step 2: Compute exp(logits - max) into local array, accumulate sum
  // Local array for intermediate exp values (avoids global memory round-trip)
  float exp_vals[10]; // Hardcoded for MNIST (n_classes = 10)
  float sum = 0.0f;
  for (int j = 0; j < 10; ++j) {
    exp_vals[j] = expf(logits[row * 10 + j] - max_val);
    sum += exp_vals[j];
  }

  // Step 3: Normalize and write to global memory once
  for (int j = 0; j < 10; ++j) {
    probs[row * 10 + j] = exp_vals[j] / sum;
  }
}

// Matrix multiply with transposed first argument: C = A^T @ B
// A is (K, M), B is (K, N), C is (M, N)
// Each thread computes one element C[row][col] = sum_k A[k][row] * B[k][col]
__global__ void matmul_at_b_kernel(float* C, float* A, float* B, int M, int N,
                                   int K) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    // A^T[row][k] = A[k][row] = A[k * M + row]
    // B[k][col] = B[k * N + col]
    sum += A[k * M + row] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

// Bias gradient: b_grad[j] = sum_i(grad[i][j])
// grad is (batch_size, dim), b_grad is (dim,)
// One thread per output element (no atomics needed)
__global__ void bias_grad_kernel(float* b_grad, float* grad, int batch_size,
                                 int dim) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  if (j >= dim)
    return;

  float sum = 0.0f;
  for (int i = 0; i < batch_size; ++i) {
    sum += grad[i * dim + j];
  }
  b_grad[j] = sum;
}

// Hidden layer gradient with ReLU mask:
// hidden_grad = (d_logits_grad @ W2^T) * (hidden > 0)
// d_logits_grad is (batch, output_dim)
// W2 is (hidden_dim, output_dim)
// hidden is (batch, hidden_dim)
// hidden_grad is (batch, hidden_dim)
__global__ void hidden_grad_kernel(float* hidden_grad, float* d_logits_grad,
                                   float* W2, float* hidden, int batch_size,
                                   int hidden_dim, int output_dim) {
  int col = threadIdx.x + blockIdx.x * blockDim.x; // hidden dimension
  int row = threadIdx.y + blockIdx.y * blockDim.y; // batch dimension

  if (row >= batch_size || col >= hidden_dim)
    return;

  // Compute d_logits_grad[row] @ W2[col]^T
  // = sum_k d_logits_grad[row][k] * W2^T[k][col]
  // = sum_k d_logits_grad[row][k] * W2[col][k]
  float sum = 0.0f;
  for (int k = 0; k < output_dim; ++k) {
    sum += d_logits_grad[row * output_dim + k] * W2[col * output_dim + k];
  }

  // Apply ReLU gradient: gradient flows only where hidden > 0
  float h = hidden[row * hidden_dim + col];
  hidden_grad[row * hidden_dim + col] = (h > 0.0f) ? sum : 0.0f;
}

// SGD update: param -= learning_rate * grad
__global__ void sgd_kernel(float* param, float* grad, float lr, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size)
    return;
  param[idx] -= lr * grad[idx];
}

// Count correct predictions: one thread per sample
// Computes argmax of logits row, compares to label, atomically increments count
__global__ void count_correct_kernel(float* logits, uint8_t* labels,
                                     int* correct_count, int batch_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size)
    return;

  int row_offset = idx * 10;

  // Find argmax over 10 classes
  int max_idx = 0;
  float max_val = logits[row_offset];
  for (int j = 1; j < 10; ++j) {
    float val = logits[row_offset + j];
    if (val > max_val) {
      max_val = val;
      max_idx = j;
    }
  }

  // Increment count if prediction matches label
  if (max_idx == labels[idx]) {
    atomicAdd(correct_count, 1);
  }
}

// Fused softmax + cross-entropy loss + gradient computation
// One thread per sample (row)
// Computes:
//   probs = softmax(logits)
//   loss += -log(probs[label]) / batch_size  (via atomicAdd)
//   logits_grad = probs - one_hot(labels)
__global__ void softmax_cross_entropy_kernel(float* logits_grad, float* loss,
                                             float* logits, uint8_t* labels,
                                             int batch_size) {
  int row = threadIdx.x + blockIdx.x * blockDim.x;

  if (row >= batch_size)
    return;

  int row_offset = row * 10;
  uint8_t label = labels[row];

  // Step 1: Find max for numerical stability
  float max_val = logits[row_offset];
  for (int j = 1; j < 10; ++j) {
    max_val = fmaxf(max_val, logits[row_offset + j]);
  }

  // Step 2: Compute exp(logits - max) and accumulate sum
  float exp_vals[10]; // Hardcoded for MNIST (n_classes = 10)
  float sum = 0.0f;
  for (int j = 0; j < 10; ++j) {
    exp_vals[j] = expf(logits[row_offset + j] - max_val);
    sum += exp_vals[j];
  }

  // Step 3: Compute loss using log-softmax trick for numerical stability
  // -1 * log(prob[label]) = log(sum) + max - logits[label]
  // This avoids computing tiny probabilities before taking log
  float sample_loss =
      (logf(sum) + max_val - logits[row_offset + label]) / (float)batch_size;
  atomicAdd(loss, sample_loss);

  // Step 4: Compute gradient and write to global memory
  // logits_grad[j] = probs[j] - one_hot(labels)[j]
  for (int j = 0; j < 10; ++j) {
    float prob_j = exp_vals[j] / sum;
    float grad_j = prob_j - (j == label ? 1.0f : 0.0f);
    logits_grad[row_offset + j] = grad_j;
  }
}

// =============================================================================
// Host Helper Functions
// =============================================================================

// Xavier initialization: weights ~ U(-sqrt(6/(fan_in+fan_out)),
// sqrt(6/(fan_in+fan_out)))
static void xavier_init(float* data, int fan_in, int fan_out, int size) {
  float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
  for (int i = 0; i < size; i++) {
    // Random float in [-limit, limit]
    data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * limit - limit;
  }
}

// =============================================================================
// Lifecycle Functions
// =============================================================================

int create_model(Model* model, int input_dim, int hidden_dim, int output_dim) {
  model->input_dim = input_dim;
  model->hidden_dim = hidden_dim;
  model->output_dim = output_dim;

  // Initialize pointers to NULL for safe cleanup on error
  model->d_W1 = NULL;
  model->d_b1 = NULL;
  model->d_W2 = NULL;
  model->d_b2 = NULL;

  // Seed random number generator
  srand((unsigned int)time(NULL));

  // Declare all variables at top to avoid C++ goto-over-initialization errors
  cudaError_t err;
  size_t W1_count = (size_t)input_dim * (size_t)hidden_dim;
  size_t W1_size = W1_count * sizeof(float);
  size_t b1_size = (size_t)hidden_dim * sizeof(float);
  size_t W2_count = (size_t)hidden_dim * (size_t)output_dim;
  size_t W2_size = W2_count * sizeof(float);
  size_t b2_size = (size_t)output_dim * sizeof(float);
  float* h_W1 = NULL;
  float* h_W2 = NULL;

  // Allocate and initialize W1: (input_dim, hidden_dim)
  h_W1 = (float*)malloc(W1_size);
  if (!h_W1)
    goto cleanup;
  xavier_init(h_W1, input_dim, hidden_dim, (int)W1_count);

  err = cudaMalloc(&model->d_W1, W1_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    free(h_W1);
    goto cleanup;
  }
  err = cudaMemcpy(model->d_W1, h_W1, W1_size, cudaMemcpyHostToDevice);
  free(h_W1);
  h_W1 = NULL;
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  // Allocate b1: (hidden_dim,) - zero initialized
  err = cudaMalloc(&model->d_b1, b1_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMemset(model->d_b1, 0, b1_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  // Allocate and initialize W2: (hidden_dim, output_dim)
  h_W2 = (float*)malloc(W2_size);
  if (!h_W2)
    goto cleanup;
  xavier_init(h_W2, hidden_dim, output_dim, (int)W2_count);

  err = cudaMalloc(&model->d_W2, W2_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    free(h_W2);
    goto cleanup;
  }
  err = cudaMemcpy(model->d_W2, h_W2, W2_size, cudaMemcpyHostToDevice);
  free(h_W2);
  h_W2 = NULL;
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  // Allocate b2: (output_dim,) - zero initialized
  err = cudaMalloc(&model->d_b2, b2_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMemset(model->d_b2, 0, b2_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  return 0;

cleanup:
  free_model(model);
  return -1;
}

int create_train_state(TrainState* state, Model* model, int batch_size) {
  state->batch_size = batch_size;

  // Initialize pointers to NULL for safe cleanup on error
  state->d_hidden = NULL;
  state->d_logits = NULL;
  state->d_logits_grad = NULL;
  state->d_hidden_grad = NULL;
  state->d_W1_grad = NULL;
  state->d_b1_grad = NULL;
  state->d_W2_grad = NULL;
  state->d_b2_grad = NULL;
  state->d_loss = NULL;

  // Cast all operands to size_t before multiplication to prevent overflow
  size_t hidden_size =
      (size_t)batch_size * (size_t)model->hidden_dim * sizeof(float);
  size_t logits_size =
      (size_t)batch_size * (size_t)model->output_dim * sizeof(float);
  size_t W1_grad_size =
      (size_t)model->input_dim * (size_t)model->hidden_dim * sizeof(float);
  size_t b1_grad_size = (size_t)model->hidden_dim * sizeof(float);
  size_t W2_grad_size =
      (size_t)model->hidden_dim * (size_t)model->output_dim * sizeof(float);
  size_t b2_grad_size = (size_t)model->output_dim * sizeof(float);

  cudaError_t err;

  // Activations
  err = cudaMalloc(&state->d_hidden, hidden_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMalloc(&state->d_logits, logits_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  // Gradients
  err = cudaMalloc(&state->d_logits_grad, logits_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMalloc(&state->d_hidden_grad, hidden_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMalloc(&state->d_W1_grad, W1_grad_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMalloc(&state->d_b1_grad, b1_grad_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMalloc(&state->d_W2_grad, W2_grad_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }
  err = cudaMalloc(&state->d_b2_grad, b2_grad_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  // Loss scalar
  err = cudaMalloc(&state->d_loss, sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  return 0;

cleanup:
  free_train_state(state);
  return -1;
}

void free_model(Model* model) {
  cudaFree(model->d_W1);
  model->d_W1 = NULL;
  cudaFree(model->d_b1);
  model->d_b1 = NULL;
  cudaFree(model->d_W2);
  model->d_W2 = NULL;
  cudaFree(model->d_b2);
  model->d_b2 = NULL;
}

void free_train_state(TrainState* state) {
  cudaFree(state->d_hidden);
  state->d_hidden = NULL;
  cudaFree(state->d_logits);
  state->d_logits = NULL;
  cudaFree(state->d_logits_grad);
  state->d_logits_grad = NULL;
  cudaFree(state->d_hidden_grad);
  state->d_hidden_grad = NULL;
  cudaFree(state->d_W1_grad);
  state->d_W1_grad = NULL;
  cudaFree(state->d_b1_grad);
  state->d_b1_grad = NULL;
  cudaFree(state->d_W2_grad);
  state->d_W2_grad = NULL;
  cudaFree(state->d_b2_grad);
  state->d_b2_grad = NULL;
  cudaFree(state->d_loss);
  state->d_loss = NULL;
}

// =============================================================================
// Training Functions (stubs for future implementation)
// =============================================================================

void forward(Model* model, TrainState* state, float* d_input) {
  // Block dimensions: 16x16 threads (256 total, warp-aligned)
  dim3 block(16, 16);

  // Layer 1: input -> hidden (with ReLU)
  dim3 grid1((model->hidden_dim + block.x - 1) / block.x,
             (state->batch_size + block.y - 1) / block.y);
  linear_relu_kernel<<<grid1, block>>>(state->d_hidden, d_input, model->d_W1,
                                       model->d_b1, state->batch_size,
                                       model->input_dim, model->hidden_dim);

  // Layer 2: hidden -> logits (no activation)
  dim3 grid2((model->output_dim + block.x - 1) / block.x,
             (state->batch_size + block.y - 1) / block.y);
  linear_kernel<<<grid2, block>>>(state->d_logits, state->d_hidden, model->d_W2,
                                  model->d_b2, state->batch_size,
                                  model->hidden_dim, model->output_dim);
}

void compute_loss(Model* model, TrainState* state, uint8_t* d_labels) {
  // Zero out loss before accumulating (we use atomicAdd)
  cudaMemset(state->d_loss, 0, sizeof(float));

  // Fused softmax + cross-entropy + gradient computation
  // One thread per sample (row)
  int block_size = (state->batch_size <= 1024) ? state->batch_size : 1024;
  int grid_size = (state->batch_size + block_size - 1) / block_size;

  softmax_cross_entropy_kernel<<<grid_size, block_size>>>(
      state->d_logits_grad, state->d_loss, state->d_logits, d_labels,
      state->batch_size);
}

void backward(Model* model, TrainState* state, float* d_input) {
  dim3 block(16, 16);

  // Step 1: W2_grad = hidden^T @ d_logits_grad
  // hidden is (batch, hidden_dim), d_logits_grad is (batch, output_dim)
  // W2_grad is (hidden_dim, output_dim)
  dim3 grid_W2((model->output_dim + block.x - 1) / block.x,
               (model->hidden_dim + block.y - 1) / block.y);
  matmul_at_b_kernel<<<grid_W2, block>>>(
      state->d_W2_grad, state->d_hidden, state->d_logits_grad,
      model->hidden_dim, model->output_dim, state->batch_size);

  // Step 2: b2_grad = sum(d_logits_grad, axis=0)
  // One thread per output neuron, loops over batch
  int block_b2 = (model->output_dim <= 1024) ? model->output_dim : 1024;
  int grid_b2 = (model->output_dim + block_b2 - 1) / block_b2;
  bias_grad_kernel<<<grid_b2, block_b2>>>(state->d_b2_grad,
                                          state->d_logits_grad,
                                          state->batch_size, model->output_dim);

  // Step 3: hidden_grad = (d_logits_grad @ W2^T) * (hidden > 0)
  // Backprop through layer 2, then apply ReLU gradient
  dim3 grid_hg((model->hidden_dim + block.x - 1) / block.x,
               (state->batch_size + block.y - 1) / block.y);
  hidden_grad_kernel<<<grid_hg, block>>>(
      state->d_hidden_grad, state->d_logits_grad, model->d_W2, state->d_hidden,
      state->batch_size, model->hidden_dim, model->output_dim);

  // Step 4: W1_grad = input^T @ hidden_grad
  // input is (batch, input_dim), hidden_grad is (batch, hidden_dim)
  // W1_grad is (input_dim, hidden_dim)
  dim3 grid_W1((model->hidden_dim + block.x - 1) / block.x,
               (model->input_dim + block.y - 1) / block.y);
  matmul_at_b_kernel<<<grid_W1, block>>>(state->d_W1_grad, d_input,
                                         state->d_hidden_grad, model->input_dim,
                                         model->hidden_dim, state->batch_size);

  // Step 5: b1_grad = sum(hidden_grad, axis=0)
  // One thread per hidden neuron, loops over batch
  int block_b1 = (model->hidden_dim <= 1024) ? model->hidden_dim : 1024;
  int grid_b1 = (model->hidden_dim + block_b1 - 1) / block_b1;
  bias_grad_kernel<<<grid_b1, block_b1>>>(state->d_b1_grad,
                                          state->d_hidden_grad,
                                          state->batch_size, model->hidden_dim);
}

void update_params(Model* model, TrainState* state, float learning_rate) {
  int block_size = 256;

  // Update W1: (input_dim × hidden_dim) elements
  int W1_size = model->input_dim * model->hidden_dim;
  int grid_W1 = (W1_size + block_size - 1) / block_size;
  sgd_kernel<<<grid_W1, block_size>>>(model->d_W1, state->d_W1_grad,
                                      learning_rate, W1_size);

  // Update b1: (hidden_dim) elements
  int b1_size = model->hidden_dim;
  int grid_b1 = (b1_size + block_size - 1) / block_size;
  sgd_kernel<<<grid_b1, block_size>>>(model->d_b1, state->d_b1_grad,
                                      learning_rate, b1_size);

  // Update W2: (hidden_dim × output_dim) elements
  int W2_size = model->hidden_dim * model->output_dim;
  int grid_W2 = (W2_size + block_size - 1) / block_size;
  sgd_kernel<<<grid_W2, block_size>>>(model->d_W2, state->d_W2_grad,
                                      learning_rate, W2_size);

  // Update b2: (output_dim) elements
  int b2_size = model->output_dim;
  int grid_b2 = (b2_size + block_size - 1) / block_size;
  sgd_kernel<<<grid_b2, block_size>>>(model->d_b2, state->d_b2_grad,
                                      learning_rate, b2_size);
}

int count_correct(TrainState* state, uint8_t* d_labels) {
  // Allocate device counter
  int* d_correct;
  cudaError_t err = cudaMalloc(&d_correct, sizeof(int));
  if (err != cudaSuccess)
    return -1;
  cudaMemset(d_correct, 0, sizeof(int));

  // Launch kernel: one thread per sample
  int block_size = 256;
  int grid_size = (state->batch_size + block_size - 1) / block_size;
  count_correct_kernel<<<grid_size, block_size>>>(state->d_logits, d_labels,
                                                  d_correct, state->batch_size);

  // Copy result back to host
  int h_correct;
  cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_correct);

  return h_correct;
}

// =============================================================================
// Model Persistence
// =============================================================================

// File format constants
static const uint32_t MODEL_MAGIC = 0x4D4E5354; // "MNST" in little-endian
static const uint32_t MODEL_VERSION = 1;

int save_model(const Model* model, const char* path) {
  FILE* fp = fopen(path, "wb");
  if (!fp) {
    fprintf(stderr, "Failed to open %s for writing\n", path);
    return -1;
  }

  // Write header
  uint32_t header[5] = {MODEL_MAGIC, MODEL_VERSION, (uint32_t)model->input_dim,
                        (uint32_t)model->hidden_dim,
                        (uint32_t)model->output_dim};
  if (fwrite(header, sizeof(uint32_t), 5, fp) != 5) {
    fprintf(stderr, "Failed to write header\n");
    fclose(fp);
    return -1;
  }

  // Compute sizes (cast all operands to size_t to prevent overflow)
  size_t W1_count = (size_t)model->input_dim * (size_t)model->hidden_dim;
  size_t b1_count = (size_t)model->hidden_dim;
  size_t W2_count = (size_t)model->hidden_dim * (size_t)model->output_dim;
  size_t b2_count = (size_t)model->output_dim;
  size_t total_count = W1_count + b1_count + W2_count + b2_count;

  // Allocate host buffer for all weights
  float* h_weights = (float*)malloc(total_count * sizeof(float));
  if (!h_weights) {
    fprintf(stderr, "Failed to allocate host buffer for weights\n");
    fclose(fp);
    return -1;
  }

  // Copy weights from device to host
  float* ptr = h_weights;
  cudaError_t err;

  err = cudaMemcpy(ptr, model->d_W1, W1_count * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy W1 from device: %s\n",
            cudaGetErrorString(err));
    free(h_weights);
    fclose(fp);
    return -1;
  }
  ptr += W1_count;

  err = cudaMemcpy(ptr, model->d_b1, b1_count * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy b1 from device: %s\n",
            cudaGetErrorString(err));
    free(h_weights);
    fclose(fp);
    return -1;
  }
  ptr += b1_count;

  err = cudaMemcpy(ptr, model->d_W2, W2_count * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy W2 from device: %s\n",
            cudaGetErrorString(err));
    free(h_weights);
    fclose(fp);
    return -1;
  }
  ptr += W2_count;

  err = cudaMemcpy(ptr, model->d_b2, b2_count * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy b2 from device: %s\n",
            cudaGetErrorString(err));
    free(h_weights);
    fclose(fp);
    return -1;
  }

  // Write all weights to file
  if (fwrite(h_weights, sizeof(float), total_count, fp) != total_count) {
    fprintf(stderr, "Failed to write weights\n");
    free(h_weights);
    fclose(fp);
    return -1;
  }

  free(h_weights);
  fclose(fp);
  return 0;
}

int load_model(Model* model, const char* path) {
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "Failed to open %s for reading\n", path);
    return -1;
  }

  // Read and validate header
  uint32_t header[5];
  if (fread(header, sizeof(uint32_t), 5, fp) != 5) {
    fprintf(stderr, "Failed to read header\n");
    fclose(fp);
    return -1;
  }

  if (header[0] != MODEL_MAGIC) {
    fprintf(stderr, "Invalid model file (bad magic number)\n");
    fclose(fp);
    return -1;
  }

  if (header[1] != MODEL_VERSION) {
    fprintf(stderr, "Unsupported model version: %u (expected %u)\n", header[1],
            MODEL_VERSION);
    fclose(fp);
    return -1;
  }

  // Validate dimensions to prevent malicious model files
  if (header[2] > INT_MAX || header[3] > INT_MAX || header[4] > INT_MAX ||
      header[2] == 0 || header[3] == 0 || header[4] == 0) {
    fprintf(stderr, "Invalid model dimensions in file\n");
    fclose(fp);
    return -1;
  }

  model->input_dim = (int)header[2];
  model->hidden_dim = (int)header[3];
  model->output_dim = (int)header[4];

  // Compute sizes (cast all operands to size_t to prevent overflow)
  size_t W1_count = (size_t)model->input_dim * (size_t)model->hidden_dim;
  size_t b1_count = (size_t)model->hidden_dim;
  size_t W2_count = (size_t)model->hidden_dim * (size_t)model->output_dim;
  size_t b2_count = (size_t)model->output_dim;
  size_t total_count = W1_count + b1_count + W2_count + b2_count;

  // Allocate host buffer
  float* h_weights = (float*)malloc(total_count * sizeof(float));
  if (!h_weights) {
    fprintf(stderr, "Failed to allocate host buffer for weights\n");
    fclose(fp);
    return -1;
  }

  // Read all weights from file
  if (fread(h_weights, sizeof(float), total_count, fp) != total_count) {
    fprintf(stderr, "Failed to read weights (file may be truncated)\n");
    free(h_weights);
    fclose(fp);
    return -1;
  }
  fclose(fp);

  // Allocate device memory and copy weights
  float* ptr = h_weights;
  cudaError_t err;

  err = cudaMalloc(&model->d_W1, W1_count * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate W1: %s\n", cudaGetErrorString(err));
    free(h_weights);
    return -1;
  }
  err = cudaMemcpy(model->d_W1, ptr, W1_count * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy W1 to device: %s\n",
            cudaGetErrorString(err));
    cudaFree(model->d_W1);
    free(h_weights);
    return -1;
  }
  ptr += W1_count;

  err = cudaMalloc(&model->d_b1, b1_count * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate b1: %s\n", cudaGetErrorString(err));
    cudaFree(model->d_W1);
    free(h_weights);
    return -1;
  }
  err = cudaMemcpy(model->d_b1, ptr, b1_count * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy b1 to device: %s\n",
            cudaGetErrorString(err));
    cudaFree(model->d_W1);
    cudaFree(model->d_b1);
    free(h_weights);
    return -1;
  }
  ptr += b1_count;

  err = cudaMalloc(&model->d_W2, W2_count * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate W2: %s\n", cudaGetErrorString(err));
    cudaFree(model->d_W1);
    cudaFree(model->d_b1);
    free(h_weights);
    return -1;
  }
  err = cudaMemcpy(model->d_W2, ptr, W2_count * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy W2 to device: %s\n",
            cudaGetErrorString(err));
    cudaFree(model->d_W1);
    cudaFree(model->d_b1);
    cudaFree(model->d_W2);
    free(h_weights);
    return -1;
  }
  ptr += W2_count;

  err = cudaMalloc(&model->d_b2, b2_count * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate b2: %s\n", cudaGetErrorString(err));
    cudaFree(model->d_W1);
    cudaFree(model->d_b1);
    cudaFree(model->d_W2);
    free(h_weights);
    return -1;
  }
  err = cudaMemcpy(model->d_b2, ptr, b2_count * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy b2 to device: %s\n",
            cudaGetErrorString(err));
    cudaFree(model->d_W1);
    cudaFree(model->d_b1);
    cudaFree(model->d_W2);
    cudaFree(model->d_b2);
    free(h_weights);
    return -1;
  }

  free(h_weights);
  return 0;
}
