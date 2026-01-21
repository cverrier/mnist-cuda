#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

// Network parameters (all pointers are device memory)
typedef struct {
  float* d_W1;    // (input_dim, hidden_dim)
  float* d_b1;    // (hidden_dim,)
  float* d_W2;    // (hidden_dim, output_dim)
  float* d_b2;    // (output_dim,)
  int input_dim;  // 784
  int hidden_dim; // 128
  int output_dim; // 10
} Model;

// Training buffers (all pointers are device memory)
typedef struct {
  // Activations from forward pass (kept for backward)
  float* d_hidden; // (batch_size, hidden_dim) post-ReLU
  float* d_logits; // (batch_size, output_dim) pre-softmax

  // Gradients
  float* d_logits_grad; // (batch_size, output_dim) computed by compute_loss
  float* d_hidden_grad; // (batch_size, hidden_dim) intermediate
  float* d_W1_grad;     // (input_dim, hidden_dim)
  float* d_b1_grad;     // (hidden_dim,)
  float* d_W2_grad;     // (hidden_dim, output_dim)
  float* d_b2_grad;     // (output_dim,)

  // Loss (single scalar on device)
  float* d_loss;

  int batch_size;
} TrainState;

// === Lifecycle ===

// Allocate device memory, initialize weights (Xavier), biases (zero)
int create_model(Model* model, int input_dim, int hidden_dim, int output_dim);

// Allocate all training buffers on device
int create_train_state(TrainState* state, Model* model, int batch_size);

// Free device memory
void free_model(Model* model);
void free_train_state(TrainState* state);

// === Training ===

// Forward pass: input -> hidden (ReLU) -> logits
// Reads:  d_input (batch_size, input_dim) from device
// Writes: state->d_hidden, state->d_logits
void forward(Model* model, TrainState* state, float* d_input);

// Compute cross-entropy loss (fused with softmax for numerical stability)
// Also computes d_logits_grad = softmax(logits) - one_hot(labels)
// Reads:  state->d_logits, d_labels (batch_size,) as uint8
// Writes: state->d_loss (scalar), state->d_logits_grad
void compute_loss(Model* model, TrainState* state, uint8_t* d_labels);

// Backward pass: compute all parameter gradients
// Reads:  d_input, state->d_hidden, state->d_logits_grad, model->d_W2
// Writes: state->d_hidden_grad, d_W1_grad, d_b1_grad, d_W2_grad, d_b2_grad
void backward(Model* model, TrainState* state, float* d_input);

// SGD update: params -= learning_rate * gradients
void update_params(Model* model, TrainState* state, float learning_rate);

// === Evaluation ===

// Count correct predictions for a batch (call after forward pass)
// Compares argmax(d_logits) to d_labels for each sample
// Returns: number of correct predictions (0 to batch_size)
int count_correct(TrainState* state, uint8_t* d_labels);

// === Persistence ===

// Save model weights to a binary file
// File format: magic (4 bytes) | version (4 bytes) | dimensions (3x4 bytes) |
// weights Returns: 0 on success, -1 on error
int save_model(const Model* model, const char* path);

// Load model weights from a binary file
// Model must be uninitialized; this function allocates GPU memory
// Returns: 0 on success, -1 on error
int load_model(Model* model, const char* path);

#endif
