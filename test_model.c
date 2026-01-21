#include "model.h"
#include <stdio.h>

int main(void) {
  printf("Testing model creation and memory management...\n");

  // Test model creation
  Model model;
  if (create_model(&model, 784, 128, 10) != 0) {
    fprintf(stderr, "FAILED: create_model\n");
    return -1;
  }
  printf("PASSED: create_model (dims: %d -> %d -> %d)\n", model.input_dim,
         model.hidden_dim, model.output_dim);

  // Test train state creation
  TrainState state;
  if (create_train_state(&state, &model, 64) != 0) {
    fprintf(stderr, "FAILED: create_train_state\n");
    free_model(&model);
    return -1;
  }
  printf("PASSED: create_train_state (batch_size: %d)\n", state.batch_size);

  // Test memory cleanup
  free_train_state(&state);
  printf("PASSED: free_train_state\n");

  free_model(&model);
  printf("PASSED: free_model\n");

  printf("All tests passed!\n");
  return 0;
}
