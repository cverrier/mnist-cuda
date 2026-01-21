# mnist-cuda

CUDA-accelerated neural network for MNIST digit classification.

> [!NOTE]
> This project is for educational purposes, and the CUDA kernels are not optimized for performance at all.

## Features

- GPU-accelerated training with custom CUDA kernels
- Complete training pipeline from raw IDX data to saved model weights
- Achieves ~97.75% validation accuracy
- Fused softmax + cross-entropy loss computation for numerical stability
- Xavier weight initialization

## Architecture

**Network:** 784 (input) → 128 (hidden, ReLU) → 10 (output logits)

```
MNIST IDX files → load_mnist_images() → MNISTDataset (uint8)
                                              ↓
                                    normalize_mnist()
                                              ↓
                                    NormalizedMNIST (float)
                                              ↓
                                    cudaMemcpy to GPU
                                              ↓
                            ┌─────────────────────────────────────┐
                            │         Training Loop               │
                            │  forward() → compute_loss()         │
                            │      → backward() → update_params() │
                            └─────────────────────────────────────┘
```

**CUDA kernels:**
- `linear_relu_kernel` - Fused linear + ReLU for hidden layer
- `linear_kernel` - Linear transformation for output
- `softmax_cross_entropy_kernel` - Fused softmax, loss, and gradient
- `matmul_at_b_kernel` - A^T × B for weight gradients
- `bias_grad_kernel` - Sum over batch for bias gradients
- `hidden_grad_kernel` - Backprop through layer 2 with ReLU mask
- `sgd_kernel` - Parameter update
- `count_correct_kernel` - Accuracy computation

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (default path: `/usr/local/cuda`)
- NVCC compiler
- Clang with C23 support
- Make

## Getting the Data

Download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place the files in `data/mnist/`:

```
data/mnist/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

Throughout this project, we use the test images as a validation set.

## Building

Set `CUDA_PATH` if CUDA is not installed at `/usr/local/cuda`:

```bash
export CUDA_PATH=/path/to/cuda
```

Build commands:

```bash
make all          # Build display_digits and tests
make train        # Build and run training
make display      # Build and run digit display
make clean        # Remove build artifacts
```

Build outputs go to the `bin/` directory.

## Training

Run training:

```bash
make train
# or
./bin/train
```

Example output:

```
=== MNIST CUDA Training ===

Loading training data...
  Loaded 60000 training images (28x28)
Loading test data...
  Loaded 10000 test images (28x28)
Copying data to GPU...
  Training: 179.44 MB, Test: 29.91 MB
Creating model...
  Architecture: 784 -> 128 -> 10
  Batch size: 64

Starting training: 10 epochs, 937 batches/epoch, lr=0.0100

Epoch  1/10: loss=0.4567, train_acc=92.15%, test_acc=92.43%
Epoch  2/10: loss=0.2634, train_acc=94.72%, test_acc=94.89%
...
Epoch 10/10: loss=0.0712, train_acc=98.21%, test_acc=97.75%

Training complete!
Saving model to model.bin...
Model saved successfully
```

**Metrics:**
- `loss` - Average cross-entropy loss over all batches
- `train_acc` - Classification accuracy on training set
- `test_acc` - Classification accuracy on test set

Model weights are saved to `model.bin`.

## Displaying Digits

View MNIST digit samples as ASCII art:

```bash
make display
# or
./bin/display_digits [n_samples]
```

Example:

```
Loaded 60000 images (28x28) with labels

=== Sample 0 | Label: 5 ===



         .+@@%*
        *@@@@@#
       .@@%=+#=
       -@#
       +@*
       -@@@@%+.
        =#@@@@#
          .=*@@-
             *@*
             +@*
    .       .@@=
   -@*     .%@%.
   .%@%+--+@@#.
    .+#@@@@#-

```

By default, 5 samples are displayed. Pass a number to display more:

```bash
./bin/display_digits 10
```

## Project Structure

```
├── model.cu/h        # CUDA neural network implementation
├── train.cu          # Training loop and data loading to GPU
├── display_digits.c  # Standalone digit display program
├── mnist.c/h         # MNIST IDX file parser and normalization
├── bswap.c/h         # Byte-swap utilities for big-endian headers
├── display.c/h       # ASCII visualization of digits
├── test_*.c          # Test files
└── Makefile          # Build configuration
```

## Testing

```bash
make test       # Run C tests (bswap, mnist loading)
make test-cuda  # Run CUDA model tests
```

## License

MIT License
