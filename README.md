# Neural Network

## MNIST

- steps:
  1.  Download MNIST dataset to:
      - TRAIN_IMAGE: `local/mnist/train-images.idx3-ubyte`
      - TRAIN_LABEL: `local/mnist/train-labels.idx1-ubyte`
      - TEST_IMAGE: `local/mnist/t10k-images.idx3-ubyte`
      - TEST_LABEL: `local/mnist/t10k-labels.idx1-ubyte`
  1.  Run:
      ```sh
      cargo test --release -- --include-ignored --nocapture mnist
      ```
  1.  Inspect parameters at: `local/mnist/params.ron`

## Backpropagation

- distribution of addends of $\frac{\partial G}{\partial f_1}$:
  ![](img/backpropagation.svg)
  - a part of the computation graph
  - $h_i : \mathbb{R}^{m_i} \to \mathbb{R}$
  - $f_j : \mathbb{R}^{n_j} \to \mathbb{R}$
  - $h_i$ are the successors of $f_j$
  - $G$ is the outmost function represented by the root node of the computation graph
  - $w$ is the tunable parameters of $f_1$
  - steps:
    1. nodes $h_1, h_2$ calculate the addends respectively
    1. nodes $h_1, h_2$ distribute the addends to $f_1, f_2$
    1. node $f_1$ calculates $\frac{\partial G}{\partial f_1}$ from the received addends
    1. node $f_1$ calculates $\frac{\partial G}{\partial w}$ using $\frac{\partial G}{\partial f_1}$
    1. node $f_1$ updates $w$ using $\frac{\partial G}{\partial w}$

## References

- the repo on which mine is based - <https://github.com/pim-book/programmers-introduction-to-mathematics>
