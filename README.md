# Fashion-MNIST Classification with PyTorch

Multilayer Perceptron implemented in PyTorch for multi-class image classification on the Fashion-MNIST dataset.

## Features

- Feed-forward neural network with two hidden layers
- ReLU activations
- CrossEntropyLoss
- SGD optimizer with momentum
- Fashion-MNIST classification
- Accuracy monitoring during training
- Confusion matrix evaluation

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-Learn
- Seaborn

## Model Architecture

Input (784)
→ Linear(256)
→ ReLU
→ Linear(128)
→ ReLU
→ Linear(10)

## What I Learned

- Building neural networks with PyTorch
- Working with DataLoaders and datasets
- Training multi-class classifiers
- Using CrossEntropyLoss and SGD with momentum
- Evaluating models using confusion matrices

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```