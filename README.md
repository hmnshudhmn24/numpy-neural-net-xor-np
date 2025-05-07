
# 🧠 NumPy Neural Net From Scratch

This project implements a simple Multi-Layer Perceptron (MLP) neural network from scratch using NumPy to solve the XOR problem.

## 🚀 Features

- Pure NumPy implementation
- Custom activation functions (Sigmoid)
- Manual backpropagation algorithm
- Trains on the XOR truth table
- Interactive prediction after training

## 📊 XOR Truth Table

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |   0    |
|    0    |    1    |   1    |
|    1    |    0    |   1    |
|    1    |    1    |   0    |

## 🛠️ How to Run

1. Ensure you have Python installed.
2. Install NumPy if not already installed:

   ```bash
   pip install numpy
   ```

3. Run the script:

   ```bash
   python main.py
   ```

## 🧪 Sample Output

```
Epoch 0, Loss: 0.25
...
Epoch 9000, Loss: 0.001
Final Output after Training:
[[0.01]
 [0.98]
 [0.98]
 [0.02]]
Prediction for [0, 1]: [[0.98]]
```

## 📚 Learn More

For a deeper understanding of neural networks and backpropagation, consider exploring the following resources:

- [Building an XOR Neural Network from Scratch](https://medium.com/@derek246810/building-an-xor-neural-network-from-scratch-learn-from-the-basics-63a2a22495ae)
- [Implementing a Simple Neural Network with NumPy](https://medium.com/@raza.mehar/implementing-a-simple-neural-network-with-numpy-a-comprehensive-guide-ffd5e077274c)
