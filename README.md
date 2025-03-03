# Restricted Boltzmann Machine (RBM) Classifier by Guni

## Overview

This project implements a **Restricted Boltzmann Machine (RBM) classifier** to categorize the **Iris dataset** into three different species. The classifier undergoes training using a **contrastive divergence learning approach**, improving its accuracy significantly from the initial state.

### **Learning Approach:**
This RBM classifier is a **semi-supervised model**, meaning that it learns feature representations in an **unsupervised manner** but requires labeled data for classification and evaluation.

### **Best Results:**
- **Pre-training accuracy:** **34%**
- **Post-training accuracy:** **98%**
This project implements a **Restricted Boltzmann Machine (RBM) classifier** to categorize the **Iris dataset** into three different species. The classifier undergoes training using a **contrastive divergence learning approach**, improving its accuracy significantly from the initial state.

### **Best Results:**
- **Pre-training accuracy:** **34%**
- **Post-training accuracy:** **98%**

## Features

- **RBM-based classification**: Uses a generative model to classify data.
- **Contrastive divergence training**: Learns features efficiently over multiple epochs.
- **Feature discretization**: Transforms continuous data into discrete binary representations.
- **Temperature-based inference**: Uses a simulated annealing approach to stabilize predictions.
- **Energy function minimization**: Iteratively reduces the system energy to reach optimal classification.
- **Shuffled train-test split**: Ensures fair evaluation with a 66% training and 33% testing ratio.

## How It Works

1. **Feature Discretization**: The Iris dataset features are transformed into discrete binary representations using `np.digitize`.
2. **Network Structure**:
   - **Visible neurons**: Represent the class labels.
   - **Attached visible neurons**: Represent the input features.
   - **Hidden neurons**: Learn feature representations.
3. **Training Process**:
   - Contrastive divergence updates weights and biases iteratively.
   - Single-step Gibbs sampling is performed.
   - Energy minimization stabilizes hidden neurons.
4. **Testing**:
   - Predictions are made by activating visible neurons.
   - The classifier's accuracy is computed before and after training.

## Installation

Ensure you have Python installed with the following dependencies:

```sh
pip install numpy scikit-learn
```

## Running the Model

1. Clone the repository:
   ```sh
   git clone https://github.com/GuniDH/Restricted-Boltzmann-Machine-Semi-Supervised-Classifier.git
   cd Restricted-Boltzmann-Machine-Semi-Supervised-Classifier
   ```
2. Run the script:
   ```sh
   python restricted_boltzmann_machine_classifier.py
   ```
3. Observe the accuracy improvement:
   ```sh
   Pre-training accuracy: 34%
   Post-training accuracy: 98%
   ```

## File Structure
```
RBM-Classifier/
│── restricted_boltzmann_machine_classifier.py
│── README.md
```

## Example Code Usage

```python
from restricted_boltzmann_machine_classifier import RBM

rbm = RBM(num_visible=3, num_attached_visible=12, num_hidden=6)
rbm.train(train_features, train_labels)
accuracy = rbm.test(test_features, test_labels)
print(f"Accuracy after training: {accuracy * 100:.2f}%")
```

## Contributions

Contributions are welcome! Feel free to submit issues and pull requests to improve the model.

## License

This project is licensed under the **MIT License**.

---
### Author
**Guni**  

