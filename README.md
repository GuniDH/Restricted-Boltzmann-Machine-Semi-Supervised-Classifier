# Restricted Boltzmann Machine (RBM) Classifier by Guni

## Overview

This project implements a **Restricted Boltzmann Machine (RBM) classifier** to categorize the **Iris dataset** into three different species. The classifier undergoes training using a **contrastive divergence learning approach**, improving its accuracy significantly from the initial state.

### **Learning Approach:**
This RBM classifier is a **semi-supervised model**, meaning that it learns feature representations in an **unsupervised manner** but requires labeled data for classification and evaluation.

### **Best Results:**
- **Pre-training accuracy:** **34%**
- **Post-training accuracy:** **98%**

## Features

- **RBM-based classification**: Uses a generative model to classify data.
- **Contrastive divergence training**: Learns features efficiently over multiple epochs.
- **Feature discretization**: Transforms continuous data into discrete binary representations.
- **Temperature-based inference**: Uses a simulated annealing approach to stabilize predictions.
- **Energy function minimization**: Iteratively reduces the system energy to reach optimal classification.

## How It Works

1. **Feature Discretization**: The Iris dataset features are transformed into discrete binary representations
2. **Network Structure**:
   - **Visible neurons**: Represent the class labels.
   - **Attached visible neurons**: Represent the input features.
   - **Hidden neurons**: Learn feature representations.
3. **Training Process**:
   - Contrastive divergence updates weights and biases iteratively.
   - Energy minimization stabilizes hidden neurons.
4. **Testing**:
   - Predictions are made by activating visible neurons.
   - The classifier's accuracy is computed before and after training.

---
### Author
**Guni**  

