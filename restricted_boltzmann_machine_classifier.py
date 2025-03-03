import numpy as np
from sklearn.datasets import load_iris


# Guni Deyo Haness 
# RBM Semi-Supervised Classifier


class RBM:
    def __init__(self, num_visible, num_attached_visible, num_hidden):
        # Initialize weights and biases
        self.weights = np.random.normal(0, 0.1, (num_visible+num_attached_visible, num_hidden))
        self.visible_bias = np.random.normal(0, 0.1,(num_visible+num_attached_visible,))
        self.hidden_bias = np.random.normal(0, 0.1, (num_hidden,))
        
        # Initialize neurons
        self.visible_neurons=np.random.choice([0, 1], size=(num_visible,))
        self.attached_visible_neurons=None # Each sample will be assigned to those
        self.hidden_neurons=np.random.choice([0, 1], size=(num_hidden,))
    
    # Calculate the energy of the system based on the visible and hidden neurons
    def calculate_energy(self):
        # Energy function: E(v, h) = - sum(v_i * w_ij * h_j) - sum(b_i * v_i) - sum(a_j * h_j)
        energy = -np.sum(np.concatenate((self.visible_neurons,self.attached_visible_neurons)) @ self.weights * self.hidden_neurons) \
                 -np.sum(self.visible_bias * np.concatenate((self.visible_neurons,self.attached_visible_neurons))) - np.sum(self.hidden_bias * self.hidden_neurons)
        return energy
            
    def deduce(self, sample, single_iteration=False, attach=True, temparture=1.00, temparture_decay=0.99, energy_threshold=1e-5, iteration_to_calculate_energy=5):
        stablized=False
        prev_total_energy=np.inf
        iteration=1
        self.attached_visible_neurons=sample
        while not stablized:
            # update hidden neurons
            delta_energies = np.dot(np.concatenate((self.visible_neurons,self.attached_visible_neurons)), self.weights) + self.hidden_bias
            p_ks = 1 / (1 + np.exp(-delta_energies / temparture)) 
            self.hidden_neurons = (p_ks > np.random.rand(self.hidden_neurons.size)).astype(int)
            
            # update visible neurons
            delta_energies = np.dot(self.hidden_neurons, self.weights.T)+self.visible_bias
            p_ks = 1 / (1 + np.exp(-delta_energies[:self.visible_neurons.size] / temparture)) 
            self.visible_neurons = (p_ks > np.random.rand(self.visible_neurons.size)).astype(int)
            if not attach: # during training we won't attach the output neurons which represent the sample and change them also
                p_ks = 1 / (1 + np.exp(-delta_energies[self.visible_neurons.size:] / temparture)) 
                self.attached_visible_neurons=(p_ks > np.random.rand(self.attached_visible_neurons.size)).astype(int)
            
            # check energy change once in a few generations to determine if we stabilized and should stop running
            if iteration%iteration_to_calculate_energy==0:
                energy=self.calculate_energy()
                delta_total_energy=prev_total_energy-energy
                if delta_total_energy<energy_threshold:
                    stablized=True
                prev_total_energy=energy
                
            if single_iteration: # for training we will need to perform only one iteration
                break
            temparture*=temparture_decay
            iteration+=1
            
        prediction=np.where(self.visible_neurons==1)
        if prediction[0].size > 0:
            return prediction[0][0]
        return np.random.randint(3)
            
    # Contrastive training 
    def train(self, discretized_features, labels, num_epochs=10000, learning_rate=0.01):
        for _ in range(num_epochs):
            i=np.random.choice(discretized_features.shape[0]) # get random sample
            attached_visible_neurons=discretized_features[i]
            visible_neurons=np.eye(1, 3, labels[i]).flatten() # if the ith class is the true class, the ith neuron out of the 3 will be the only one turned on
            delta_energies = np.dot(np.concatenate((visible_neurons,attached_visible_neurons)), self.weights) + self.hidden_bias
            p_ks = 1 / (1 + np.exp(-delta_energies)) 
            self.deduce(attached_visible_neurons,single_iteration=True,attach=False)
            h1,v1=self.hidden_neurons,np.concatenate((self.visible_neurons,self.attached_visible_neurons))
            self.visible_bias=self.visible_bias+learning_rate*(np.concatenate((visible_neurons,attached_visible_neurons))-v1)
            self.hidden_bias=self.hidden_bias+learning_rate*(self.hidden_neurons-h1)
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    self.weights[i][j]=self.weights[i][j]+learning_rate*(np.concatenate((visible_neurons,attached_visible_neurons))[i] * p_ks[j] - v1[i]*h1[j])
                    
    def test(self, features, labels):
        accuracy=0
        for i in range(len(features)):
            attached_visible_neurons = features[i]
            prediction = self.deduce(attached_visible_neurons)
            if labels[i]==prediction: accuracy+=1
        accuracy/=len(features)
        return accuracy

def discretize_data(data, bins):
    # Discretize each feature into bins (each of the 4 sizes will be labled as one of the bins (if bins=3 it will be small, medium and big))
    discretized = np.zeros((data.shape[0], data.shape[1] * bins), dtype=int)
    for i in range(data.shape[1]):
        bin_indices = np.digitize(data[:, i], np.percentile(data[:, i], [100 / bins, 200 / bins]))
        for j, val in enumerate(bin_indices):
            discretized[j, i * bins + (val - 1)] = 1 
    return discretized

iris = load_iris()
features = iris.data # array of shape (150, 4) because there are 4 sizes for each feature.
                     # first 50 elements are the first kind, second 50 for the second and third 50 for the third
labels = iris.target # array of shape (150,). 0 is label for the first kind, 1 for the second and 2 for the third
discretized_features = discretize_data(features, bins=3)

num_visible=labels[-1]+1 # should be 3 neurons that represent a label prediction to one of the 3 classes
num_attached_visible = discretized_features.shape[1] # should be 4*bins neurons that represent a sample (4 sizes where each one is labled to one of bins neurons that represent the scale of the size)
num_hidden = num_attached_visible//2 
rbm = RBM(num_visible=num_visible,num_attached_visible=num_attached_visible ,num_hidden=num_hidden)

# shuffle data
permutation = np.random.permutation(len(labels))
shuffled_features = discretized_features[permutation]
shuffled_labels = labels[permutation]

# i split data 66 percent train 33 percent test
train_size=int(len(labels)*(2/3))
train_features,train_labels=shuffled_features[:train_size],shuffled_labels[:train_size]
test_features,test_labels=shuffled_features[train_size:],shuffled_labels[train_size:]

pre_training_accuracy=rbm.test(test_features,test_labels)
rbm.train(train_features,train_labels)
post_training_accuracy=rbm.test(test_features,test_labels)

print(f'Pre-training accuracy: {pre_training_accuracy * 100:.2f}%\n')
print(f'Post-training accuracy: {post_training_accuracy * 100:.2f}%\n')
