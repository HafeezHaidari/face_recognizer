'''
Import all the necessary libraries
'''
import numpy as np
import pickle
import data_loader


class NeuralNetwork:
    '''
    Define the NeuralNetwork class with fixed architecture
    '''
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size, alpha=0.01, lambd=0.01):
        # He initialization for weights
        self.W1 = np.random.randn(hidden_size1, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size1, 1))
        self.W2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((hidden_size2, 1))
        self.W3 = np.random.randn(hidden_size3, hidden_size2) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((hidden_size3, 1))
        self.W4 = np.random.randn(hidden_size4, hidden_size3) * np.sqrt(2. / hidden_size3)
        self.b4 = np.zeros((hidden_size4, 1))
        self.W5 = np.random.randn(output_size, hidden_size4) * np.sqrt(2. / hidden_size4)
        self.b5 = np.zeros((output_size, 1))
        
        # Learning rate
        self.alpha = alpha
        # L2 regularization factor 
        self.lambd = lambd  
    
    # Activation functions and their derivatives
    @staticmethod
    def LeakyReLU(Z, alpha=0.01):
        return np.where(Z > 0, Z, Z * alpha)

    @staticmethod
    def LeakyReLU_deriv(Z, alpha=0.01):
        return np.where(Z > 0, 1, alpha)

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_deriv(A):
        return A * (1 - A)

    # Forward propagation
    def forward_prop(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.LeakyReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.LeakyReLU(self.Z2)
        self.Z3 = self.W3.dot(self.A2) + self.b3
        self.A3 = self.LeakyReLU(self.Z3)
        self.Z4 = self.W4.dot(self.A3) + self.b4
        self.A4 = self.LeakyReLU(self.Z4)
        self.Z5 = self.W5.dot(self.A4) + self.b5
        self.A5 = self.sigmoid(self.Z5)  # Output layer with sigmoid
        return self.A5

    # Loss function
    def compute_loss(self, Y, A5):
        # Number of examples (since Y is 1D, use shape[0])
        m = Y.shape[0] 
        # Binary cross-entropy loss
        loss = -1/m * np.sum(Y * np.log(A5 + 1e-8) + (1 - Y) * np.log(1 - A5 + 1e-8))
        # Add L2 regularization
        reg_loss = (self.lambd / (2 * m)) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) +
                                         np.sum(np.square(self.W3)) + np.sum(np.square(self.W4)) + np.sum(np.square(self.W5)))
        return loss + reg_loss

    # Backward propagation
    def backward_prop(self, X, Y):
        m = X.shape[1]

        # Gradient of the loss with respect to A5
        dZ5 = self.A5 - Y
        dW5 = 1 / m * dZ5.dot(self.A4.T) + (self.lambd / m) * self.W5
        db5 = 1 / m * np.sum(dZ5, axis=1, keepdims=True)

        # Gradient for the forth hidden layer
        dZ4 = self.W5.T.dot(dZ5) * self.LeakyReLU_deriv(self.Z4)
        dW4 = 1 / m * dZ4.dot(self.A3.T) + (self.lambd / m) * self.W4
        db4 = 1 / m * np.sum(dZ4, axis=1, keepdims=True)

        # Gradient for the third hidden layer
        dZ3 = self.W4.T.dot(dZ4) * self.LeakyReLU_deriv(self.Z3)
        dW3 = 1 / m * dZ3.dot(self.A2.T) + (self.lambd / m) * self.W3
        db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

        # Gradient for the second hidden layer
        dZ2 = self.W3.T.dot(dZ3) * self.LeakyReLU_deriv(self.Z2)
        dW2 = 1 / m * dZ2.dot(self.A1.T) + (self.lambd / m) * self.W2
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        
        # Gradient for the first hidden layer
        dZ1 = self.W2.T.dot(dZ2) * self.LeakyReLU_deriv(self.Z1)
        dW1 = 1 / m * dZ1.dot(X.T) + (self.lambd / m) * self.W1
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5

    # Update weights and biases
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, alpha):
        # Update weights and biases with gradient descent
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        self.W3 -= self.alpha * dW3
        self.b3 -= self.alpha * db3
        self.W4 -= self.alpha * dW4
        self.b4 -= self.alpha * db4
        self.W5 -= self.alpha * dW5
        self.b5 -= self.alpha * db5

    # Get predictions
    def get_predictions(self):
        return (self.A5 > 0.5).astype(int).flatten()  # Ensure it returns a 1D array

    # Get accuracy
    def get_accuracy(self, predictions, Y):
        predictions = predictions.flatten()  # Ensure predictions are 1D
        return np.sum(predictions == Y) / Y.size

    # Mini-batch gradient descent
    def gradient_descent(self, X, Y, batch_size, iterations):
        m = X.shape[1]  # number of training examples

        for i in range(iterations):
            for j in range(0, m, batch_size):
                # Mini-batch selection
                X_batch = X[:, j:j+batch_size]
                Y_batch = Y[j:j+batch_size]  # Fix for Y being 1D

                # Forward and backward propagation for the mini-batch
                self.forward_prop(X_batch)
                dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5 = self.backward_prop(X_batch, Y_batch)
                self.update_params(dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, self.alpha)
            print(i)
            if i % 10 == 0:
                # Compute accuracy on the entire training set
                self.forward_prop(X)  # Forward propagate on the full dataset
                predictions = self.get_predictions()
                accuracy = self.get_accuracy(predictions, Y)
                loss = self.compute_loss(Y, self.A5)  # Compute loss for the full dataset
                print(f"Iteration: {i}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5



    # Save and load model
    def save_model(self, filename):
        model = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
            'W4': self.W4,
            'b4': self.b4,
            'W5': self.W5,
            'b5': self.b5
        }
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        self.W1 = model['W1']
        self.b1 = model['b1']
        self.W2 = model['W2']
        self.b2 = model['b2']
        self.W3 = model['W3']
        self.b3 = model['b3']
        self.W4 = model['W4']
        self.b4 = model['b4']
        self.W5 = model['W5']
        self.b5 = model['b5']
        print(f"Model loaded from {filename}")
   
   
# Main function     
def main(data_loader, name):
    # Load data
    X_train, Y_train, X_dev, Y_dev, m_train = data_loader.split_data()

    # Initialize the neural network
    nn = NeuralNetwork(input_size=16384, hidden_size1=8192, hidden_size2=4096, hidden_size3=1024, hidden_size4=512, output_size=1, alpha=0.001, lambd=0.01)
    
    # Train the model using mini-batch gradient descent
    nn.gradient_descent(X_train, Y_train, batch_size=64, iterations=10)
    
    # Save the trained model
    nn.save_model(f'test_model.pkl')
    
    # Evaluate the model on the development set
    nn.forward_prop(X_dev)
    dev_predictions = nn.get_predictions()
    dev_accuracy = nn.get_accuracy(dev_predictions, Y_dev)
    print(f"Development Set Accuracy: {dev_accuracy:.4f}")

# Run the main function    
if __name__ == '__main__':
    data_loader = data_loader.DataLoader(r'C:\Users\zstro\Downloads\complete_image_data.csv')
    main(data_loader, 'test_model')
