# Import necessary libraries
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def fit(self, X, y, learning_rate, num_epochs, X_test, y_test):
        # Training the logistic regression model with manual gradient computation
        accuracies = []  # Store accuracies for plotting
        
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(X)
            logits = self.sigmoid(outputs)

            # Calculate loss manually with epsilon term
            epsilon = 1e-7  # Small epsilon to prevent log(0)
            loss = -((y * torch.log(logits + epsilon) + (1 - y) * torch.log(1 - logits + epsilon)).mean())

            # Manual gradient computation
            dL_dy = logits - y

            # Backward pass (compute gradients)
            self.zero_grad()
            grad_weight = torch.mm(X.t(), dL_dy)
            grad_bias = dL_dy.sum()

            # Update model parameters manually
            with torch.no_grad():
                self.linear.weight -= learning_rate * grad_weight.t()
                self.linear.bias -= learning_rate * grad_bias

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}]')
            
            # Calculate accuracy for this epoch
            predicted = self.predict(X_test)
            accuracy = (predicted.view(-1) == y_test.view(-1)).float().mean()
            accuracies.append(accuracy)
        
        # Plot accuracy over epochs
        plt.plot(range(1, num_epochs + 1), accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.grid(True)
        plt.show()

    def predict(self, data):
        with torch.no_grad():
            outputs = self(data)
            logits = self.sigmoid(outputs)
            return (logits >= 0.5).float()

def run_logistic_regression(data_x, data_y):
    print(data_x)
    print(data_y)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

    # Convert the data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).view(-1, 1)  # Reshape y_train to a column vector
    y_test = torch.FloatTensor(y_test).view(-1, 1)  # Reshape y_test to a column vector

    # Initialize the model and choose hyperparameters
    input_size = 430  # Number of features in the Iris dataset
    learning_rate = 0.01
    num_epochs = 1000

    model = LogisticRegression(input_size)
    print(model)

    # Train the model
    model.fit(X_train, y_train, learning_rate, num_epochs, X_test, y_test)
    
    # Test the model
    predicted = model.predict(X_test)
    accuracy = (predicted.view(-1) == y_test.view(-1)).float().mean()
    print(f'Accuracy on the test set: {accuracy:.2f}')
