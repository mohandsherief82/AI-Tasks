import pandas as pd
import numpy as np

# Data preprocessing
train_data = pd.read_csv('train.csv')

# Shuffle data and reset the indices
train_data = train_data.sample(frac=1, random_state=45).reset_index(drop=True)

Y = pd.DataFrame(train_data['is_spam'][:])
X = pd.DataFrame(train_data.drop(columns=['is_spam', 'message_id']))

class LogisticRegression:
    
    # A private class attribute for the epsilon value to prevent log(0) errors
    __epsilon = 1e-15

    def __init__(self, learning_rate=0.01):
        """
        Initializes the Logistic Regression model with a specified learning rate.
        """
        self.learning_rate = learning_rate
        # Initialize the weights and bias
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        """
        Applies the sigmoid activation function to the input `z`.
        This function squashes any real-valued number to a value between 0 and 1,
        making it suitable for interpreting as a probability.
        """
        return 1 / (1 + np.exp(-z))

    def __predict_proba(self, data_points):
        """
        Calculates the probability of the positive class (y=1) for given data points.
        It can handle either a single data point (as a 1D NumPy array) or
        multiple data points (as a 2D NumPy array, where each row is a data point).

        Args:
            data_points (np.ndarray): Input features. Can be 1D for a single sample
                                     or 2D for a batch of samples.

        Returns:
            np.ndarray: Predicted probabilities, ranging from 0 to 1.
        """
        # Calculate the linear combination: z = X * weights + bias.
        z = np.dot(data_points, self.weights) + self.bias
        # Apply the sigmoid function to convert linear combination to probability
        return self._sigmoid(z)

    def __log_loss(self, y_true, y_pred):
        """
        Calculates the binary cross-entropy loss (log loss).
        This loss function is commonly used for binary classification problems.

        Args:
            y_true (np.ndarray): The true binary labels (0 or 1).
            y_pred (np.ndarray): The predicted probabilities (between 0 and 1).

        Returns:
            float: The calculated log loss.
        """
        # Clip predicted probabilities to prevent log(0) or log(1) issues.
        y_pred_clipped = np.clip(y_pred, self.__epsilon, 1 - self.__epsilon)
        # Binary Cross-Entropy formula: -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        return -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    def fit(self, X, y, epochs=1000, batch_size=None, gradient_descent_type='full'):
        """
        Fits the Logistic Regression model to the training data using the specified
        gradient descent optimization algorithm.

        Args:
            X (pd.DataFrame): Training features. Each row is a sample, each column is a feature.
            y (pd.DataFrame): Target labels. Should be a single column of binary (0 or 1) values.
            epochs (int): The maximum number of complete passes through the entire training dataset.
            batch_size (int, optional): The number of samples per gradient update.
                                        - For 'mini_batch', this must be a positive integer.
                                        - For 'sgd', it is internally set to 1.
                                        - For 'full', this parameter is ignored as the batch size
                                          is the total number of samples.
            gradient_descent_type (str): The type of gradient descent to use:
                                         - 'full': Full Batch Gradient Descent (default)
                                         - 'mini_batch': Mini-Batch Gradient Descent
                                         - 'sgd': Stochastic Gradient Descent
        Raises:
            ValueError: If an invalid `gradient_descent_type` is provided or
                        `batch_size` is invalid for 'mini_batch'.
        """
        n_samples, n_features = X.shape
        # Initialize weights with random values.
        self.weights = np.random.rand(n_features)
        self.bias = 0.0

        # Convert pandas DataFrames to NumPy arrays for efficient numerical operations.
        X_np = X.values
        y_np = y.values.flatten()

        # Validate the chosen gradient descent type
        if gradient_descent_type not in ['full', 'mini_batch', 'sgd']:
            raise ValueError("Invalid 'gradient_descent_type'. Choose from 'full', 'mini_batch', or 'sgd'.")

        # Determine the effective batch size based on the gradient descent type
        if gradient_descent_type == 'full':
            # Use all samples for each update
            effective_batch_size = n_samples
        elif gradient_descent_type == 'sgd':
            # Use one sample for each update
            effective_batch_size = 1
        elif gradient_descent_type == 'mini_batch':
            if batch_size is None or not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("For 'mini_batch' gradient descent, 'batch_size' must be a positive integer.")
            # Use the specified batch size
            effective_batch_size = batch_size

        print(f"Starting training with {gradient_descent_type.replace('_', ' ').title()} Gradient Descent (Batch Size: {effective_batch_size})...")

        # Training loop over epochs
        for epoch in range(1, epochs + 1):
            total_loss_train_epoch = 0

            # Iterate through the training data in batches
            for i in range(0, n_samples, effective_batch_size):
                # Extract the current batch of features and labels
                X_batch = X_np[i:i + effective_batch_size]
                y_batch = y_np[i:i + effective_batch_size]
                
                # Get the actual size of the current batch (important for the last batch)
                current_batch_size = len(X_batch)
                if current_batch_size == 0:
                    continue

                # Predict probabilities for the current batch
                yhat_batch = self.__predict_proba(X_batch)
                
                # Calculate the gradients for the weights and bias.
                # The gradient of the log-loss is the average of (y_pred - y_true).
                error = yhat_batch - y_batch
                
                weights_gradient = np.dot(X_batch.T, error) / current_batch_size
                bias_gradient = np.sum(error) / current_batch_size
                
                # Update the model parameters (weights and bias) using the gradients
                # and the learning rate.
                self.weights -= self.learning_rate * weights_gradient
                self.bias -= self.learning_rate * bias_gradient
                
                # Calculate the loss for the current batch and accumulate it for the epoch
                total_loss_train_epoch += np.sum(self.__log_loss(y_batch, yhat_batch))

            # Calculate average training loss for the epoch
            avg_train_loss = total_loss_train_epoch / n_samples
            # Print training loss for every epoch
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Average Loss: {avg_train_loss:.4f}")

        print("\nTraining completed all epochs.")

    def predict(self, X):
        """
        Predicts class labels (0 or 1) for the given input features (X)
        and saves these predictions to a CSV file named 'submission.csv'
        in the Kaggle working directory (`/kaggle/working/`).

        Args:
            X (pd.DataFrame): Features for which to make predictions.
                              Each row is a sample, each column is a feature.

        Returns:
            np.ndarray: An array of predicted class labels (0 or 1).
        """
        # Ensure weights are initialized before making predictions
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() before .predict().")

        # Store message_id if it exists, and create a feature-only DataFrame
        message_ids = None
        X_features = X.copy() # Create a copy to avoid modifying the original DataFrame

        # Check if 'message_id' column exists in the input DataFrame X
        if 'message_id' in X_features.columns:
            message_ids = X_features['message_id'] # Store the message_ids
            X_features = X_features.drop(columns=['message_id']) # Remove it from features

        # Calculate probabilities for all data points in X_features.
        probabilities = self.__predict_proba(X_features.values)

        # Convert probabilities to binary class predictions (0 or 1).
        predictions = (probabilities > 0.5).astype(int)

        # Prepare data for the results DataFrame.
        data_for_df = {'Predicted_Class': predictions}
        columns_for_df = ['Predicted_Class']

        # If message_ids were extracted, add them back to the output DataFrame
        if message_ids is not None:
            data_for_df['message_id'] = message_ids.values
            columns_for_df.insert(0, 'message_id') # Add message_id as the first column

        # Create a pandas DataFrame from the predictions and message_id (if present).
        results_df = pd.DataFrame(data_for_df, columns=columns_for_df)

        # Define the full path for the output CSV file in the Kaggle working directory.
        output_csv_path = 'submission.csv'

        # Save the DataFrame to a CSV file.
        results_df.to_csv(output_csv_path, index=False)

        print(f"\nPredictions successfully saved to: {output_csv_path}")

        # Return the predictions array for potential further in-memory use.
        return predictions
    

# Training
regressor = LogisticRegression(learning_rate=0.01)

regressor.fit(X=X, y=Y, epochs=100)

# Testing
test_data = pd.read_csv('/kaggle/input/f1-spam-detection/test.csv')

print(regressor.predict(test_data))