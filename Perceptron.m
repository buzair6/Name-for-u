% Perceptron Model in MATLAB

% Define the training data
X = [1 0; 0 1; 1 1; 0 0];  % Input features
Y = [1; 1; 1; 0];          % Corresponding labels

% Initialize weights and bias
weights = randn(1, size(X, 2));
bias = randn();

% Set learning rate and number of epochs
learning_rate = 0.1;
epochs = 1000;

% Training the perceptron
for epoch = 1:epochs
    for i = 1:size(X, 1)
        % Compute the predicted output
        prediction = sum(X(i, :) .* weights) + bias;
        
        % Apply step function as activation
        output = prediction > 0;
        
        % Update weights and bias using the perceptron learning rule
        error = Y(i) - output;
        weights = weights + learning_rate * error * X(i, :);
        bias = bias + learning_rate * error;
    end
end

% Test the trained perceptron
test_data = [1 1; 0 1; 1 0; 0 0];
predictions = sum(test_data .* weights, 2) + bias;
final_outputs = predictions > 0;

% Display the results
disp('Trained Weights:');
disp(weights);
disp('Trained Bias:');
disp(bias);
disp('Test Data Predictions:');
disp(final_outputs);

Backpropagation
% Backpropagation in MATLAB

% Define the training data
X = [0 0; 0 1; 1 0; 1 1];  % Input features
Y = [0; 1; 1; 0];          % Corresponding labels

% Initialize neural network parameters
input_size = size(X, 2);
hidden_size = 4;
output_size = 1;

% Initialize weights and biases
W1 = randn(input_size, hidden_size);
b1 = zeros(1, hidden_size);
W2 = randn(hidden_size, output_size);
b2 = zeros(1, output_size);

% Set learning rate and number of epochs
learning_rate = 0.01;
epochs = 10000;

% Training the neural network with backpropagation
for epoch = 1:epochs
    % Forward pass
    a1 = X;
    z2 = a1 * W1 + b1;
    a2 = sigmoid(z2);
    z3 = a2 * W2 + b2;
    output = sigmoid(z3);
    
    % Compute the cost (mean squared error)
    cost = 0.5 * sum((output - Y).^2) / size(X, 1);
    
    % Backward pass
    delta3 = (output - Y) .* sigmoid_derivative(z3);
    delta2 = (delta3 * W2') .* sigmoid_derivative(z2);
    
    % Update weights and biases
    W2 = W2 - learning_rate * (a2' * delta3) / size(X, 1);
    b2 = b2 - learning_rate * sum(delta3) / size(X, 1);
    W1 = W1 - learning_rate * (a1' * delta2) / size(X, 1);
    b1 = b1 - learning_rate * sum(delta2) / size(X, 1);
end

% Test the trained neural network
test_data = [0 0; 0 1; 1 0; 1 1];
test_output = predict(test_data, W1, b1, W2, b2);
