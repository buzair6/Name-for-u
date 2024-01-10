% Forward Propagation in MATLAB

% Define the input data
input_data = randn(1, 3);  % Example input with 3 features

% Define the neural network architecture
input_size = 3;
hidden_size = 4;
output_size = 2;

% Initialize weights and biases
weights_input_hidden = randn(input_size, hidden_size);
bias_hidden = randn(1, hidden_size);
weights_hidden_output = randn(hidden_size, output_size);
bias_output = randn(1, output_size);

% Forward Propagation
hidden_activation = input_data * weights_input_hidden + bias_hidden;
hidden_output = sigmoid(hidden_activation);  % Assuming sigmoid activation for the hidden layer
output_activation = hidden_output * weights_hidden_output + bias_output;
final_output = sigmoid(output_activation);  % Assuming sigmoid activation for the output layer

% Display the results
disp('Input Data:');
disp(input_data);
disp('Hidden Layer Output:');
disp(hidden_output);
disp('Final Output:');
disp(final_output);

