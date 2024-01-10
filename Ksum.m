% Kohonen Self-Organizing Map (SOM) in MATLAB

% Define the training data
data = rand(100, 2);  % Example random 2D data

% SOM parameters
grid_size = [5, 5];   % Size of the SOM grid
learning_rate = 0.1;  % Initial learning rate
epochs = 100;         % Number of training epochs

% Initialize SOM weights
weights = rand(grid_size(1), grid_size(2), size(data, 2));

% Training the SOM
for epoch = 1:epochs
    for i = 1:size(data, 1)
        % Find the winning neuron (Best Matching Unit)
        [~, winner] = min(sum((squeeze(weights - data(i, :))).^2, 3));
        
        % Update the weights of the neighborhood
        for row = 1:grid_size(1)
            for col = 1:grid_size(2)
                % Calculate the distance to the winner neuron
                distance = sqrt((row - winner(1))^2 + (col - winner(2))^2);
                
                % Update the weights using the neighborhood function
                influence = exp(-distance^2 / (2 * learning_rate^2));
                weights(row, col, :) = weights(row, col, :) + learning_rate * influence * (data(i, :) - weights(row, col, :));
            end
        end
    end
    
    % Update learning rate for the next epoch
    learning_rate = learning_rate * 0.9;
end

% Display the SOM grid
figure;
scatter(data(:, 1), data(:, 2), 'b');
hold on;

% Plot the SOM grid
for row = 1:grid_size(1)
    for col = 1:grid_size(2)
        plot(weights(row, col, 1), weights(row, col, 2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    end
end
hold off;
title('Kohonen SOM');
