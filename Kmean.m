KMeans
% K-Means Clustering in MATLAB

% Define the data
data = [randn(100, 2); randn(100, 2) + 5];  % Example 2D data with two clusters

% Set the number of clusters
k = 2;

% Initialize centroids randomly
centroids = datasample(data, k, 'Replace', false);

% Set the number of iterations
max_iterations = 100;

% Initialize variables
previous_centroids = zeros(size(centroids));
iterations = 0;

while ~isequal(centroids, previous_centroids) && iterations < max_iterations
    % Assign each data point to the nearest centroid
    distances = pdist2(data, centroids);
    [~, labels] = min(distances, [], 2);
    
    % Update centroids based on the assigned points
    for i = 1:k
        centroids(i, :) = mean(data(labels == i, :));
    end
    
    % Update iteration count and store previous centroids
    iterations = iterations + 1;
    previous_centroids = centroids;
end

% Display the results
figure;
scatter(data(:, 1), data(:, 2), 30, labels, 'filled');
hold on;
plot(centroids(:, 1), centroids(:, 2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
title('K-Means Clustering');
