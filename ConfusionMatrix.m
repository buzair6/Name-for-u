Confusion Matrix in MATLAB
% Example true labels and predicted labels
true_labels = [1, 1, 2, 2, 3, 3, 3];
predicted_labels = [1, 1, 2, 3, 3, 3, 3];

% Number of classes
num_classes = max(max(true_labels), max(predicted_labels));

% Initialize the confusion matrix
confusion_matrix = zeros(num_classes);

% Populate the confusion matrix
for i = 1:length(true_labels)
    confusion_matrix(true_labels(i), predicted_labels(i)) = confusion_matrix(true_labels(i), predicted_labels(i)) + 1;
end

% Display the confusion matrix
disp('Confusion Matrix:');
disp(confusion_matrix);
 