clc;
clear;
close;

input_layer_size  = 400;
num_labels = 10;
lambda = 0.1;
load('ex3data1.mat');
m = size(X, 1);
n = size(X, 2);
theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
for j = 1:num_labels
    theta(j,:) = fmincg(@(t)(lrCostFunction(t, X, (y == j), lambda)), initial_theta, options);
end
m = size(X, 1);
num_labels = size(theta, 1);
p = zeros(size(X, 1), 1);
hypo = X * theta';
prob = 1.0 ./ (1.0 + exp(-hypo));
[~,p] = max(prob, [], 2);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);


rp = randperm(m);
count = 1;
for i = 1:m
    fprintf('\nDisplaying Example Image\n');
    if count == 1
    X = X(:, 2:size(X,2));
    end
    displayData(X(rp(i), :));

    pred = predict_digit(theta, X(rp(i),:));
    fprintf('\npredicted digit: %d (digit %d)\n', pred, mod(pred, 10));
   
    s = input('Paused - press enter to continue, q to exit:','s');
    count= count - 1;
    if s == 'q'
      break
    end
end

function p = predict_digit(theta, X)
m = size(X, 1);
num_labels = size(theta, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
hypo = X * theta';
prob = 1.0 ./ (1.0 + exp(-hypo));
[~,p] = max(prob, [], 2);
end

