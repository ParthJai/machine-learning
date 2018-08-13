clc;
clear;
close;

data_class_1 = load('data1.txt');
data_class_2 = load('data2.txt');
data_class_3 = load('data3.txt');

test_set = data_class_1(26:35,:);       %randomly choose rows from data sets for test set
data_class_1(26:35, :) = [];

test_set = [test_set;data_class_2(22:31,:)];
data_class_2(22:31,:) = [];

test_set = [test_set;data_class_3(38:47, :)];
data_class_3(38:47, :) = [];


x1 = data_class_1(:, 1);        %extracting features from data set
x2 = data_class_1(:, 2);
x3 = data_class_1(:, 3);
x4 = data_class_1(:, 4);
y = data_class_1(:, 5);

x1 = [x1; data_class_2(:, 1)];
x2 = [x2; data_class_2(:, 2)];
x3 = [x3; data_class_2(:, 3)];
x4 = [x4; data_class_2(:, 4)];
y = [y; data_class_2(:, 5)];

x1 = [x1; data_class_2(:, 1)];
x2 = [x2; data_class_2(:, 2)];
x3 = [x3; data_class_2(:, 3)];
x4 = [x4; data_class_2(:, 4)];
y = [y; data_class_2(:, 5)];

x1 = [x1; data_class_3(:, 1)];
x2 = [x2; data_class_3(:, 2)];
x3 = [x3; data_class_3(:, 3)];
x4 = [x4; data_class_3(:, 4)];
y = [y; data_class_3(:, 5)];

data_set = [x1, x2, x3, x4, y];         %combine sepal length, sepal width, petal length and petal width from all classes separately in 4 columns
[r c] = size(data_set);
shuffledRow = randperm(r);
data_set = data_set(shuffledRow, :);    %shuffling rows of data set
x1 = data_set(:,1);
x2 = data_set(:,2);
x3 = data_set(:,3);
x4 = data_set(:,4);     %finally extract 4 feature vectors
y = data_set(:,5);



lambda = 0.1; %randomly chosen regularization paramter. One can also determine bias and variance for optimal value.
x = [x1, x2, x3, x4];
theta = zeros(3, 5);
initial_theta = zeros(5,1);
m = size(x,1);
n = size(x,2);
x = [ones(m,1) x]; %append ones in feature vector as fist column
options = optimset('GradObj', 'on', 'MaxIter', 50);
for j = 1:3
    theta(j,:) = fmincg(@(t)(costfunction(t, x, (y == j), lambda)), initial_theta, options);    %https://in.mathworks.com/matlabcentral/fileexchange/42770-logistic-regression-with-regularization-used-to-classify-hand-written-digits?focused=3791937&tab=function
end

%find training set accuracy
m = size(x, 1);
num_labels = size(theta, 1);
p = zeros(size(x, 1), 1);
hypo = x * theta';
prob = 1.0 ./ (1.0 + exp(-hypo));
[~,p] = max(prob, [], 2);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);

%find test set accuracy
test_set_y = test_set(:,5);
test_set(:,5) = [];
test_set = [ones(size(test_set,1),1) test_set];
choice = randperm(30);
count = 0;
for j = 1 : length(choice)
    test = choice(j);
    var = test_set_y(test);
    test = test_set(test, :);
    test_prob(j,:) = sigmoid(test * theta');
    [~,test_p] = max(test_prob, [], 2);
    var2 = test_p(j);
    if (var2 == var)
        count = double(count + 1);   %increase a variable if we do a correct prediction
    end
       
end
fprintf("Testing set accuracy: %f\n", (count / 30) * 100);  %no. of correct predictions / length of test set is accuracy??


function [J, grad] = costfunction(theta, x, y, lambda)     %function to compute cost function and gradient. These values are used by fmincg function to find optimal parameter vector.
temp = theta;
m = length(y);
J = 0;
grad = zeros(size(theta));
hypo = sigmoid(x);
J = ((1 / m) * sum(-y' * log(sigmoid(x * theta)) - (1 - y)' * log(1 - sigmoid(x * theta)))) + ((lambda / (2 * m))  * sum(theta(2:length(theta)) .* theta(2:length(theta))));
grad = ((1 / m) * sum( x .* repmat((sigmoid(x*theta) - y), 1, size(x,2))));
grad = grad';
grad(2:length(grad),1) = grad(2:length(grad),1) + (lambda / m) * theta(2:length(theta), 1);
grad = grad(:);
end

function g = sigmoid(x)     %function to compute hypothesis
g = 1.0 ./ (1.0 + exp(-x));
end