clc;
clear;
close;

input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;
load('ex4data1.mat');
m = size(X, 1);
load('ex4weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];
lambda = 1;
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
a1 = [ones(m,1) X];
hypo = a1 * Theta1';
a2 = sigmoid(hypo);
a2 = [ones(size(a2,1),1) a2];
hypo = a2 * Theta2';
a3 = sigmoid(hypo);
hypo = a3;
yvec = zeros(m, num_labels);
for i = 1:m
    yvec(i, y(i)) = 1;
end

cost = (-yvec .* log(hypo)) - ((1 - yvec) .* log(1 - hypo));
J = (1 / m) * sum(sum(cost));
reg_theta1 = Theta1(:,2:size(Theta1,2)).* Theta1(:,2:size(Theta1,2));
reg_theta2 = Theta2(:,2:size(Theta2,2)) .* Theta2(:,2:size(Theta2,2));
reg_theta1 = sum(sum(reg_theta1));
reg_theta2 = sum(sum(reg_theta2));
J = J + (lambda / (2*m)) * (reg_theta1 + reg_theta2);
count = 0;
for i = 1 : m
    a_1 = [X(i,:)];
    a_1 = [1 a_1];
    z2 = a_1 * Theta1';
    a_2 = sigmoid(z2);
    a_2 = [ones(size(a_2,1),1) a_2];
    a_3 = sigmoid(a_2 * Theta2');
    a_1 = a_1';
    a_2 = a_2';
    a_3 = a_3';
    delta3 = a_3 - yvec(i,:)';
    delta2 = Theta2' * delta3 .* (a_2 .* (1 - a_2));
    delta2 = delta2(2:end,:);
    Theta1_grad = Theta1_grad + delta2 * (a_1)';
    Theta2_grad = Theta2_grad + delta3 * (a_2)';

end
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));
initial_Theta1 = randomWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randomWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 50);
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);
rp = randperm(m);
count = 1;
X = [ones(size(X,1),1) X];
for i = 1:m
    fprintf('\nDisplaying Example Image\n');
    if count == 1
    X = X(:, 2:size(X,2));
    end
    displayData(X(rp(i), :));

    pred = predict(Theta1,Theta2, X(rp(i),:));
    fprintf('\npredicted digit: %d (digit %d)\n', pred, mod(pred, 10));
   
    s = input('Paused - press enter to continue, q to exit:','s');
    count= count - 1;
    if s == 'q'
      break
    end
end

function W = randomWeights(L_in, L_out)
W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end
