function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 



Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
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















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
