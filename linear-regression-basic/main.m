clc; 
clear;
close all;

num_iterations = 1500;
alpha = 0.0100;
data = load('ex1data2.txt');
X1 = data(:,1);
X2 = data(:,2);
y = data(:,3);
X = [X1 X2];
mu = mean(X);
sigma = std(X);
X = (X - mu) ./ sigma;
m = length(X);
X = [ones(m,1) X];
theta = zeros(size(X,2),1);
J = zeros(1, num_iterations);

for i = 1:num_iterations
    error = (X * theta) - y;
    temp0 = theta(1) - alpha * 1 / m * sum(error.*X(:,1));
    temp1 = theta(2) - alpha * 1 / m * sum(error.*X(:,2));
    temp2 = theta(3) - alpha * 1 / m * sum(error.*X(:,3));
    theta = [temp0;temp1;temp2];
    cost = X * theta;
    J(i) = (0.5 / m) * sum((cost - y).^2);
end
sqfeet = input('enter the size in sq feet: ');
numBdRoom = input('enter number of bedrooms: ');
temp = [1 sqfeet numBdRoom];
temp(2) = (temp(2) - mu(1))/(sigma(1));
temp(3) = (temp(3) - mu(2))/(sigma(2));
price = temp * theta;
fprintf("Cost is: ");
disp(price);
plot(J);