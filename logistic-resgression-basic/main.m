clc; 
clear;
close;

data = load('ex2data1.txt');
X = data(:,[1, 2]);
y = data(:, 3);
[m,n] = size(X);
X = [ones(m, 1) X];
X1 = X(:,1);
X2 = X(:,2);
X3 = X(:,3);
theta = zeros(n + 1, 1);
m = length(y);
J = 0;
alpha = 0.0100;
grad = zeros(size(theta));
hypo = sigmoid(X * theta);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), theta, options);
J = zeros(1, 2000);

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), theta, options);

plotDecisionBoundary(theta, X, y);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
m1= input('enter marks in first exam');
m2 = input('enter marks in secod exam');
prob = sigmoid([1 m1 m2] * theta);
fprintf(['For a student with scores %f and %f, we predict an admission ' ...
         'probability of %f\n'], m1, m2, prob);