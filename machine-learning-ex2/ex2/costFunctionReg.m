function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;
for i=1:m
    h = sigmoid(X(i,:)*theta);
    sum = sum + (-y(i)*log(h)-(1-y(i))*log(1-h));
end
psm = 0;
for j=2:n 
    % j starts from 2, since j=1 (theta0) shall not be concluded when calculating square sum
    psm = psm + theta(j)^2;
end
J = sum/m + lambda*psm/(2*m);


grad_sum = zeros(size(theta));
for i=1:m
    h = sigmoid(X(i,:)*theta);
    grad_sum = grad_sum + (h-y(i))*X(i,:)';
end
for j=2:n
    % j starts from 2, since j=1 (theta0) is different
    grad_sum(j) = grad_sum(j) + lambda*theta(j);
end
grad = grad_sum/m;


% =============================================================

end
