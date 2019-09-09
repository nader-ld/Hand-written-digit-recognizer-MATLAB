function [J, grad] = CostFunction(theta, X, y, lambda)
%   LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%   regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J0=1/m*sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)));
grad0=1/m*(X'*(sigmoid(X*theta)-y));
J=J0+lambda/(2*m)*sum(theta(2:end).^2);
grad=grad0+lambda/m*theta;
grad(1)=grad0(1);

grad = grad(:);

end
