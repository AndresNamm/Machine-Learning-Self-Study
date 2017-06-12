function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% Cost calculations
% Cost without regularization
temp = -(1-y).*log(1-sigmoid(X*theta)) - (y).*log(sigmoid(X*theta));
J =  sum(temp)/m ;
% Add regularisation

J = J  +  sum(theta(2:end,1).**2) *  lambda / (2 * m );

% Gradient calculations

temp_theta = theta;
temp_theta(1,1) = 0;
grad = (X'*(sigmoid(X*theta)-y))./m + temp_theta .* lambda/m ;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% =============================================================

end
