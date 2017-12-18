function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
n=size(X,2);
error1=-(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)))/m;
theta1=theta(2:n);
error2=sum(theta(2:n).^2)*lambda/(2*m);
J = error1+error2;
grad=ones(n,1);
grad(1)=sum(sigmoid(X*theta)-y)/m;
grad(2:n)=(X'(2:n,:)*(sigmoid(X*theta)-y))/m+lambda*theta(2:n)/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
