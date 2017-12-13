function theta=gradientDescent(X, y, theta, alpha, num_iters)
m=length(y);
for iter=1:num_iters
	grad =(X'*(sigmoid(X*theta)-y))/m;
	theta=theta-alpha*grad/m;
end
end