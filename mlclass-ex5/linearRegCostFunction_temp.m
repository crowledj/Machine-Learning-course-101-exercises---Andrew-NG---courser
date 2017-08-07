function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% =========================================================================
%%%%    		COMPUTE REGULARIZED COST                       %%%%
% =========================================================================

h=0;

printf("value of theta params initial is : \n");

disp(theta)


h=theta(1,1) + theta(2,1)*X(:,2);



diff =h-y;

printf("value of (h-y) is : \n");

disp(diff)


printf("old value of (h-y)^2 is : \n");

disp(sum(diff.^2))



printf("newly computed value of (h-y)^2 is : \n");

disp((diff'*diff))


J_noReg = (1/(2*m))*(sum(diff.^2));


printf("newly computed value of (h-y)^2/ (2*m) is : \n");

disp((diff'*diff)/(2*m))

reg_term=theta'*theta;

printf("value of sum(theta squared) is : \n");

disp(reg_term)


printf("lambda/m =  : \n");

disp(lambda/m)


J= J_noReg + (lambda/(2*m))*reg_term;


% =========================================================================
%%%%                  COMPUTE REGULARIZED GRADIENT                     %%%%
% =========================================================================


grad_noReg=(1/m)*(diff'*X);


grad = grad_noReg' + (lambda/(m))*(theta);


% =========================================================================

grad = grad(:);

end
