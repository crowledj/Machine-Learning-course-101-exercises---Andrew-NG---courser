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

%%%  NOTE THIS DID NOT WORK FOR COMMENTED OUT h and reg_term variables below , as these were not written generally in vectorized form.


h=0;

h=theta(1) + theta(2:end,1)'*X(:,2:end)';


diff=h'-y;


J_noReg = (1/(2*m))*(sum(diff.^2));


temp_theta=theta;
temp_theta(1)=0;
reg_term=temp_theta'*temp_theta;


J= J_noReg + (lambda/(2*m))*reg_term;


% =========================================================================
%%%%                  COMPUTE REGULARIZED GRADIENT                     %%%%
% =========================================================================

grad_noReg=(1/m)*(diff'*X);


%% do a vactorised calc. for grad but remove reg. for first term by setting it to zero.
temp=theta;
temp(1)=0;

grad = grad_noReg' + (lambda/(m))*(temp);

% =========================================================================

grad = grad(:);

end
