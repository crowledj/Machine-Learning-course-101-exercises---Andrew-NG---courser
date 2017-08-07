function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

printf("m = \n");
disp(m);


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

%printf("value of theta params initial is : \n");
%disp(theta)

%h=theta(1,1) + theta(2,1)*X(:,2);

h=theta(1) + theta(2:end,1)'*X(:,2:end)';


%printf("value of h is : \n");
%disp(h)


diff=h'-y;

%printf("value of (h-y) is : \n");
%disp(diff)


%printf("old value of (h-y)^2 is : \n");
%disp(sum(diff.^2))


%printf("newly computed value of (h-y)^2 is : \n");
%disp((diff'*diff))


J_noReg = (1/(2*m))*(sum(diff.^2));

%printf("value of J_noReg is : \n");
%disp(J_noReg)


%printf("newly computed value of (h-y)^2/ (2*m) is : \n");
%disp((diff'*diff)/(2*m))

temp_theta=theta;

temp_theta(1)=0;

%reg_term=theta(2,1)*theta(2,1);

reg_term=temp_theta'*temp_theta;


%printf("value of sum(theta squared) is : \n");
%disp(reg_term)

%printf("lambda/m =  : \n");
%disp(lambda/m)


J= J_noReg + (lambda/(2*m))*reg_term;


% =========================================================================
%%%%                  COMPUTE REGULARIZED GRADIENT                     %%%%
% =========================================================================

printf("size of J = ");
disp(size(J))


grad_noReg=(1/m)*(diff'*X);


%% do a vactorised calc. for grad but remove reg. for first term by setting it to zero.
temp=theta;
temp(1)=0;

grad = grad_noReg' + (lambda/(m))*(temp);

printf("size of grad = ");
disp(size(grad))

% =========================================================================

grad = grad(:);

end
