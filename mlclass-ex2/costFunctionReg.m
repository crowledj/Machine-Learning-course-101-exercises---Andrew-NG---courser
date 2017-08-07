function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  my work - cost function calculation %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initializations
arg=0;
h=0;


%calculate vecorized arg for hypothesis
arg = X*theta;


h=sigmoid(arg);

temp_J = (1/m)*sum(-y.*log(h) - (1-y).*(log(1-h)));

%printf("temp_J = \n");
%disp(temp_J);


%printf("size theta = \n");
%disp(size(theta));


%%this term must be error in causing submission of reg. coest to fail ...??! 
%square_theta= sum(theta(1,end).^2);

temp_theta=theta;
temp_theta(1)=0;

square_theta= temp_theta'*temp_theta;

J = temp_J + (lambda/(2*m))*square_theta;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  gradient calculation %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


beta = h-y;

grad = 1/m*(X'*beta);


temp = theta;
temp(1) = 0;


grad= grad + (lambda/m)*temp; 


% =============================================================

end
