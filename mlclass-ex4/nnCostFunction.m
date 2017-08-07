function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% ------------------------------------------------------------%
%%% Part 1: Feedforward the neural network and return the cost
% ------------------------------------------------------------%


%%  initialise recoded_y, hypothesis vector and all layer units %%

recoded_y=zeros(num_labels,m);

h=zeros(num_labels,m);

z_2=zeros(hidden_layer_size,m);
a_2=zeros(hidden_layer_size,m);
z_3=zeros(num_labels,m);


%% compute 're-coded' y  label vector %%
for i =1:m
	num =y(i);
	recoded_y(num,i)=1;
end


%% add bias column vector to X %%
X=[ones(m,1) X];



%% compute 'hidden' layer units %%

z_2=Theta1*X';
a_2=sigmoid(z_2);


%% compute 'hidden' layer units %%
%(add a row to the hidden layer matrix)%


a_2=[ones(1,m);a_2];


%% compute 'output' layer units %%

z_3=Theta2*a_2;
h=sigmoid(z_3);


%% compute Cost J %%

for i =1:m

	row_y=recoded_y'(i,:);
	col_h=h(:,i);

	J = J + (-row_y*log(col_h)-(1-row_y)*log(1-col_h));

end


J = J/m;


% -----------------------------------------------------------------------%
%%% Part 1: Feedforward the neural network and return the Regularized cost
% -----------------------------------------------------------------------%


hidden_layer_term=0;
output_layer_term=0;

% calc. hidden layer term %

for j = 1:hidden_layer_size

	%% exclude column corresponding to bias layer in sum %%
	hidden_layer_term =  hidden_layer_term + sum(Theta1(j,2:input_layer_size+1).^2); 
end

% calc. output layer term %

for j = 1:num_labels

	%% exclude column corresponding to bias layer in sum %%
	output_layer_term =  output_layer_term + sum(Theta2(j,2:hidden_layer_size+1).^2);  
end

%% compute the regularized cost function inc. regularization term using old cost function.
J = J + (lambda/(2*m))*(hidden_layer_term + output_layer_term);




% -----------------------------------------------------------------------%
%%%    Part 2: backpropagation algorithm to compute the gradients
% -----------------------------------------------------------------------%

%% initialize all variables that will be needed in back - prop. %%

a_1=zeros(input_layer_size,1);
a_2=zeros(hidden_layer_size,1);
a_3=zeros(num_labels,1);
z_2=zeros(hidden_layer_size,1);
z_3=zeros(num_labels,1);


delta_2=zeros(hidden_layer_size+1,1);
delta_3=zeros(num_labels,1);

Delta_grad_2=zeros(num_labels,hidden_layer_size+1);
Delta_grad_1=zeros(hidden_layer_size,input_layer_size+1);


%     ------------------------------------   %
%     Main loop over all training examples
%     ------------------------------------   %

for t=1:m

	%     ------------------------------------   %
	%     Comute activations along each layer
	%     ------------------------------------   %

	a_1=X(t,:);

	%% compute hidden layer activation value for a given example.  %%

	z_2=Theta1*a_1';
	a_2=sigmoid(z_2);

	%% compute output layer activation value %%

	a_2=[1;a_2];

	z_3=Theta2*a_2;
	a_3=sigmoid(z_3);


	%     ------------------------------------   %
	%     	Compute errors for output layer
	%     ------------------------------------   %

	y_k=recoded_y(:,t);

	delta_3= (a_3-y_k);

	%     ------------------------------------   %
	%     	Compute errors for hidden layer
	%     ------------------------------------   %

	grad_sig_z_2= sigmoidGradient(z_2);
	grad_sig_z_2=[1;grad_sig_z_2];

	delta_2=(Theta2'*delta_3).*grad_sig_z_2;


	%     ------------------------------------   %
	%     	    Accumulate the gradients
	%     ------------------------------------   %

	%% for hidden layer %%

	Delta_grad_2=Delta_grad_2 .+ (delta_3*a_2');



	%% for input layer %%

	delta_2 = delta_2(2:end);	
	Delta_grad_1=Delta_grad_1 .+ (delta_2*a_1);

end

	%     ---------------------------------------------------------------   %
	%     		 Obtain the unregularized gradient
	%     ---------------------------------------------------------------   %

	Theta1_grad=Delta_grad_1/m;
	Theta2_grad=Delta_grad_2/m;


	%     ---------------------------------------------------------------   %
	%      Obtain the regularized gradient(when lambda not equal to zero)
	%     ---------------------------------------------------------------   %

	Theta1(:,1)=zeros(hidden_layer_size,1);
	Theta2(:,1)=zeros(num_labels,1);

	Theta1_grad=Theta1_grad + (lambda/m)*(Theta1);
	Theta2_grad=Theta2_grad + (lambda/m)*(Theta2);




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
