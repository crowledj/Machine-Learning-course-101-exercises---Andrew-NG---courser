function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


printf("m =  \n");
disp(size(X, 1))

%printf("size of Theta1 is: \n");
%disp(size(Theta1))

%printf("size of Theta2 is: \n");
%disp(size(Theta2))

%printf("before adding column size of X is: \n");
%disp(size(X))


%add a column of ones to the matrix X :
X = [ones(m, 1) X];


%printf("after adding column size of X is: \n");
%disp(size(X))
	

%compute the unit values for the hidden layer. (z^2 = theta1*a^1)
z_2=zeros(25,1);


%transpose X 
transpose_X = X';

%calculate the unit values in the hidden layer

z_2= Theta1*transpose_X;
a_2=sigmoid(z_2);


printf("size of a_2 is: \n");
disp(size(a_2))




%calculate the unit values for the output layer (hypothesis)

%add a row of ones to the hidden layer.invert a_2 first to add column.Then re- invert back to "add a row".

inv_a_2=a_2';

printf("size of inv_a_2 is: \n");
disp(size(inv_a_2))


inv_a_2=[ones(m,1) inv_a_2];
a_2=inv_a_2';

%printf("after re-arrangements size of a_2 is: \n");
%disp(size(a_2))


%compute the unit values for the output layer (hypothesis)

z_3=Theta2*a_2;
a_3=sigmoid(z_3);


%printf("size of a_3 is: \n");
%disp(size(a_3))

%printf("some of a_3 is: \n");
%disp(a_3(3,4:12))



%determine the predictions made using the neural network classifier (forward propagation) , by finding the max. probability per output layer matrix  row - element.
%invert a_3 first so that the examples are ordered row-wise.

a_3=a_3';

[a,b]=max(a_3,[],2);

%printf("size of b = \n");
%disp(size(b))


%printf("size of p = \n");
%disp(size(p))

%store these predictions in the p function output vector
p = b;


% =========================================================================


end
