function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

       %X_subset=zeros(size(X(1:i, :))   

       theta_test=zeros(2,1);

       %theta_test=trainLinearReg(X, y, lambda);


       for i = 1:m
            %Compute train/cross validation errors using training examples 

            X_subset=X(1:i, :);
	    y_subset=y(1:i);

	    %% optimized (trained) theta must be calculated for each i - i.e on each subset of the
	    %% training data.This is because we are looking for the error in the model as a function of 	    %% training example numbers.This means is has to be 'trained' on each subset before being evaluated.
	    [theta_test]=trainLinearReg(X_subset,y_subset,lambda);
	    
            %error_train(i)= (1/(2*m))*(((X_subset*theta_test)-y_subset)')*((X_subset*theta_test)-y_subset);
            error_train(i)=linearRegCostFunction(X_subset,y_subset,theta_test,0);  		


  	    X_val_subset=Xval(1:i, :);
	    y_val_subset=yval(1:i);
	    %error_val(i)= (1/(2*m))*((X_val_subset*theta_test)-y_val_subset)'*((X_val_subset*theta_test)-y_val_subset);
	    
	    %% As in the assignment description, the cross -validatin error must be computed on the entire set 		       each time as it is then a fair comparison each time.The training error above is done on each     	       subset  of the data as it is a reflection of how well it is at fitting the data it has just  
	    %% been trained on.

	    error_val(i)=linearRegCostFunction(Xval,yval,theta_test,0);	
            
       end





printf("out of error calc . loop !! \n");


% -------------------------------------------------------------

% =========================================================================

end
