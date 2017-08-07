function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features

printf("in grad. desc. n = \n");

disp(n)

J_history = zeros(num_iters, 1);




for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %intialize variables used in main loop 

    temp_1=0;
    temp_2=0;   
    h     =0;
    diff_operand_1=0;
    diff_operand_2=0;
    sum_operand_1=0;
    sum_operand_2=0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%UN - VECTORIZED APPROACH %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	

    % USE: theta_j+1 = theta_j - alpha/m*( SUM [ {h(x_i) - y_i} .x_i]);
    % and simultaneously update theta_0 and theta_1


	%training example loop to calculate 'sums'
	%for i = 1:m
	%	h=X(i,1)*theta(1,1) + X(i,2)*theta(2,1);  % compute i-th hypothesis func. value.
	%	diff_operand_1 = (h - y(i))*X(i,1);       % compute updtae sum component for each feature per training example
	%	diff_operand_2 = (h - y(i))*X(i,2); 

	%	sum_operand_1 = sum_operand_1 + diff_operand_1;
	%	sum_operand_2 = sum_operand_2 + diff_operand_2;
	%end


	%temp_1 =  theta(1,1) - (alpha/m)*(sum_operand_1);
	%temp_2 =  theta(2,1) - (alpha/m)*(sum_operand_2);

	%be sure to update elements simultaneously.
	%theta(1,1) = temp_1;
	%theta(2,1) = temp_2;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%VECTORIZED APPROACH %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	


	diff_operand=zeros(n,1);
        sum_operand=zeros(n,1);
	temp=zeros(n,1);

	
	%training example loop to calculate 'sums'
	for i = 1:m

		x_rowVect=X(i,:);         		                   %store each row of x for each training example

		h = x_rowVect * theta;                                     % compute i-th hypothesis func. value using vectorization.
		
		%no. of features loop
		for j = 1:n

			diff_operand(j) = X(i,j) * (h - y(i));
			sum_operand(j) = sum_operand(j) + diff_operand(j);   % compute updtae sum component for each feature per training example
		end

	end


	
	for k = 1:n
	
		temp(k) = theta(k,1) -  (alpha/m)*(sum_operand(k));

	end

	%update theta parameters simultaneously
	for k =1 :n

		theta(k,1) = temp(k);	

	end




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
