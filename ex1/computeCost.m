ufunction J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



%initialize the hypothesis func. value and the diff, sum_sqs vars.

h=0;
diff_sq=0;
sum_sqs=0;

%compute squared difference between hypothesis and output vector


%un-vectorized approach
%for i = 1 : m
%	h=X(i,1)*theta(1,1) + X(i,2)*theta(2,1);  % compute i-th hypothesis func. value.
%	diff_sq = (h - y(i))*(h - y(i));          % compute i-th hypothesis func. value.

%	sum_sqs = sum_sqs + diff_sq;

%end

%J = (1/(2*m))*sum_sqs;



%vectorized approach

for i = 1 : m

x_rowVect=X(i,:);                %store each row of x for each training example

h= x_rowVect * theta;		     %compute it's corresponding hypothesis function

diff_sq=(h - y(i)).*(h - y(i));  %compute the squre of difference between output vector elements and hypoth. func.

sum_sqs =  sum_sqs + diff_sq;    %keep a running sum of these squared differences

end

J = (1/(2*m))*sum_sqs;


% =========================================================================

end


std::vect<double> row_vect;


row_vect.push_back()

