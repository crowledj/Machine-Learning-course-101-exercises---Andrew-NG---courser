#Support vector machine code
function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%


%% only 2 vectors (or 'training examples'), so 2 'comparisons' in kernel - betwenn x1 and itself and x1 and x2.
%% (comparison with itself = 1 , btw.)


%% intiialize all vectors
diff=zeros(3,1);
diff_sq=zeros(3,1);
diff_sq_sum=zeros(3,1);
arg=zeros(3,1);


diff = (x1 .- x2)

diff_sq = diff.^2;

diff_sq_sum= sum(diff_sq);

arg = (-diff_sq_sum)/(2*(sigma^2));


sim = exp(arg);


% =============================================================
    
end
