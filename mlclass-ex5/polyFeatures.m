function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 



%printf("SIZE OF X = \n");
%disp(size(X))


%printf("SIZE OF x_POLY = \n");
%disp(size(X_poly))


%printf(" X = \n");
%disp(X)


for i=1:p

%% map each power of X's elemnts directly to the i-th column of X_poly
X_poly(:,i)=X.^i;

end 

%printf(" X_POLY = \n");
%disp(X_poly)



% =========================================================================

end
