function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%


printf("in projectData func. -- K = %d \n",K);


printf("size of X =  \n");

disp(size(X));


printf("size of U=  \n");

disp(size(U));



% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% store the first k columns of U
U_reduce = U(:,1:K);



%% vectorized computation of reduced dimension variable Z . as K = 1, this returns a
%% (50*2)*(2*1) = 50 * 1 vector. i.e 50 pts in 1-D from 50 2-D pts , which is correct.
Z=X*U_reduce;



% =============================================================

end
