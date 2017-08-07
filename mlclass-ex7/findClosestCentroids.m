function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);


printf("size of centriods vector is : \n");

disp(size(centroids))


printf("centriods vector is : \n");

disp(centroids)


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

num=size(X,1);

tmp_vec = zeros(1,size(X,2));


small=zeros(K,1);


idx=zeros(K,1);

for i = 1 : num

	tmp_vec = X(i,:);
	min_val=100000000;

	for j = 1 : K

		#store each row vector to use in follwing computations
		tmp_centroid = centroids(j,:);
		
		dist = tmp_vec .- tmp_centroid;
		dist_sq = dist .* dist;
		dist_sq_sum = (sum(dist_sq .+ dist_sq))/2;					
			
		#printf("training eg. i = \n");

		#disp(i)

		#printf("X(i) = \n");

		#disp(X(i,:))

		#printf(" j = \n")

		#disp(j)

		#printf("dist is equal to : \n")

		#disp(dist)

		#printf(" and squared dist for centroid j  \n")
   
                #printf("is equal to : \n")

		#disp(dist_sq)


		#printf(" and value of centroid j  \n")
					

		#disp(centroids(j,:))



		#printf(" and sum of these squared dists for centroid j  \n")

	        #printf("is equal to : \n")

		#disp(dist_sq_sum)

		#printf("OUTSIDE min loop --  for i = %d -- and j = %d -- selected small value is : \n",i,j);
		#disp(dist_sq_sum)


		if( dist_sq_sum < min_val)
		
			min_val = dist_sq_sum;
			small(i)=dist_sq_sum;

			#printf("got into min loop --  for i = %d -- and j = %d -- selected small value is : \n",i,j);
			#disp(dist_sq_sum)

			idx(i) = j;
		end

	end

end



% =============================================================

end

