import numpy as np

a = np.zeros((2,2))                		# Creates an array 2x2 of all zeros
b = np.ones((1,2))						# Creates an array 1x2 of all ones
c = np.full((2,2), 7)					# Creates a constant array of 7s
d = np.eye(2)							# Creates a 2x2 identity array [[1,0],[0,1]]
e = np.random.random((2,2))				# Create an array filled with random values

# Sum function
x = np.array([[1,2],[3,4]])

print np.sum(x)				# Computes sum of all elements; prints 10
print np.sum(x, axis = 0)	# Computes sum of each column; prints [4 6]
print np.sum(x, axis = 1)	# Computes sum of each row; prints [3 7]

# Transpose

print x			# prints [[1 2]
				#		  [3 4]]
print x.T		# prints [[1 3]
				# 		   2 4]]

v = np.array([1,2,3])	#transposing of a rank 1 does nothing
print v 				#prints [1 2 3]
print v.T				#prints [1 2 3]

# Broadcasting

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])
y = np.empty_like(x)		# empty matrix with the same shape as x

y = x + v			# Add the vector v to each row of the
	 				# matrix x with an explicit loop

print y 					# Now y is the following
							# [[ 2  2  4]
							#  [ 5  5  7]
							#  [ 8  8 10]
							#  [11 11 13]]

# Some more applications of Broadcasting

v = np.array([1,2,3])	# v has shape (3,)
w = np.array([4,5])		# w has shape (2,)

print np.reshape(v, (3,1)) * w 		# To compute an outer product, we first reshape v to be a column
									# vector of shape (3, 1); we can then broadcast it against w to yield
									# an output of shape (3, 2), which is the outer product of v and w:
									# [[ 4  5]
									#  [ 8 10]
									#  [12 15]]

x = np.array([[1,2,3],[4,5,6]])		# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
print x + v 						# giving the following matrix:
									# [[2 4 6]
									#  [5 7 9]]

print (x.T + w).T 					# x has shape (2, 3) and w has shape (2,).
									# If we transpose x then it has shape (3, 2) and can be broadcast
									# against w to yield a result of shape (3, 2); transposing this result
									# yields the final result of shape (2, 3) which is the matrix x with
									# the vector w added to each column. Gives the following matrix:
									# [[ 5  6  7]
									#  [ 9 10 11]]

print x + np.reshape(w, (2,1))		# same output [[ 5  6  7]
									#  			   [ 9 10 11]]

print x * 2 						# [[ 2  4  6]
									#  [ 8 10 12]]	

# Distance between points

from scipy.spatial.distance import pdist, squareform

x = np.array([[0, 1], [1, 0], [2, 0]])	# [[0 1]
print x									#  [1 0]
										#  [2 0]]

d = squareform(pdist(x,'euclidean'))	# Compute the Euclidean distance between all rows of x.
print d 								# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
										# and d is the following array:
										# [[ 0.          1.41421356  2.23606798]
										#  [ 1.41421356  0.          1.        ]
										#  [ 2.23606798  1.          0.        ]]
						



