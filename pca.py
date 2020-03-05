import seaborn as sb

# get n samples of m-dimensional data (vectors x_1 to x_n)

S = np.array([[95, 1], [1, 5]])

vals, vecs = np.linalg.eig(S)

print(vals)
print(vecs)

# compute the mean mu of said vectors

# Build the matrix B, which is the column matrix comprized of those vectors adjusted by the mean

# Compute S = BB_t

# find and sort the eigenvalues of S in decreasing order, as well as the corresponding eigenvectors

# If certain lambdas are significantly larger than others, this indicates correlation and possibility for dimension
# Reduction