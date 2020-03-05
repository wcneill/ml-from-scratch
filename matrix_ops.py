import numpy as np
import sympy as sp


def mprint(A):
    for i in A:
        print(i)
    print('\n')


# Dot product function from scratch
def dprod(v1, v2):
    return sum((a * b for a, b in zip(v1, v2)))


# Write a function to do matrix vector multiplication from scratch:
def mvprod(M, v):
    result = []
    for row in M:
        element = sum((a * b for a, b in zip(row, v)))
        result.append(element)
    return result


# Write a function that multiplies matrices
def mprod(M, N):
    mrows = len(M)
    ncols = len(N[0])
    result = [[] for i in range(mrows)]

    for i in range(mrows):
        for j in range(ncols):
            row = M[i]
            col = [nrow[j] for nrow in N]
            element = sum((r * c for r, c in zip(row, col)))
            result[i].append(element)
    return result


# transpose a matrix:
def transpose(M):
    cols = len(M[0])
    result = []
    for i in range(cols):
        tcol = [row[i] for row in M]
        result.append(tcol)
    return result


# Rotate a matrix 90 degrees clockwise (using numpy)
def rot90cw(M):
    nM = np.array(M)
    return np.flip(nM, 0).T


# Rotate a matrix 90 degrees ccw
def rot90ccw(M):
    numcols = len(M[0])
    temp = []
    final = []

    for i in range(numcols):
        temp.append([row[i] for row in M])

    numrows = len(temp)

    for j in range(numrows):
        final.append(temp[numrows - 1 - j])

    return final


# element wise sum of two vectors


# element wise difference of two vectors.


if __name__ == '__main__':
    # create a new ndarray from a nested list
    A = [[1, 2, 3],
         [2, 4, 6],
         [1, 1, 1]]

    B = [[1, 2],
         [1, 1],
         [1, 3]]

    x = [1, 2]
    y = [1, 2, 3]
    z = [1, 1, 1]

    print("dot product using numpy and my function: ")
    print(dprod(y, z))
    print(np.dot(y, z), '\n')

    print("Matrix vector multiplication: ")
    mprint(mvprod(B, x))
    print(np.array(B) @ np.array(x), '\n')

    print("Matrix multiplication :")
    mprint(mprod(A, B))
    print(np.array(A) @ np.array(B), '\n')

    print("Transpose with numpy and my function:")
    mprint(transpose(B))
    print(np.transpose(np.array(B)), '\n')

    print("rotate matrix 90 degrees clockwise: ")
    mprint(B)
    mprint(rot90cw(B))

    # counter clock-wise
    print("rotate matrix 90 degrees counter-clockwise: ")
    mprint(B)
    mprint(rot90ccw(B))

    # matrix creation with numpy (random, zeros, ones, reshape)
    print("practice with ndarray creation using random, zeros, ones and reshape")
    C = np.random.randint(1, 3, 4).reshape(2, 2)
    mprint(C)
    D = np.zeros((4, 4))
    mprint(D)

    # show that numpy.reshape and ndarray.reshape both return reshaped references to the original object, not a copy.
    E = D.reshape((2, 8))
    print(E, '\n')
    E[0, 0] = 1
    print(E, '\n')
    print(D, '\n')

    F = np.reshape(E, (2, 8))
    print(F, '\n')
    F[0, 0] = 2
    print(D, '\n')

    # finding maximum and minimum elements:
    X = np.random.randint(1, 11, 25).reshape(5, 5)
    print(X)

    print("The maximum value of any element in X is {}, and occurs at index {}.".format(np.max(X), np.argmax(X)))

    print("find the maximum sum along rows and columns")
    rowsums = np.array([sum(row) for row in X])
    colsums = np.array([sum(X[:, i]) for i in range(len(X[0]))])
    print("max row sum: {}".format(np.max(rowsums)))
    print("max col sum: {} \n".format(np.max(colsums)))

    print("practice appending rows and columns, using the sums above")
    print(X.shape)
    print(colsums.shape)
    print(rowsums.shape)

    colsums = colsums.reshape(1, 5)
    rowsums = rowsums.reshape(5, 1)

    # all_data = np.append(X, rowsums, axis=1)
    all_data = np.append(X, colsums, axis=0)

    print("X with sums appended as an additional row or column:")
    print(all_data, '\n')

    # Matrix Powers
    print("X^2= \n", np.linalg.matrix_power(X, 2), '\n')

    # practice slicing lists (1-D) (works the same for Numpy ndarrays)
    list1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sublist = list1[0:5]

    # slicing lists in 2-D (must use list comprehension, as 2-D slicing is unavailable for lists)
    list2 = [list1, list1, list1]
    list3 = [row[0:3] for row in list2]
    print(list3, '\n')

    # practice slicing 2-D with Numpy ndarrays (recreate list3 but with better slicing capabilities)
    list4 = np.array(list2)
    list3 = list4[0:2, 0:3]
    print(list3, '\n')

    # ***NOTE*** that using slicing in numpy returns a view of the original ndarray. Any changes made to the slice
    # will be reflected in the original. In order to avoid this, you must make a copy.

    # finding eigenvalues and vectors with numpy
    A = np.array([[-2, -9],
                  [1, 4]])

    print(A)

    vals, vecs = np.linalg.eig(A)
    print("numpy eigenvalues: {}".format(vals))
    print("numpy eigenvects: {} \n".format(vecs))

    A = sp.Matrix(A)
    print(type(A))
    vals = A.eigenvals() # dictionary of key value {pairs eigenvalue:algebraic multiplicity}
    vects = A.eigenvects() # list of tuples of form (egenvalue, algebraic multiplicity, [eigenvectors]
    print(vals)
    print(vects)

    print("eigenvector: \n ", vects[0][2])





