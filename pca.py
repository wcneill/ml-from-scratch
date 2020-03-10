import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # get variable names
    cols = ['sepal length', 'sepal width', 'petal length', 'petal width', 'classification']

    # PCA Step 1: get n samples of m-dimensional data (sample vectors s_1 to s_n, which hare samples of variables
    # x_1 to x_m)

    # read data (long form, for plotting in seaborn):
    df = pd.read_csv('data/iris.data', names=cols)

    # set classification column data type to string
    df['classification'] = df['classification'].astype("string")

    # set index by classification for easier access to subsets of the data
    df.set_index('classification', inplace=True, drop=False)
    classification = df.pop('classification')
    print(df)

    # set figure size:
    palette = sb.color_palette("husl", 3)
    x, y = ("petal length", "petal width")
    sb.relplot(x=x, y=y, data=df, hue=classification, palette=palette)
    sb.lmplot(x=x, y=y, data=df, palette=palette)
    # plt.show()


    # PCA Step 2: compute the average of each variable
    mean = df.mean(0) # average of each column (average of each variable)

    # PCA Step 3: We have the average of each variable in a row vector, now subtract it from each row.
    Z = df.apply(lambda x: x - mean, axis=1).to_numpy()

    # PCA Step 4: find the covariance matrix (1/n)(Z.T * Z):
    C = (1 / (len(Z) - 1)) * (Z.T @ Z)

    # PCA Step 5: find the eigenvalues and eigenvectors of the covariance matrix C
    vals, vects = np.linalg.eig(C)

    # PCA Step 5: sort the eigenvalues and eigenvectors in decreasing order:
    idx = np.argsort(vals)[::-1]  # returns a list of indices that represent descending order of eigenvalues.
    vals = vals[idx]
    vects = vects[:, idx]

    # PCA Step 6: Compose matrix P of columns made from sorted eigenvectors (C = PDP^(-1)):
    # This step is already done, as numpy returns eigenvectors as a list of lists, where the eigenvectors are in the
    # columns.

    # Step 7: Reconstitute our original data by multiplying it by our matrix P:
    Z_prime = Z @ vects

    











# Build the matrix B, which is the column matrix comprized of those vectors adjusted by the mean

# Compute S = BB_t

# find and sort the eigenvalues of S in decreasing order, as well as the corresponding eigenvectors

# If certain lambdas are significantly larger than others, this indicates correlation and possibility for dimension
# Reduction
