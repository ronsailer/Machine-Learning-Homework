import numpy as np
from scipy.misc import face
from scipy.stats import norm
import scipy as sp
import matplotlib.pyplot as plt
import timeit


def one_hot_5_of_10():
    """
    Return a zero vector of size 10 but the fifth value which is 1
    """
    # +++your code here+++
    vec = np.zeros(10)
    vec[4] = 1
    return vec


def negate_3_to_8(x):
    """
    Given a 1D array, negate all elements which are between 3 and 8
    """
    # +++your code here+++
    return np.vectorize(lambda t: -t if t>3 and t<8 else t)(x)


def get_size_properties(x):
    """
    Given an array x, return a tuple with the following properties:
    (num_rows, num_cols, num_elements, num_dimensions)
    """
    # +++your code here+++
    return x.shape + (x.size, x.ndim)


def append_vector_to_matrix(x, y):
    """
    Append row vector y to the end (bottom) of matrix x
    """
    # +++your code here+++
    return np.vstack((x, y))


def column_sum(x):
    """
    Return a vector containing the sum of each column of x
    """
    # +++your code here+++
    return np.sum(x, axis=0)


def multiplication_table():
    """
    print the multiplication table ("lu'ach ha'kefel") using Python's broadcasting
    """
    # +++your code here+++
    vec = np.arange(1,11)
    v = np.reshape(vec,(10,1))
    print v*vec

def view_face():
    """
    View the face image using Scipy's scipy.misc.face() and display the image
    """
    # +++your code here+++
    plt.imshow(face())
    plt.show()


def q1():
    a = np.arange(4)
    b = a[2:4]
    b[0] = 10
    return a


def q2():
    a, b = np.meshgrid(np.arange(4), np.arange(0, 30, 10))
    mesh = a + b
    return mesh


def plot_samples(sample, x):
    """
    Fill in the missing lines to match the titles of the subplots
    """
    plt.figure()

    plt.subplot(2,2,1)
    plt.title('Normal Random Variable')
    plt.plot(sample)

    sample = sorted(sample)

    plt.subplot(2,2,2)
    plt.title('Probability Distribution Function')
    plt.plot(sample, norm.pdf(sample))

    plt.subplot(2,2,3)
    plt.title('Cumulative Distribution Function')
    plt.plot(sample, norm.cdf(sample))

    plt.subplot(2,2,4)
    plt.title('Percent Point Function')
    plt.plot(sample, norm.ppf(sample))
    plt.show(block=True)


def seed_zero():
    """
    Seed numpy's random generator with the value 0
    """
    # +++your code here+++
    np.random.seed(0)


def test(got, expected):
    """
    Simple provided test() function used in main() to print
    what each function returns vs. what it's supposed to return.
    """
    if got == expected:
        prefix = ' OK '
    else:
        prefix = '  X '
    print('%s got: %s expected: %s' % (prefix, repr(got), repr(expected)))


def test_array(got, expected):
    if np.array_equal(got, expected):
        prefix = ' OK '
    else:
        prefix = '  X '
    print('%s got:\n    %s\n expected:\n    %s' % (prefix, repr(got), repr(expected)))


def mat_mul_pure_python(x, y):
    result = [[0] * len(x)] * len(y[0])
    # iterate through rows of X
    for i in range(len(x)):
        # iterate through columns of Y
        for j in range(len(y[0])):
            # iterate through rows of Y
            for k in range(len(y)):
                result[i][j] += x[i][k] * y[k][j]


# Calls the above functions with interesting inputs.
def main():
    # Numpy
    seed_zero()

    x = np.array([[0, 1, 2, 3],
                 [10, 11, 12, 13]])
    y = np.array([20, 21, 22, 23])
    z = np.arange(10)

    # Implement function a-?
    test_array(one_hot_5_of_10(), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    test_array(negate_3_to_8(z), [0, 1, 2, 3, -4, -5, -6, -7, 8, 9])
    test(get_size_properties(x), (2, 4, 8, 2))
    test_array(append_vector_to_matrix(x, y), np.array([[0, 1, 2, 3],
                                                        [10, 11, 12, 13],
                                                        [20, 21, 22, 23]]))
    test_array(column_sum(x), np.array([10, 12, 14, 16]))

    # Fill in the expected value of the functions Q1-Q2.
    # Yes, we're aware you can print the value and copy-paste it.
    # Please try to think about it before you do so.
    a1 = np.array([0, 1, 10, 3])
    test_array(a1, q1())
    a2 = [[0, 01, 02, 03],
         [10, 11, 12, 13],
         [20, 21, 22, 23]]
    test_array(a2, q2())

    multiplication_table()

    # SciPy
    view_face()

    sample = norm.rvs(size=100)
    x = sp.r_[-5:5:100j]
    plot_samples(sample, x)

    # Compare the execution speed of matrix multiplication using pure python and using SciPy.
    # Use the mat_mul_pure_python function and the timeit module.
    setup1 = \
"""from __main__ import mat_mul_pure_python
from scipy.stats import norm
import numpy as np
X = np.random.random((100, 100))
Y = np.random.random((100, 100))
x_list = X.tolist()
y_list = Y.tolist()"""
    print(timeit.timeit("mat_mul_pure_python(x_list, y_list)", setup=setup1, number=10))

    setup2 = \
"""from scipy.stats import norm
import numpy as np
X = np.random.random((100, 100))
Y = np.random.random((100, 100))
"""
    print(timeit.timeit("np.matmul(X, Y)", setup=setup1, number=10))


if __name__ == '__main__':
    main()
