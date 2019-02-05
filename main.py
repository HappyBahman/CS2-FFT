from sklearn import datasets
from sklearn import preprocessing
import numpy as np
from PIL import Image
import cvxpy as cvx
from feature import extract_cs2fft, amp_filter
import matplotlib.pyplot as plt
from comp_sensing import generate_matrix, generate_gaussian_matrix
# dataset = datasets.fetch_lfw_people(data_home='./lfw_data')
import copy

max_th = None
max_t = -1 * np.inf
threshold = 0
num_of_windows = 190
As = None
num_of_classes = 40
feature_size = 120
# bernoulli_mtrx = np.array(generate_matrix([10, num_of_windows]))
bernoulli_mtrx = np.array(generate_gaussian_matrix([feature_size, num_of_windows]))

As = None
for cls in range(1, num_of_classes + 1):
    new_A = np.zeros((5, feature_size))
    for d in range(0, 10, 2):
        training_sample = np.array(Image.open("./att_faces/orl_faces/s" + str(cls) + "/" + str(d + 1) + ".pgm")).flatten()
        fft_sum = extract_cs2fft(training_sample, num_of_windows)
        fft_sum = fft_sum[1:]
        amp_filter(fft_sum, threshold)
        # fft_sum_normalized = preprocessing.normalize(fft_sum.reshape(1, -1), norm='l2')
        fft_sum_normalized = fft_sum.reshape(1, -1)
        x_f = np.matmul(bernoulli_mtrx.transpose(), fft_sum_normalized.T)
        x_f = x_f[:, 0]
        new_A[int(d / 2), :] = x_f
    if As is None:
        As = new_A.T
    else:
        As = np.concatenate((As, new_A.T), 1)

t = f = 0

for tc in range(1, num_of_classes + 1):
    for i in range(0, 9, 2):
        test = np.array(Image.open("./att_faces/orl_faces/s" + str(tc) + "/" + str(i + 2) + ".pgm")).flatten()
        fft_sum = extract_cs2fft(test, num_of_windows)
        fft_sum = fft_sum[1:]
        amp_filter(fft_sum, threshold)
        fft_sum_normalized = preprocessing.normalize(fft_sum.reshape(1, -1), norm='l2')[0]
        y = np.matmul(bernoulli_mtrx.transpose(), fft_sum_normalized.T)
        idx = None
        min_err = np.inf

        theta = cvx.Variable(200)
        # D = phi * psi (psi is A and phi is bernoulli)
        D = As
        eps = 0.001
        constraints = [cvx.norm1(cvx.matmul(D.real, theta) - y.real) <= eps,
                       cvx.norm1(cvx.matmul(D.imag, theta) - y.imag) <= eps]
        objective = cvx.Minimize(cvx.norm(theta, 1))
        prob = cvx.Problem(objective, constraints)
        result = prob.solve()

        for cls in range(0, num_of_classes):
            if theta.value is not None:
                delta = np.zeros(200)
                delta[5 * cls: 5 * (cls + 1)] = theta.value[5 * cls: 5 * (cls + 1)]

                err = cvx.norm1(cvx.matmul(D.real, delta) - y.real) + \
                      cvx.norm1(cvx.matmul(D.imag, delta) - y.imag)
                err = err.value
            else:
                err = np.inf

            if abs(err) < min_err:
                min_err = abs(err)
                idx = cls
        if idx == tc - 1:
            t += 1
        else:
            f += 1

print('f ' + str(f))
print('t ' + str(t))
