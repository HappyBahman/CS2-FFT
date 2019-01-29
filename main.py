from sklearn import datasets
from sklearn import preprocessing
import numpy as np
from PIL import Image
import cvxpy as cvx
from feature import extract_cs2fft, amp_filter
import matplotlib.pyplot as plt
from comp_sensing import generate_matrix
# dataset = datasets.fetch_lfw_people(data_home='./lfw_data')

threshold = 1
num_of_windows = 100
As = []
num_of_classes = 5
bernoulli_mtrx = np.array(generate_matrix([10, num_of_windows]))

for cls in range(1, num_of_classes + 1):
    new_A = np.zeros((5, 10))
    for d in range(0, 10, 2):
        training_sample = np.array(Image.open("./att_faces/orl_faces/s" + str(cls) + "/" + str(d + 1) + ".pgm")).flatten()
        fft_sum = extract_cs2fft(training_sample, num_of_windows)
        # todo: check, are we throwing away important data?
        fft_sum = fft_sum[1:]
        amp_filter(fft_sum, threshold)
        x_f = np.matmul(bernoulli_mtrx.transpose(), fft_sum)
        x_f_normalized = preprocessing.normalize(x_f.reshape(1, -1), norm='l2')
        new_A[int(d/2), :] = x_f_normalized
    As.append(new_A)



t = f = 0

for tc in range(1, num_of_classes + 1):
    for i in range(0, 9, 2):
        test = np.array(Image.open("./att_faces/orl_faces/s" + str(tc) + "/" + str(i + 2) + ".pgm")).flatten()
        fft_sum = extract_cs2fft(test, num_of_windows)
        fft_sum = fft_sum[1:]
        amp_filter(fft_sum, threshold)
        x_f = np.matmul(bernoulli_mtrx.transpose(), fft_sum)
        y = preprocessing.normalize(x_f.reshape(1, -1), norm='l2')[0]
        idx = None
        min_err = np.inf

        for cls in range(0, num_of_classes):
            theta = cvx.Variable(5)
            # D = phi * psi (psi is A and phi is bernoulli)
            D = As[cls]
            # eps = 0.000001
            constraints = [cvx.matmul(D.real.transpose(), theta) == y.real,
                           cvx.matmul(D.imag.transpose(), theta) == y.imag]
            # constraints = [cvx.norm1(cvx.matmul(D.real.transpose(), theta) - y.real) +
            #                cvx.norm1(cvx.matmul(D.imag.transpose(), theta) - y.imag) <= eps]
            objective = cvx.Minimize(cvx.norm(theta, 1))
            prob = cvx.Problem(objective, constraints)
            result = prob.solve()
            print(prob.status)
            if theta.value is not None:
                err = cvx.norm1(cvx.matmul(D.real.transpose(), theta.value) - y.real) +\
                      cvx.norm1(cvx.matmul(D.imag.transpose(), theta.value) - y.imag)
                err = err.value
            else:
                err = np.inf
            print(err)
            if abs(err) < min_err:
                min_err = abs(err)
                idx = cls
        if idx == tc - 1:
            t += 1
            print("correct")
        else:
            f += 1

print('f ' + str(f))
print('t ' + str(t))
