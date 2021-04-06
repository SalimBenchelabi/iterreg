import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


from celer import Lasso
from celer.datasets import make_correlated_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score

from sklearn.linear_model import LassoLars

from iterreg.sparse import dual_primal
from iterreg.utils import shrink
from celer.plot_utils import configure_plt

configure_plt()


n_samples = 1_500
n_features = 3_000

X_, y_, w_true = make_correlated_data(
    n_samples=2 * n_samples, n_features=n_features, snr=5, density=0.01,
    random_state=0)

X, X_test, y, y_test = train_test_split(X_, y_, test_size=0.5)

clf_bp = LassoLars(alpha=1e-12, max_iter=10_000, fit_intercept=0).fit(X, y)
bp = clf_bp.coef_


n_iter_ista = 100
max_iter = 10_000

# ISTA part:
n_features = X.shape[1]
L = norm(X, ord=2) ** 2

w = np.zeros(n_features)
target = np.zeros_like(y)
all_w_breg = np.zeros([max_iter, n_features])
for outer in range(max_iter // n_iter_ista):
    print(outer * n_iter_ista)
    target += y - X @ w

    for t in range(n_iter_ista):
        R = target - X @ w
        w[:] = shrink(w + 1. / L * X.T @ R, 1 / L)
        all_w_breg[outer * n_iter_ista + t] = w


# sigma_heuri = 1. / np.sort(np.abs(X.T @ y))[int(0.99 * X.shape[1])] / 2
# step_heuri = 1 / (sigma_heuri * norm(X, ord=2))


print("PD starts")
w, theta, _, all_w = dual_primal(
    X, y, step=1, f_store=1, max_iter=max_iter)
print("PD ends")


mses_breg = [mse(y_test, X_test @ w) for w in all_w_breg]
f1_breg = [f1_score(w_true != 0, w != 0) for w in all_w_breg]
mses_cp = [mse(y_test, X_test @ w) for w in all_w]
f1_cp = [f1_score(w_true != 0, w != 0) for w in all_w]


fig, axarr = plt.subplots(3, 2, sharex='col', sharey='row')
axarr[0, 0].semilogy(norm(all_w_breg - bp, axis=1))
axarr[0, 0].set_ylabel(r'$\Vert w^k - \bar{w}_\delta \Vert$')
axarr[0, 0].set_title("Bregman residuals")
axarr[1, 0].plot(mses_breg)
axarr[1, 0].set_ylabel("pred MSE left out")
axarr[2, 0].plot(f1_breg)
axarr[2, 0].set_ylabel("F1 true support")

axarr[0, 1].set_title("Iterative regularization")
axarr[0, 1].semilogy(norm(all_w - bp, axis=1))
axarr[1, 1].plot(mses_cp)
axarr[2, 1].plot(f1_cp)
axarr[2, 1].set_xlabel("CP iterations")
axarr[2, 0].set_xlabel("FB iterations")

plt.show(block=False)
