from copy import copy
import numpy as np
from numpy.random import default_rng
rng = default_rng()
import torch
import time
from utils_d3roma.utils import compute_scale_and_shift

def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mean_square_error(y_true, y_pred):
    return torch.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]

def mean_absolute_error(y_true, y_pred):
    # return np.abs(y_true - y_pred).mean()
    return torch.abs(y_true - y_pred).mean(1)

def mean_accuracy_inverse(y_true, y_pred):
    thresh = torch.maximum(y_true / y_pred, y_pred / y_true)
    return  1 / torch.mean((thresh < 1.25).float())


class ScaleShiftEstimator:
    def __init__(self):
        self.params = (1, 0) # s,t

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """ X & Y: Nx1 """
        start = time.time()
        self.params = compute_scale_and_shift(X, Y)
        end = time.time()
        print(f"ssi: {end - start:.5f}")
        return self

    def predict(self, X: np.ndarray):
        return X * self.params[0] + self.params[1]
    
class RANSAC:
    def __init__(self, n=0.1, k=100, t=0.05, d=0.5, model=ScaleShiftEstimator(), loss=square_error_loss, metric=mean_accuracy_inverse):
        self.n = n              # `n`: (percent) Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: (percent)Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = None

    def fit(self, X, Y, mask):
        """ X: source
            Y: target
        """
        assert X.shape == Y.shape == mask.shape
        B, HW = X.shape

        X = X.clone()
        Y = Y.clone()
        mask = mask.clone()
        N = int(self.n * HW)
        T = self.t
        # T = self.t * torch.abs(Y[mask.bool()]).mean()
        D = int(self.d * HW)

        assert D < HW and N < HW, "N, D must be less than HW"

        self.best_num_inlier = torch.zeros((B, 1), device=X.device).to(torch.int32)
        self.best_mask_inlier = torch.zeros((B, HW), device=X.device).to(torch.bool)
        self.best_error = torch.full((B, 1), torch.inf, device=X.device)
        self.best_fit = torch.empty((B, 2), device=X.device) 
        self.best_fit[:,0] = 1.0 # init s=1, t=0
        self.best_fit[:,1] = 0.0 

        for _ in range(self.k):
            ids = torch.randperm(HW, device=X.device).repeat(B, 1) # torch.arange(HW, device=X.device).repeat(B, 1) #
            maybe_inliers = ids[:, :N]
            maybe_model = compute_scale_and_shift(
                                            torch.gather(X, 1, maybe_inliers), 
                                            torch.gather(Y, 1, maybe_inliers),
                                            torch.gather(mask, 1, maybe_inliers))

            X_ = X * maybe_model[:, 0:1] + maybe_model[:,1:]
            threshold = torch.where(self.loss(Y, X_,) < T, 1, 0).to(torch.bool) & mask.bool()

            better_model = compute_scale_and_shift(X, Y, threshold)
            X__ = X * better_model[:, 0:1] + better_model[:, 1:]
            this_error = self.metric(Y, X__)[...,None]
            this_num_inlier = torch.sum(threshold, 1)[...,None]
            select = (this_num_inlier > D) & (this_error < self.best_error)

            self.best_num_inlier = torch.where(select, this_num_inlier, self.best_num_inlier)
            self.best_mask_inlier = torch.where(select, threshold, self.best_mask_inlier)
            self.best_fit = torch.where(select, better_model, self.best_fit)
            self.best_error = torch.where(select, this_error, self.best_error)
        return self

    def predict(self, X):
        return self.best_fit.predict(X)

class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params


if __name__ == "__main__":

    regressor = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)

    X = np.array([-0.848,-0.800,-0.704,-0.632,-0.488,-0.472,-0.368,-0.336,-0.280,-0.200,-0.00800,-0.0840,0.0240,0.100,0.124,0.148,0.232,0.236,0.324,0.356,0.368,0.440,0.512,0.548,0.660,0.640,0.712,0.752,0.776,0.880,0.920,0.944,-0.108,-0.168,-0.720,-0.784,-0.224,-0.604,-0.740,-0.0440,0.388,-0.0200,0.752,0.416,-0.0800,-0.348,0.988,0.776,0.680,0.880,-0.816,-0.424,-0.932,0.272,-0.556,-0.568,-0.600,-0.716,-0.796,-0.880,-0.972,-0.916,0.816,0.892,0.956,0.980,0.988,0.992,0.00400]).reshape(-1,1)
    y = np.array([-0.917,-0.833,-0.801,-0.665,-0.605,-0.545,-0.509,-0.433,-0.397,-0.281,-0.205,-0.169,-0.0531,-0.0651,0.0349,0.0829,0.0589,0.175,0.179,0.191,0.259,0.287,0.359,0.395,0.483,0.539,0.543,0.603,0.667,0.679,0.751,0.803,-0.265,-0.341,0.111,-0.113,0.547,0.791,0.551,0.347,0.975,0.943,-0.249,-0.769,-0.625,-0.861,-0.749,-0.945,-0.493,0.163,-0.469,0.0669,0.891,0.623,-0.609,-0.677,-0.721,-0.745,-0.885,-0.897,-0.969,-0.949,0.707,0.783,0.859,0.979,0.811,0.891,-0.137]).reshape(-1,1)

    regressor.fit(X, y)

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1)
    ax.set_box_aspect(1)

    plt.scatter(X, y)

    line = np.linspace(-1, 1, num=100).reshape(-1, 1)
    plt.plot(line, regressor.predict(line), c="peru")
    # plt.show()
    plt.savefig("ransac.png")
    plt.close()