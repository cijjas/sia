import numpy as np

class Optimizer:
    def __init__(self, method="gradient_descent", mini_batch_size=16, eta=0.01,
                 alpha=0.9, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.method = method
        self.eta = eta  # Learning rate
        self.mini_batch_size = mini_batch_size
        self.alpha = alpha
        self.beta_1 = beta_1  # Exponential decay rate for the first moment estimates
        self.beta_2 = beta_2  # Exponential decay rate for the second moment estimates
        self.epsilon = epsilon
        self.v_w = None  # For momentum and Adam
        self.m_w = None  # For Adam
        self.t = 0       # Time step for Adam

    def get_method(self):
        return self.method

    def update_parameters(self, weights, grads_w):
        if self.method == "gradient_descent":
            return self.gradient_descent(weights, grads_w)
        elif self.method == "momentum":
            return self.momentum(weights, grads_w)
        elif self.method == "adam":
            return self.adam(weights, grads_w)
        else:
            raise ValueError("Unknown optimization method")

    def gradient_descent(self, weights, grads_w):
        lr = self.eta / self.mini_batch_size
        weights = [w - lr * gw for w, gw in zip(weights, grads_w)]
        return weights

    def momentum(self, weights, grads_w):
        if self.v_w is None:
            # Initialize velocities
            self.v_w = [np.zeros_like(w) for w in weights]

        lr = self.eta / self.mini_batch_size
        momentum = self.alpha

        # Update velocities and parameters
        self.v_w = [momentum * vw - lr * gw for vw, gw in zip(self.v_w, grads_w)]
        weights = [w + vw for w, vw in zip(weights, self.v_w)]
        return weights

    def adam(self, weights, grads_w):
        if self.m_w is None:
            # Initialize first and second moment vectors
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]

        self.t += 1  # Update time step

        lr_t = self.eta / self.mini_batch_size
        beta_1, beta_2, epsilon = self.beta_1, self.beta_2, self.epsilon
        beta_1_t = 1 - beta_1 ** self.t
        beta_2_t = 1 - beta_2 ** self.t

        # Update biased first and second moment estimates
        self.m_w = [beta_1 * mw + (1 - beta_1) * gw for mw, gw in zip(self.m_w, grads_w)]
        self.v_w = [beta_2 * vw + (1 - beta_2) * (gw ** 2) for vw, gw in zip(self.v_w, grads_w)]

        # Compute bias-corrected estimates
        m_w_hat = [mw / beta_1_t for mw in self.m_w]
        v_w_hat = [vw / beta_2_t for vw in self.v_w]

        # Update weights
        weights = [w - lr_t * mwh / (np.sqrt(vwh) + epsilon)
                   for w, mwh, vwh in zip(weights, m_w_hat, v_w_hat)]
        return weights
