import numpy as np

class Optimizer:
    def __init__(self, method="gradient_descent", mini_batch_size=16, eta=0.01,
                 alpha=0.9, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.method = method
        self.eta = eta # aka alpha= stepsize of momentum
        self.mini_batch_size = mini_batch_size
        self.alpha = alpha
        self.beta_1 = beta_1 # require exponential decay rates for the moment estimates [0,1)
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.v_w = None  # For momentum and Adam
        self.v_b = None
        self.m_w = None  # For Adam
        self.m_b = None
        self.t = 0       # Time step for Adam

    def get_method(self):
        return self.method

    def update_parameters(self, weights, biases, grads_w, grads_b):
        if self.method == "gradient_descent":
            return self.gradient_descent(weights, biases, grads_w, grads_b)
        elif self.method == "momentum":
            return self.momentum(weights, biases, grads_w, grads_b)
        elif self.method == "adam":
            return self.adam(weights, biases, grads_w, grads_b)
        else:
            raise ValueError("Unknown optimization method")

    def gradient_descent(self, weights, biases, grads_w, grads_b):
        lr = self.eta / self.mini_batch_size
        weights = [w - lr * gw for w, gw in zip(weights, grads_w)]
        biases = [b - lr * gb for b, gb in zip(biases, grads_b)]
        return weights, biases

    def momentum(self, weights, biases, grads_w, grads_b):
        if self.v_w is None:
            # Initialize velocities
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]

        lr = self.eta / self.mini_batch_size
        momentum = self.alpha

        # Update velocities and parameters
        self.v_w = [momentum * vw - lr * gw for vw, gw in zip(self.v_w, grads_w)]
        self.v_b = [momentum * vb - lr * gb for vb, gb in zip(self.v_b, grads_b)]
        weights = [w + vw for w, vw in zip(weights, self.v_w)]
        biases = [b + vb for b, vb in zip(biases, self.v_b)]
        return weights, biases

    def adam(self, weights, biases, grads_w, grads_b):
        if self.m_w is None:
            # v and m are moment vectors
            # Initialize first and second moment vectors
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1  # t <- t + 1

        lr_t = (self.eta / self.mini_batch_size)
        beta_1, beta_2, epsilon = self.beta_1, self.beta_2, self.epsilon
        beta_1_t = 1 - beta_1 ** self.t
        beta_2_t = 1 - beta_2 ** self.t

        # get gradients w.r.t. stochastic objective at timestep t
        # g_t = grads_w, grads_b

        # Update biased first and second moment vectors for weights
        # m_t = β1 * m_{t-1} + (1 - β1) * g_t
        self.m_w = [beta_1 * mw + (1 - beta_1) * gw for mw, gw in zip(self.m_w, grads_w)]
        # v_t = β2 * v_{t-1} + (1 - β2) * (g_t)^2
        self.v_w = [beta_2 * vw + (1 - beta_2) * (gw ** 2) for vw, gw in zip(self.v_w, grads_w)]
        # Compute bias-corrected estimates
        # m_hat = m_t / (1 - β1^t)
        m_w_hat = [mw / beta_1_t for mw in self.m_w] # promedio exponencial de los gradientes
        # v_hat = v_t / (1 - β2^t)
        v_w_hat = [vw / beta_2_t for vw in self.v_w] # promedio cuadratico de los gradientes
        # Update weights
        weights = [w - lr_t * mwh / (np.sqrt(vwh) + epsilon)
                   for w, mwh, vwh in zip(weights, m_w_hat, v_w_hat)]

        # Same for biases
        self.m_b = [beta_1 * mb + (1 - beta_1) * gb for mb, gb in zip(self.m_b, grads_b)]
        self.v_b = [beta_2 * vb + (1 - beta_2) * (gb ** 2) for vb, gb in zip(self.v_b, grads_b)]
        # Compute bias-corrected estimates
        m_b_hat = [mb / beta_1_t for mb in self.m_b]
        v_b_hat = [vb / beta_2_t for vb in self.v_b]
        # Update biases
        biases = [b - lr_t * mbh / (np.sqrt(vbh) + epsilon)
                  for b, mbh, vbh in zip(biases, m_b_hat, v_b_hat)]

        return weights, biases
