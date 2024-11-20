import numpy as np


class Optimizer:
    def __init__(
        self,
        method: str = "gradient_descent",
        eta: float = 0.01,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        self.method = method.lower()
        self.eta = eta
        self.weight_decay = weight_decay

        # Parameters for Momentum
        self.alpha = kwargs.get("alpha", 0.9)

        # Parameters for Adam
        self.beta_1 = kwargs.get("beta_1", 0.9)
        self.beta_2 = kwargs.get("beta_2", 0.999)
        self.epsilon = kwargs.get("epsilon", 1e-8)

        # Initialize moments for Momentum and Adam
        self.v_w, self.v_b = None, None  # Velocities
        self.m_w, self.m_b = None, None  # Moments
        self.t = 0  # Time step for Adam

        # Store exponential decay rates to avoid recomputing
        self.beta_1_pow = 1.0
        self.beta_2_pow = 1.0

        # Validate optimization method
        valid_methods = ["gradient_descent", "momentum", "adam"]
        if self.method not in valid_methods:
            raise ValueError(f"Unknown optimization method: {self.method}")

    def update_parameters(self, weights, biases, grads_w, grads_b, mini_batch_size):
        # Adjust the grads_w with weight decay
        if self.weight_decay != 0 and self.method != "adam":
            for gw, w in zip(grads_w, weights):
                gw += self.weight_decay * w

        if self.method == "gradient_descent":
            return self._gradient_descent(
                weights, biases, grads_w, grads_b, mini_batch_size
            )
        elif self.method == "momentum":
            return self._momentum(weights, biases, grads_w, grads_b, mini_batch_size)
        elif self.method == "adam":
            return self._adam(weights, biases, grads_w, grads_b, mini_batch_size)

    def _gradient_descent(self, weights, biases, grads_w, grads_b, mini_batch_size):
        lr = self.eta / mini_batch_size
        for w, gw in zip(weights, grads_w):
            w -= lr * gw
        for b, gb in zip(biases, grads_b):
            b -= lr * gb
        return weights, biases

    def _momentum(self, weights, biases, grads_w, grads_b, mini_batch_size):
        if self.v_w is None:
            # Initialize velocities
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]

        lr = self.eta / mini_batch_size
        alpha = self.alpha

        # Update velocities and parameters
        for vw, w, gw in zip(self.v_w, weights, grads_w):
            vw *= alpha
            vw -= lr * gw
            w += vw

        for vb, b, gb in zip(self.v_b, biases, grads_b):
            vb *= alpha
            vb -= lr * gb
            b += vb

        return weights, biases

    def _adam(self, weights, biases, grads_w, grads_b, mini_batch_size):
        if self.m_w is None:
            # Initialize first and second moments
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1
        lr_t = self.eta / mini_batch_size
        beta_1, beta_2, epsilon = self.beta_1, self.beta_2, self.epsilon

        # Update exponential decay rates
        self.beta_1_pow *= beta_1
        self.beta_2_pow *= beta_2

        for i in range(len(weights)):
            # Update biased first moment estimate for weights
            self.m_w[i] *= beta_1
            self.m_w[i] += (1 - beta_1) * grads_w[i]

            # Update biased second raw moment estimate for weights
            self.v_w[i] *= beta_2
            np.add(self.v_w[i], (1 - beta_2) * np.square(grads_w[i]), out=self.v_w[i])

            # Compute bias-corrected first and second moment estimates for weights
            m_w_hat = self.m_w[i] / (1 - self.beta_1_pow)
            v_w_hat = self.v_w[i] / (1 - self.beta_2_pow)

            # Update weights
            weights[i] -= lr_t * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

            # Apply weight decay (AdamW)
            if self.weight_decay != 0:
                weights[i] -= lr_t * self.weight_decay * weights[i]

            # Update biased first moment estimate for biases
            self.m_b[i] *= beta_1
            self.m_b[i] += (1 - beta_1) * grads_b[i]

            # Update biased second raw moment estimate for biases
            self.v_b[i] *= beta_2
            np.add(self.v_b[i], (1 - beta_2) * np.square(grads_b[i]), out=self.v_b[i])

            # Compute bias-corrected first and second moment estimates for biases
            m_b_hat = self.m_b[i] / (1 - self.beta_1_pow)
            v_b_hat = self.v_b[i] / (1 - self.beta_2_pow)

            # Update biases
            biases[i] -= lr_t * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        return weights, biases
