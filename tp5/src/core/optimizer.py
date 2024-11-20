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

        # Validate optimization method
        valid_methods = ["gradient_descent", "momentum", "adam"]
        if self.method not in valid_methods:
            raise ValueError(f"Unknown optimization method: {self.method}")

    def update_parameters(self, weights, biases, grads_w, grads_b, mini_batch_size):
        # Adjust the grads_w with weight decay
        if self.weight_decay != 0:
            grads_w = [gw + self.weight_decay * w for gw, w in zip(grads_w, weights)]

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
        new_weights = [w - lr * gw for w, gw in zip(weights, grads_w)]
        new_biases = [b - lr * gb for b, gb in zip(biases, grads_b)]
        return new_weights, new_biases

    def _momentum(self, weights, biases, grads_w, grads_b, mini_batch_size):
        if self.v_w is None:
            # Initialize velocities
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]

        lr = self.eta / mini_batch_size
        alpha = self.alpha

        # Update velocities
        self.v_w = [alpha * vw - lr * gw for vw, gw in zip(self.v_w, grads_w)]
        self.v_b = [alpha * vb - lr * gb for vb, gb in zip(self.v_b, grads_b)]

        # Update parameters
        new_weights = [w + vw for w, vw in zip(weights, self.v_w)]
        new_biases = [b + vb for b, vb in zip(biases, self.v_b)]
        return new_weights, new_biases

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

        # Update moments for weights
        self.m_w = [
            beta_1 * mw + (1 - beta_1) * gw for mw, gw in zip(self.m_w, grads_w)
        ]
        self.v_w = [
            beta_2 * vw + (1 - beta_2) * (gw ** 2) for vw, gw in zip(self.v_w, grads_w)
        ]
        m_w_hat = [mw / (1 - beta_1 ** self.t) for mw in self.m_w]
        v_w_hat = [vw / (1 - beta_2 ** self.t) for vw in self.v_w]

        # Update moments for biases
        self.m_b = [
            beta_1 * mb + (1 - beta_1) * gb for mb, gb in zip(self.m_b, grads_b)
        ]
        self.v_b = [
            beta_2 * vb + (1 - beta_2) * (gb ** 2) for vb, gb in zip(self.v_b, grads_b)
        ]
        m_b_hat = [mb / (1 - beta_1 ** self.t) for mb in self.m_b]
        v_b_hat = [vb / (1 - beta_2 ** self.t) for vb in self.v_b]

        # Update parameters
        new_weights = [
            w - lr_t * mwh / (np.sqrt(vwh) + epsilon)
            for w, mwh, vwh in zip(weights, m_w_hat, v_w_hat)
        ]
        new_biases = [
            b - lr_t * mbh / (np.sqrt(vbh) + epsilon)
            for b, mbh, vbh in zip(biases, m_b_hat, v_b_hat)
        ]

        # Apply weight decay (AdamW)
        if self.weight_decay != 0:
            new_weights = [
                nw - lr_t * self.weight_decay * w for nw, w in zip(new_weights, weights)
            ]

        return new_weights, new_biases
