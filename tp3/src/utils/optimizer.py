import numpy as np

class Optimizer:
    def __init__(self, method="gradient_descent", eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None, t=0):
        self.method = method
        self.eta = eta 
        self.beta1 = beta1  
        self.beta2 = beta2  
        self.epsilon = epsilon  
        self.m = m
        self.v = v  
        self.t = t  


    def update(self, weights, biases, grads_w, grads_b, mini_batch_size):
        if self.method == "gradient_descent":
            return self.gradient_descent(weights, biases, grads_w, grads_b, mini_batch_size)
        elif self.method == "momentum":
            return self.momentum(weights, biases, grads_w, grads_b, mini_batch_size)
        elif self.method == "adam":
            return self.adam(weights, biases, grads_w, grads_b, mini_batch_size)
        else:
            raise ValueError("Unknown optimization method")
   

    def gradient_descent(self, weights, biases, grads_w, grads_b, mini_batch_size):
        weights = [w - (self.eta / mini_batch_size) * gw for w, gw in zip(weights, grads_w)]
        biases = [b - (self.eta / mini_batch_size) * gb for b, gb in zip(biases, grads_b)]
        return weights, biases

    def momentum(self, weights, biases, grads_w, grads_b, mini_batch_size):
        if self.m is None:
            self.m = {"w": [np.zeros_like(w) for w in weights], "b": [np.zeros_like(b) for b in biases]}

        self.m["w"] = [self.beta1 * mw + (1 - self.beta1) * gw for mw, gw in zip(self.m["w"], grads_w)]
        self.m["b"] = [self.beta1 * mb + (1 - self.beta1) * gb for mb, gb in zip(self.m["b"], grads_b)]

        weights = [w - (self.eta / mini_batch_size) * mw for w, mw in zip(weights, self.m["w"])]
        biases = [b - (self.eta / mini_batch_size) * mb for b, mb in zip(biases, self.m["b"])]
        return weights, biases

    def adam(self, weights, biases, grads_w, grads_b, mini_batch_size):
        if self.m is None or self.v is None:
            self.m = {"w": [np.zeros_like(w) for w in weights], "b": [np.zeros_like(b) for b in biases]}
            self.v = {"w": [np.zeros_like(w) for w in weights], "b": [np.zeros_like(b) for b in biases]}

        self.t += 1  # Increase timestep
        self.m["w"] = [self.beta1 * mw + (1 - self.beta1) * gw for mw, gw in zip(self.m["w"], grads_w)]
        self.m["b"] = [self.beta1 * mb + (1 - self.beta1) * gb for mb, gb in zip(self.m["b"], grads_b)]

        self.v["w"] = [self.beta2 * vw + (1 - self.beta2) * (gw ** 2) for vw, gw in zip(self.v["w"], grads_w)]
        self.v["b"] = [self.beta2 * vb + (1 - self.beta2) * (gb ** 2) for vb, gb in zip(self.v["b"], grads_b)]

        # Bias correction
        m_hat_w = [mw / (1 - self.beta1 ** self.t) for mw in self.m["w"]]
        m_hat_b = [mb / (1 - self.beta1 ** self.t) for mb in self.m["b"]]

        v_hat_w = [vw / (1 - self.beta2 ** self.t) for vw in self.v["w"]]
        v_hat_b = [vb / (1 - self.beta2 ** self.t) for vb in self.v["b"]]

        weights = [w - (self.eta / mini_batch_size) * mw / (np.sqrt(vw) + self.epsilon)
                   for w, mw, vw in zip(weights, m_hat_w, v_hat_w)]
        biases = [b - (self.eta / mini_batch_size) * mb / (np.sqrt(vb) + self.epsilon)
                  for b, mb, vb in zip(biases, m_hat_b, v_hat_b)]

        return weights, biases
