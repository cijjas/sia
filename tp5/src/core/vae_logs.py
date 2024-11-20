import numpy as np
import matplotlib.pyplot as plt

from .optimizer import Optimizer
from .activation_function import ActivationFunction
from .initialization import WeightInitializer

from numba import njit


class VAE(object):
    def __init__(
        self,
        seed,
        encoder_topology,
        decoder_topology,
        activation_function: ActivationFunction,
        encoder_optimizer: Optimizer,
        decoder_optimizer: Optimizer,
        weights_encoder=None,
        biases_encoder=None,
        weights_decoder=None,
        biases_decoder=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.activation_function = activation_function
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

        if weights_encoder is not None and biases_encoder is not None:
            self.weights_encoder = weights_encoder
            self.biases_encoder = biases_encoder
        else:
            weight_initializer = WeightInitializer(seed)
            self.weights_encoder = weight_initializer.initialize_weights(
                encoder_topology, activation_function.method
            )
            self.biases_encoder = weight_initializer.initialize_biases(encoder_topology)

        if weights_decoder is not None and biases_decoder is not None:
            self.weights_decoder = weights_decoder
            self.biases_decoder = biases_decoder
        else:
            weight_initializer = WeightInitializer(seed)
            self.weights_decoder = weight_initializer.initialize_weights(
                decoder_topology, activation_function.method
            )
            self.biases_decoder = weight_initializer.initialize_biases(decoder_topology)

        self.encoder_topology = encoder_topology
        self.decoder_topology = decoder_topology
        self.latent_dim = decoder_topology[0]

    def encode(self, x):
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases_encoder[:-1], self.weights_encoder[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function.activation(z)
            activations.append(activation)

        z = np.dot(self.weights_encoder[-1], activation) + self.biases_encoder[-1]
        zs.append(z)
        mu = z[: self.latent_dim, :]
        log_var = z[self.latent_dim :, :]
        activations.append(z)

        return mu, log_var, activations, zs

    def decode(self, z):
        activation = z
        activations = [z]
        zs = []

        for b, w in zip(self.biases_decoder[:-1], self.weights_decoder[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function.activation(z)
            activations.append(activation)

        z = np.dot(self.weights_decoder[-1], activation) + self.biases_decoder[-1]
        zs.append(z)
        reconstruction = 1 / (1 + np.exp(-z))
        activations.append(reconstruction)

        return reconstruction, activations, zs

    def feedforward(self, x):
        mu, log_var, encoder_activations, encoder_zs = self.encode(x)
        std = np.exp(0.5 * log_var)
        epsilon = np.random.normal(size=std.shape)
        z = mu + std * epsilon

        recon_x, decoder_activations, decoder_zs = self.decode(z)

        return (
            recon_x,
            mu,
            log_var,
            z,
            epsilon,
            encoder_activations,
            encoder_zs,
            decoder_activations,
            decoder_zs,
        )

    def compute_loss(self, x, recon_x, mu, log_var):
        epsilon = 1e-8
        recon_loss = -np.sum(
            x * np.log(recon_x + epsilon) + (1 - x) * np.log(1 - recon_x + epsilon)
        )
        kl_div = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        loss = recon_loss + kl_div
        return loss, recon_loss, kl_div

    @staticmethod
    @njit
    def backpropagation_numba(
        recon_x,
        x,
        weights_decoder,
        decoder_activations,
        decoder_zs,
        weights_encoder,
        encoder_activations,
        encoder_zs,
        mu,
        log_var,
        epsilon,
        latent_dim,
        activation_prime_func,
    ):
        nabla_b_encoder = [np.zeros_like(b) for b in weights_encoder]
        nabla_w_encoder = [np.zeros_like(w) for w in weights_encoder]

        nabla_b_decoder = [np.zeros_like(b) for b in weights_decoder]
        nabla_w_decoder = [np.zeros_like(w) for w in weights_decoder]

        delta = recon_x - x

        num_layers_decoder = len(weights_decoder)

        for l in range(num_layers_decoder - 1, -1, -1):
            if l == num_layers_decoder - 1:
                activation_prime = recon_x * (1 - recon_x)
                delta *= activation_prime
            else:
                a = decoder_activations[l + 1]
                activation_prime = activation_prime_func(a)
                delta = np.dot(weights_decoder[l + 1].T, delta) * activation_prime

            nabla_b_decoder[l] += np.sum(delta, axis=1, keepdims=True)
            nabla_w_decoder[l] += np.dot(delta, decoder_activations[l].T)

        delta_z = np.dot(weights_decoder[0].T, delta)

        std = np.exp(0.5 * log_var)
        delta_mu = delta_z + mu
        delta_log_var = delta_z * epsilon * 0.5 * std + 0.5 * (np.exp(log_var) - 1)

        delta_z_full = np.vstack([delta_mu, delta_log_var])

        num_layers_encoder = len(weights_encoder)
        delta = delta_z_full

        for l in range(num_layers_encoder - 1, -1, -1):
            if l == num_layers_encoder - 1:
                nabla_b_encoder[l] += np.sum(delta, axis=1, keepdims=True)
                nabla_w_encoder[l] += np.dot(delta, encoder_activations[l].T)
            else:
                a = encoder_activations[l + 1]
                activation_prime = activation_prime_func(a)
                delta = np.dot(weights_encoder[l + 1].T, delta) * activation_prime
                nabla_b_encoder[l] += np.sum(delta, axis=1, keepdims=True)
                nabla_w_encoder[l] += np.dot(delta, encoder_activations[l].T)

        return nabla_b_encoder, nabla_w_encoder, nabla_b_decoder, nabla_w_decoder

    def backpropagation(
        self,
        x,
        recon_x,
        mu,
        log_var,
        z,
        epsilon,
        encoder_activations,
        encoder_zs,
        decoder_activations,
        decoder_zs,
    ):
        activation_prime_func = self.activation_function.activation_prime

        (
            nabla_b_encoder,
            nabla_w_encoder,
            nabla_b_decoder,
            nabla_w_decoder,
        ) = self.backpropagation_numba(
            recon_x,
            x,
            self.weights_decoder,
            decoder_activations,
            decoder_zs,
            self.weights_encoder,
            encoder_activations,
            encoder_zs,
            mu,
            log_var,
            epsilon,
            self.latent_dim,
            activation_prime_func,
        )

        return nabla_b_encoder, nabla_w_encoder, nabla_b_decoder, nabla_w_decoder

    def update_weights_and_biases(self, mini_batch):
        x_batch = np.hstack([x for x, _ in mini_batch])

        (
            recon_x,
            mu,
            log_var,
            z,
            epsilon,
            encoder_activations,
            encoder_zs,
            decoder_activations,
            decoder_zs,
        ) = self.feedforward(x_batch)
        loss, recon_loss, kl_div = self.compute_loss(x_batch, recon_x, mu, log_var)

        (
            delta_nabla_b_encoder,
            delta_nabla_w_encoder,
            delta_nabla_b_decoder,
            delta_nabla_w_decoder,
        ) = self.backpropagation(
            x_batch,
            recon_x,
            mu,
            log_var,
            z,
            epsilon,
            encoder_activations,
            encoder_zs,
            decoder_activations,
            decoder_zs,
        )

        self.weights_encoder, self.biases_encoder = (
            self.encoder_optimizer.update_parameters(
                weights=self.weights_encoder,
                biases=self.biases_encoder,
                grads_w=delta_nabla_w_encoder,
                grads_b=delta_nabla_b_encoder,
                mini_batch_size=len(mini_batch),
            )
        )

        self.weights_decoder, self.biases_decoder = (
            self.decoder_optimizer.update_parameters(
                weights=self.weights_decoder,
                biases=self.biases_decoder,
                grads_w=delta_nabla_w_decoder,
                grads_b=delta_nabla_b_decoder,
                mini_batch_size=len(mini_batch),
            )
        )

        return loss / len(mini_batch)

    def fit(self, training_data, epochs, mini_batch_size):
        n = len(training_data)
        avg_loss_history = []
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            epoch_loss = 0
            for mini_batch in mini_batches:
                avg_loss = self.update_weights_and_biases(mini_batch)
                epoch_loss += avg_loss * len(mini_batch)
            epoch_avg_loss = epoch_loss / n
            avg_loss_history.append(epoch_avg_loss)
            print(
                f"Epoch {epoch+1}/{epochs} complete with average loss: {epoch_avg_loss}"
            )
