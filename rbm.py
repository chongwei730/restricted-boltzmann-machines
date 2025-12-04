from __future__ import print_function
import numpy as np
import os
from datetime import datetime
from sample import sample_hidden, sample_visible, gibbs_step, gibbs_chain, _logistic, gibbs_mult_chain_birthdeath, gibbs_mult_chain, gibbs_mult_chain_birthdeath_torch

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True

        # Initialize weights using Xavier/Glorot initialization
        np_rng = np.random.RandomState(1234)
        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def train(self, data, max_epochs=1000, learning_rate=0.1, batch_size=None, cd_k=1, reg_lambda=1e-4):
        """
        Train the machine.
        
        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.    
        """
        num_examples = data.shape[0]

        for epoch in range(max_epochs):
            if batch_size is None:
                # Full-batch training (original behavior)
                # Positive phase - Add bias unit here
                data_with_bias = np.insert(data, 0, 1, axis=1)
                pos_hidden_states, pos_hidden_probs = sample_hidden(data_with_bias, self.weights, add_bias=False)
                pos_associations = np.dot(data_with_bias.T, pos_hidden_probs)

                # Negative phase: run CD-k starting from the data
                neg_vis = data_with_bias[:, 1:]
                neg_visible_states = None
                neg_visible_probs = None
                neg_hidden_probs = None
                for _ in range(cd_k):
                    neg_visible_states, neg_visible_probs, _, neg_hidden_probs = gibbs_step(neg_vis, self.weights)
                    # Prepare next iteration using the sampled visible states (without bias column)
                    neg_vis = neg_visible_states[:, 1:]

                # Calculate negative associations (use probabilities from last step)
                neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

                # Update weights (normalize by number of examples)
                grad = (pos_associations - neg_associations) / num_examples

                # === L2 regularization (skip bias row/col) ===
                W_no_bias = self.weights.copy()
                W_no_bias[0, :] = 0
                W_no_bias[:, 0] = 0
                grad -= reg_lambda * W_no_bias     # reg_lambda is e.g. 1e-4

                # === Update ===
                self.weights += learning_rate * grad

                # Calculate reconstruction error (mean squared error per pixel)
                mse = np.mean((data - neg_visible_probs[:, 1:]) ** 2)
                if self.debug_print:
                    print("Epoch %s: mse (per-pixel) = %g" % (epoch, mse))
            else:
                # Mini-batch training
                perm = np.random.permutation(num_examples)
                mse_acc = 0.0
                n_batches = 0
                for start in range(0, num_examples, batch_size):
                    end = min(start + batch_size, num_examples)
                    batch_idx = perm[start:end]
                    batch = data[batch_idx]

                    # Positive phase for the batch
                    batch_with_bias = np.insert(batch, 0, 1, axis=1)
                    _, pos_hidden_probs = sample_hidden(batch_with_bias, self.weights, add_bias=False)
                    pos_associations = np.dot(batch_with_bias.T, pos_hidden_probs)

                    # Negative phase (CD-k) on the batch
                    neg_vis = batch_with_bias[:, 1:]
                    neg_visible_states = None
                    neg_visible_probs = None
                    neg_hidden_probs = None
                    for _ in range(cd_k):
                        neg_visible_states, neg_visible_probs, _, neg_hidden_probs = gibbs_step(neg_vis, self.weights)
                        neg_vis = neg_visible_states[:, 1:]

                    neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

                    # Update weights normalized by batch size
                    bsize = batch.shape[0]
                    grad = (pos_associations - neg_associations) / bsize

                    # L2 regularization (no bias)
                    W_no_bias = self.weights.copy()
                    W_no_bias[0, :] = 0
                    W_no_bias[:, 0] = 0
                    grad -= reg_lambda * W_no_bias

                    self.weights += learning_rate * grad

                    # Accumulate MSE for reporting
                    mse_acc += np.mean((batch - neg_visible_probs[:, 1:]) ** 2)
                    n_batches += 1

                if self.debug_print:
                    print("Epoch %s: mse (per-pixel) = %g" % (epoch, mse_acc / max(1, n_batches)))

    def run_visible(self, data):
        """
        Run the network on a set of visible units to get a sample of the hidden units.
        
        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.
        
        Returns
        -------
        hidden_states: A matrix of hidden units sampled from the visible units.
        """
        # Add bias unit before computing hidden probabilities
        data_with_bias = np.insert(data, 0, 1, axis=1)
        # sample_hidden returns (states, probs). For reconstruction/evaluation
        # we prefer probabilities (deterministic expectation) instead of stochastic samples
        hidden_states, hidden_probs = sample_hidden(data_with_bias, self.weights, add_bias=False)
        return hidden_states[:, 1:]  # Remove bias unit

    def run_hidden(self, data):
        """
        Run the network on a set of hidden units to get a sample of the visible units.
        
        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden units.
        
        Returns
        -------
        visible_states: A matrix of visible units sampled from the hidden units.
        """
        # Add bias unit before computing visible probabilities from hidden activations
        data_with_bias = np.insert(data, 0, 1, axis=1)
        # sample_visible returns (states, probs); return probabilities for deterministic recon
        visible_states, visible_probs = sample_visible(data_with_bias, self.weights, add_bias=False)
        return visible_states[:, 1:]  # Remove bias unit

    def save_weights(self, filepath=None):

        if filepath is None:

            os.makedirs("models", exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"models/rbm_weights_{ts}.npy"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        np.save(filepath, self.weights)
        print(f"Saved RBM weights to: {filepath}")
        return filepath

    def load_weights(self, filepath):

        loaded_weights = np.load(filepath)
        if loaded_weights.shape != self.weights.shape:
            raise ValueError(
                f"Loaded weights shape {loaded_weights.shape} "
                f"does not match model shape {self.weights.shape}"
            )
        self.weights = loaded_weights
        print(f"Loaded RBM weights from: {filepath}")
    
    def daydream(self, num_samples, initial_visible=None, burn_in=0, thinning=1, 
                mode="normal", alpha=0.00005, num_particles=10):
        """
        Generate samples using Gibbs chain or Birth-Death dynamics.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        initial_visible : array-like, optional
            Initial visible states. If None, random uniform initialization is used.
        burn_in : int
            Number of initial Gibbs steps to discard.
        thinning : int
            Interval between collected samples.
        mode : {'normal', 'bd'}, default='normal'
            'normal' → regular Gibbs sampling;
            'bd' → Birth-Death dynamics sampling.
        alpha : float, optional
            Birth-Death rate coefficient (used when mode='bd').
        num_particles : int, optional
            Number of parallel particles (used when mode='bd').
        
        Returns
        -------
        samples : array
            Sampled visible states after burn-in and thinning.
        """

        # Prepare initial visible states (without bias column)
        if initial_visible is None:
            init_vis = np.random.rand(num_particles, self.num_visible)
        else:
            arr = np.asarray(initial_visible)

            if arr.ndim == 1:
                init_vis = np.repeat(arr.reshape(1, -1), num_particles, axis=0)
            elif arr.ndim == 2:

                init_vis = np.repeat(arr[:1], num_particles, axis=0) 
            else:
                raise ValueError("initial_visible must be 1D or 2D array")

        if mode == "bd":
            print(f"[Birth-Death mode] Using {num_particles} parallel particles, alpha={alpha}")
            total_steps = int(burn_in + num_samples * thinning)
            samples, lst = gibbs_mult_chain_birthdeath(init_vis, self.weights, num_steps=total_steps, alpha=alpha)
            # selected contains the kept samples after burn-in and thinning
            selected = samples[burn_in:total_steps:thinning]
            # Keep backward compatibility: if there's a single particle, squeeze
            if selected.ndim == 3 and selected.shape[1] == 1:
                # shape (T, 1, V) -> (T, V)
                return selected.squeeze(axis=1)
            return selected, lst

        else:

            print("[Normal Gibbs mode]")
            total_steps = int(burn_in + num_samples * thinning)
            samples, lst = gibbs_mult_chain(init_vis, self.weights, total_steps)
            selected = samples[burn_in:total_steps:thinning]
            # Keep backward compatibility: if there's a single particle, squeeze
            if selected.ndim == 3 and selected.shape[1] == 1:
                return selected.squeeze(axis=1)
            return selected, lst