from __future__ import print_function
import numpy as np
from sample import sample_hidden, sample_visible, gibbs_step, gibbs_chain, _logistic

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

    def train(self, data, max_epochs=1000, learning_rate=0.1, batch_size=None, cd_k=1):
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
                self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

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
                    self.weights += learning_rate * ((pos_associations - neg_associations) / bsize)

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
        _, hidden_probs = sample_hidden(data_with_bias, self.weights, add_bias=False)
        return hidden_probs[:, 1:]  # Remove bias unit

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
        _, visible_probs = sample_visible(data_with_bias, self.weights, add_bias=False)
        return visible_probs[:, 1:]  # Remove bias unit

    def daydream(self, num_samples, initial_visible=None, burn_in=0, thinning=1):
        """
        Generate samples using a Gibbs chain.
        
        Parameters
        ----------
        num_samples: Number of samples to generate.
        
        Returns
        -------
        samples: Matrix where each row is a sample of the visible units.
        """

        # Prepare initial visible states (without bias column) for the chain
        if initial_visible is None:
            # Random initialization: shape (1, num_visible)
            init_vis = np.random.rand(1, self.num_visible)
        else:
            arr = np.asarray(initial_visible)
            # Accept either (num_visible,) or (batch, num_visible)
            if arr.ndim == 1:
                init_vis = arr.reshape(1, -1)
            elif arr.ndim == 2:
                init_vis = arr
            else:
                raise ValueError("initial_visible must be 1D or 2D array")

        # Total steps to run: burn-in + num_samples * thinning
        total_steps = int(burn_in + num_samples * thinning)
        if total_steps <= 0:
            raise ValueError("total Gibbs steps must be positive")

        # Run Gibbs chain starting from init_vis
        samples = gibbs_chain(init_vis, self.weights, total_steps)
        # samples shape: (total_steps, batch, num_visible)

        # Select samples after burn-in with thinning
        selected = samples[burn_in:total_steps:thinning]
        # selected shape: (num_samples_selected, batch, num_visible)

        # If user requested num_samples but thinning arithmetic yields fewer/more,
        # return what we have (caller can slice).
        # Squeeze batch dim when batch==1 for convenience
        if selected.shape[1] == 1:
            return selected[:, 0, :]
        return selected



