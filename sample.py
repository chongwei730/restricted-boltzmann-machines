import numpy as np
import torch 
from utils import rbf_mmd
from scipy.spatial.distance import pdist


def _logistic(x, T=2):
    """
    Compute the logistic sigmoid function.
    
    Parameters
    ----------
    x : array-like
        Input data.
        
    Returns
    -------
    array-like
        The logistic sigmoid of the input.
    """
    return 1.0 / (1 + np.exp(-x / T))

def sample_hidden(visible_data, weights, add_bias=True):
    """
    Sample hidden units given visible units.
    
    Parameters
    ----------
    visible_data : array-like
        A matrix where each row consists of the states of the visible units.
    weights : array-like
        The weight matrix of the RBM.
    add_bias : bool, optional
        Whether to add bias unit to the input data. Default is True.
        
    Returns
    -------
    tuple
        (hidden_states, hidden_probs) where hidden_states are the sampled binary states
        and hidden_probs are the probabilities.
    """
    num_examples = visible_data.shape[0]
    num_hidden = weights.shape[1] - 1
    
    # Add bias unit if needed
    if add_bias:
        visible_data = np.insert(visible_data, 0, 1, axis=1)
    
    # Calculate activations and probabilities
    hidden_activations = np.dot(visible_data, weights)
    hidden_probs = _logistic(hidden_activations)
    
    # Sample states
    hidden_states = hidden_probs > np.random.rand(num_examples, num_hidden + 1)
    
    # Always fix the bias unit to 1 in both probabilities and states
    hidden_probs[:, 0] = 1
    hidden_states[:, 0] = 1
    
    return hidden_states, hidden_probs

def sample_visible(hidden_data, weights, add_bias=True):
    """
    Sample visible units given hidden units.
    
    Parameters
    ----------
    hidden_data : array-like
        A matrix where each row consists of the states of the hidden units.
    weights : array-like
        The weight matrix of the RBM.
    add_bias : bool, optional
        Whether to add bias unit to the input data. Default is True.
        
    Returns
    -------
    tuple
        (visible_states, visible_probs) where visible_states are the sampled binary states
        and visible_probs are the probabilities.
    """
    num_examples = hidden_data.shape[0]
    num_visible = weights.shape[0] - 1
    
    # Add bias unit if needed
    if add_bias:
        hidden_data = np.insert(hidden_data, 0, 1, axis=1)
    
    # Calculate activations and probabilities
    visible_activations = np.dot(hidden_data, weights.T)
    visible_probs = _logistic(visible_activations)
    
    # Sample states
    visible_states = visible_probs > np.random.rand(num_examples, num_visible + 1)
    
    # Always fix the bias unit to 1 in both probabilities and states
    visible_probs[:, 0] = 1
    visible_states[:, 0] = 1
    
    return visible_states, visible_probs

def gibbs_step(visible_data, weights):
    """
    Perform one step of Gibbs sampling.
    
    Parameters
    ----------
    visible_data : array-like
        A matrix where each row consists of the states of the visible units.
    weights : array-like
        The weight matrix of the RBM.
        
    Returns
    -------
    tuple
        (new_visible_states, new_visible_probs, hidden_states, hidden_probs)
    """
    # Add bias to visible data
    visible_data_with_bias = np.insert(visible_data, 0, 1, axis=1)
    
    # Sample hidden given visible
    hidden_states, hidden_probs = sample_hidden(visible_data_with_bias, weights, add_bias=False)
    
    # Sample visible given hidden (use hidden_states with bias)
    hidden_states_with_bias = np.insert(hidden_states[:, 1:], 0, 1, axis=1)
    visible_states, visible_probs = sample_visible(hidden_states_with_bias, weights, add_bias=False)
    
    return visible_states, visible_probs, hidden_states, hidden_probs

def gibbs_chain(initial_visible, weights, num_steps):
    """
    Run a Gibbs chain for a specified number of steps.
    
    Parameters
    ----------
    initial_visible : array-like
        Initial state of visible units.
    weights : array-like
        The weight matrix of the RBM.
    num_steps : int
        Number of Gibbs steps to perform.
        
    Returns
    -------
    array-like
        A matrix where each row is a sample of the visible units produced
        during the Gibbs sampling chain.
    """
    num_visible = weights.shape[0] - 1
    samples = np.ones((num_steps, initial_visible.shape[0], num_visible + 1))
    
    # Initialize with the provided visible state
    current_visible = initial_visible
    if current_visible.shape[1] == num_visible:  # If no bias unit
        current_visible = np.insert(current_visible, 0, 1, axis=1)
    
    samples[0] = current_visible
    
    # Run the Gibbs chain
    for i in range(1, num_steps):
        visible_states, visible_probs, _, _ = gibbs_step(samples[i-1, :, 1:], weights)
        samples[i] = visible_states
    
    # Return samples without the bias unit
    return samples[:, :, 1:]



def gibbs_mult_chain(initial_visible, weights, num_steps):
    """
    Run multiple Gibbs chains in parallel for a specified number of steps.
    
    Parameters
    ----------
    initial_visible : array-like, shape (N, num_visible)
        Initial batch of visible states (N = number of independent chains).
    weights : array-like
        The RBM weight matrix (including bias units).
    num_steps : int
        Number of Gibbs sampling steps to perform.
        
    Returns
    -------
    samples : array, shape (num_steps, N, num_visible)
        Visible samples across all chains and steps.
    """
    target_samples = np.load("./ground_truth.npy")
    num_visible = weights.shape[0] - 1
    N = initial_visible.shape[0]  # number of chains

    # Allocate storage: (steps, chains, num_visible)
    samples = np.zeros((num_steps, N, num_visible))

    # Add bias unit if missing
    current_visible = initial_visible.copy()
    if current_visible.shape[1] == num_visible:
        current_visible = np.insert(current_visible, 0, 1, axis=1)

    # Save initial visible states
    samples[0] = current_visible[:, 1:]
    
    lst = []

    # Main Gibbs loop
    for t in range(1, num_steps):
        # One full Gibbs step for all chains
        visible_states, visible_probs, hidden_states, hidden_probs = gibbs_step(
            current_visible[:, 1:], weights
        )

        # Update current visible layer
        current_visible = visible_states

        # Store current sample (excluding bias)
        samples[t] = current_visible[:, 1:]
        mmd_val = rbf_mmd(current_visible[:, 1:], target_samples)
        lst.append(mmd_val)

        print(mmd_val)


    return samples, lst






def gibbs_mult_chain_birthdeath(initial_visible, weights, num_steps, alpha=0.1, dt=1.0):
    """
    RBM Birth-Death Sampling (BD-FP version) with KDE (Hamming kernel + median heuristic sigma)
    """
    target_samples = np.load("./ground_truth.npy")
    num_visible = weights.shape[0] - 1
    num_hidden = weights.shape[1] - 1
    N = initial_visible.shape[0]

    samples = np.zeros((num_steps, N, num_visible))
    current_visible = initial_visible.copy()
    if current_visible.shape[1] == num_visible:
        current_visible = np.insert(current_visible, 0, 1, axis=1)

    # === bias handling ===
    v = initial_visible.copy()
    if v.shape[1] != num_visible + 1:
        v = np.insert(v, 0, 1, axis=1)
    else:
        v[:, 0] = 1
    samples[0] = v[:, 1:]

    # === extract parameters ===
    vbias = weights[1:, 0]
    hbias = weights[0, 1:]
    W     = weights[1:, 1:]

    # === helper functions ===
    def free_energy(v_no_bias):
        wx_b = v_no_bias @ W + hbias
        hidden_term = np.sum(np.log1p(np.exp(wx_b)), axis=1)
        visible_term = v_no_bias @ vbias
        return -visible_term - hidden_term

    def kde_log_density_binary(X, sigma=2.0):
        # Hamming kernel KDE
        N = X.shape[0]
        D = np.sum(X[:, None, :] != X[None, :, :], axis=2)
        K = np.exp(-D / sigma)
        rho = np.mean(K, axis=1) + 1e-12
        return np.log(rho)

    def median_heuristic_sigma(X, c=0.5):
        dists = pdist(X, metric="euclidean")
        return c * np.median(dists)

    lst = []

    # === main loop ===
    for t in range(1, num_steps):

        # 1. Gibbs sampling
            v_gibbs, v_prob, h_state, h_prob = gibbs_step(v[:, 1:], weights)
            v = v_gibbs
            v[:, 0] = 1
            v_no_bias = v[:, 1:]


            # 2. Free energy
            V = free_energy(v_no_bias)

            # 3. Bandwidth via median heuristic
            sigma = median_heuristic_sigma(v_no_bias, c=0.5)

            # 4. KDE log-density (Hamming kernel)
            log_rho = kde_log_density_binary(v_no_bias, sigma=sigma)

            # 5. Relative density difference β_i
            beta = log_rho + V
            beta -= np.mean(beta)
            beta /= (np.std(beta) + 1e-12)



            p_target = 0.1
            max_rate = float(np.max(np.abs(beta)))
            dt = min(1e-2, p_target / (max_rate + 1e-12))

            # 6. Birth–Death
            rate_death = dt * np.maximum(beta, 0)
            rate_birth = dt * np.maximum(-beta, 0)
            p_birth = 1.0 - np.exp(-rate_birth )
            p_death = 1.0 - np.exp(-rate_death )
            pmax = max(p_birth.max(), p_death.max())
            print("pmax", pmax)
            
      
          
            p_death = np.clip(p_death, 0, 1)
            p_birth = np.clip(p_birth, 0, 1)

            death_mask = np.random.rand(N) < p_death
            n_deaths = np.sum(death_mask)
            if n_deaths > 0:
                birth_weights = p_birth / np.sum(p_birth) if np.sum(p_birth) > 0 else np.ones(N) / N
                parents = np.random.choice(N, size=n_deaths, p=birth_weights)
                v[death_mask] = v[parents]

            # 7. Record and diagnostics
            samples[t] = v[:, 1:]
            mmd_val = rbf_mmd(v[:, 1:], target_samples)
            lst.append(mmd_val)
            bd_rate   = death_mask.mean()        
            mean_pd   = p_death.mean()
            mean_pb   = p_birth.mean()
            beta_std  = beta.std()
            print(f"Step {t:03d} | bd_rate={bd_rate:.3%}  mean(p_d)={mean_pd:.3e}  "
                f"mean(p_b)={mean_pb:.3e}  std(beta)={beta_std:.3e}")
            print(f"Step {t:03d} | mean(F)={V.mean():.3f}, mean(beta)={beta.mean():.3f}, sigma={sigma:.3f}, MMD={mmd_val:.5f}")

    return samples, lst


