import numpy as np

def _logistic(x):
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
    return 1.0 / (1 + np.exp(-x))

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

    return samples



def gibbs_mult_chain_birthdeath(initial_visible, weights, num_steps, alpha=0.1):
    """
    Run a parallel Gibbs chain with Birth-Death resampling mechanism.
    
    Parameters
    ----------
    initial_visible : array-like, shape (N, num_visible)
        Initial batch of visible units (N = number of particles).
    weights : array-like
        The RBM weight matrix.
    num_steps : int
        Number of Gibbs + Birth-Death steps to perform.
    alpha : float, optional
        Birth-Death rate coefficient (controls how strongly high-energy
        samples are removed and low-energy samples replicated).
        
    Returns
    -------
    samples : array, shape (num_steps, N, num_visible)
        Sequence of visible samples across all particles.
    """
    num_visible = weights.shape[0] - 1
    num_hidden = weights.shape[1] - 1
    N = initial_visible.shape[0]

    # Initialize storage
    samples = np.zeros((num_steps, N, num_visible))
    current_visible = initial_visible.copy()

    # Add bias if needed
    if current_visible.shape[1] == num_visible:
        current_visible = np.insert(current_visible, 0, 1, axis=1)

    samples[0] = current_visible[:, 1:]

    for t in range(1, num_steps):
        # ----- Step 1: Gibbs update -----
        v_next, v_probs, h_states, h_probs = gibbs_step(current_visible[:, 1:], weights)

        # ----- Step 2: Compute energies -----
        # Remove bias before computing E = -v^T W h
        v_no_bias = v_next[:, 1:]
        h_no_bias = h_states[:, 1:]
        E = -np.sum(v_no_bias @ weights[1:, 1:] * h_no_bias, axis=1)
        E_mean = np.mean(E)

        # ----- Step 3: Birth–Death resampling -----
        fitness = np.exp(-alpha * (E - E_mean))
        probs = fitness / np.sum(fitness) 
        indices = np.random.choice(N, size=N, replace=True, p=probs)

        v_resampled = v_next[indices]
        current_visible = v_resampled
        samples[t] = v_resampled[:, 1:]

    return samples