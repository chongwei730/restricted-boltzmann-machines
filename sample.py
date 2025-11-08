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
