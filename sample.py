import numpy as np
import torch 
from utils import rbf_mmd



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
        print("start cal")
        mmd_val = rbf_mmd(current_visible[:, 1:], target_samples)
        print(mmd_val)
        lst.append(mmd_val)


    return samples, lst







def gibbs_mult_chain_torch(initial_visible, weights, num_steps, device="cuda"):
    """
    GPU version of gibbs_mult_chain using PyTorch.

    initial_visible: (N, num_visible), NO BIAS
    weights: (1+num_visible, 1+num_hidden)
    """
    # ============================================================
    # 1. Ensure tensors are torch + cuda
    # ============================================================
    if not torch.is_tensor(initial_visible):
        v = torch.tensor(initial_visible, dtype=torch.float32, device=device)
    else:
        v = initial_visible.to(device).float()

    if not torch.is_tensor(weights):
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        weights = weights.to(device).float()

    N = v.shape[0]
    num_visible = weights.shape[0] - 1

    # ============================================================
    # 2. Add visible bias column
    # ============================================================
    if v.shape[1] == num_visible:
        ones = torch.ones((N, 1), device=device)
        v = torch.cat([ones, v], dim=1)   # (N, 1+num_visible)

    # ============================================================
    # 3. Allocate storage (T, N, V)
    # ============================================================
    samples = torch.zeros((num_steps, N, num_visible), device=device)

    # store initial sample (exclude bias)
    samples[0] = v[:, 1:]

    # ============================================================
    # 4. Main Gibbs loop
    # ============================================================
    for t in range(1, num_steps):

        # GPU-based Gibbs sampling
        v_new, v_prob, h_state, h_prob = gibbs_step_torch(v[:, 1:], weights)

        # v_new already includes bias
        v = v_new

        samples[t] = v[:, 1:]   # no bias

    return samples








def sample_hidden_torch(v_with_bias, weights, add_bias=False):
    """
    v_with_bias: (N, 1+num_visible)
    weights: (1+num_visible, 1+num_hidden)
    """
    activations = v_with_bias @ weights  # (N, 1+num_hidden)
    # hidden block is activations[:,1:]
    probs = torch.sigmoid(activations[:, 1:])  # (N, num_hidden)

    # sample Bernoulli
    states = (torch.rand_like(probs) < probs).float()

    if add_bias:
        # prepend bias=1
        ones = torch.ones((probs.shape[0], 1), device=probs.device)
        states = torch.cat([ones, states], dim=1)

    return states, probs


def sample_visible_torch(h_with_bias, weights, add_bias=False):
    """
    h_with_bias: (N, 1+num_hidden)
    weights: (1+num_visible, 1+num_hidden)^T  ← note RBM weight symmetry
    """
    activations = h_with_bias @ weights.T  # (N, 1+num_visible)
    probs = torch.sigmoid(activations[:, 1:])  # (N, num_visible)

    states = (torch.rand_like(probs) < probs).float()

    if add_bias:
        ones = torch.ones((probs.shape[0], 1), device=probs.device)
        states = torch.cat([ones, states], dim=1)

    return states, probs


def gibbs_step_torch(visible_data, weights):
    """
    visible_data: (N, num_visible)  (NO BIAS)
    weights: (1+num_visible, 1+num_hidden)
    """

    device = visible_data.device

    # ---- Add visible bias ----
    ones = torch.ones((visible_data.shape[0], 1), device=device)
    v_with_bias = torch.cat([ones, visible_data], dim=1)

    # ---- Sample h|v ----
    h_states, h_probs = sample_hidden_torch(v_with_bias, weights, add_bias=True)
    # h_states: (N, 1+num_hidden)

    # ---- Sample v|h ----
    v_states, v_probs = sample_visible_torch(h_states, weights, add_bias=True)
    # v_states: (N, 1+num_visible)

    # return without leading bias on visible
    return v_states, v_probs, h_states, h_probs

def gibbs_mult_chain_birthdeath(initial_visible, weights, num_steps, alpha=0.1, dt=1.0):
    """
    Correct RBM Birth-Death Sampling (BD-FP version).
    """
    target_samples = np.load("./ground_truth.npy")
    num_visible = weights.shape[0] - 1
    num_hidden = weights.shape[1] - 1
    N = initial_visible.shape[0]

    samples = np.zeros((num_steps, N, num_visible))

    # add bias
    v = initial_visible.copy()

    # enforce v_no_bias must be num_visible dim
    if v.shape[1] != num_visible + 1: 
        # then initial_visible had no bias column
        v = np.insert(v, 0, 1, axis=1)
    else:
        # initial_visible already has bias; ensure first col = 1
        v[:, 0] = 1

    # extract rbm parameters
    vbias = weights[1:, 0]        
    hbias = weights[0, 1:]        
    W     = weights[1:, 1:]     
    samples[0] = v[:, 1:]
    def free_energy(v_no_bias):
        wx_b = v_no_bias @ W + hbias
        hidden_term = np.sum(np.log1p(np.exp(wx_b)), axis=1)
        visible_term = v_no_bias @ vbias
        return -visible_term - hidden_term

    lst = []

    for t in range(1, num_steps):

        # ---- Gibbs ----
        v_gibbs, v_prob, h_state, h_prob = gibbs_step(v[:, 1:], weights)

        # re-add bias column
        v = v_gibbs
        v[:, 0] = 1  
        v_no_bias = v[:, 1:]

 
    

        # ---- Free Energy ----
        E = free_energy(v_no_bias)
        E_mean = np.mean(E)
        beta = E - E_mean

        # ---- Birth–Death ----
        p_death = dt * alpha * np.maximum(beta, 0)
        p_birth = dt * alpha * np.maximum(-beta, 0)

        # clip to valid probability range
        p_death = np.clip(p_death, 0, 1)
        p_birth = np.clip(p_birth, 0, 1)

        death_mask = np.random.rand(N) < p_death
        n_deaths = np.sum(death_mask)

        if n_deaths > 0:

            # if no one has birth probability > 0, fallback to uniform
            if np.sum(p_birth) == 0:
                birth_weights = np.ones(N) / N
            else:
                birth_weights = p_birth / np.sum(p_birth)

            parents = np.random.choice(N, size=n_deaths, p=birth_weights)

            # replace dead particles with birth parents
            v[death_mask] = v[parents]

        samples[t] = v[:, 1:]
        print("start cal")
        mmd_val = rbf_mmd(v[:, 1:], target_samples)
        print(mmd_val)
        lst.append(mmd_val)


    return samples, lst






def gibbs_mult_chain_birthdeath_torch(initial_visible, weights, num_steps,
                                      alpha=0.1, dt=1.0, device='cuda'):

    # move inputs to GPU
    v = initial_visible.to(device)
    weights = weights.to(device)

    num_visible = weights.shape[0] - 1
    num_hidden = weights.shape[1] - 1
    N = v.shape[0]

    samples = torch.zeros(num_steps, N, num_visible, device=device)

    # ensure bias column
    if v.shape[1] != num_visible + 1:
        ones = torch.ones(N, 1, device=device)
        v = torch.cat([ones, v], dim=1)
    else:
        v[:, 0] = 1.0

    # parameters
    vbias = weights[1:, 0]
    hbias = weights[0, 1:]
    W     = weights[1:, 1:]

    def free_energy(v_no_bias):
        wx_b = v_no_bias @ W + hbias
        hidden_term = torch.log1p(torch.exp(wx_b)).sum(dim=1)
        visible_term = v_no_bias @ vbias
        return -(visible_term + hidden_term)

    samples[0] = v[:, 1:]

    for t in range(1, num_steps):

        # ---- Gibbs ----
        v_gibbs, _, _, _ = gibbs_step_torch(v[:, 1:], weights)
        v = v_gibbs
        v[:, 0] = 1.0

        v_no_bias = v[:, 1:]

        # ---- Free Energy ----
        E = free_energy(v_no_bias)
        E_mean = E.mean()
        beta = E - E_mean

        # ---- Birth–Death ----
        p_death = dt * alpha * torch.clamp(-beta, min=0)
        p_birth = dt * alpha * torch.clamp(beta, min=0)

        p_death = torch.clamp(p_death, 0, 1)
        p_birth = torch.clamp(p_birth, 0, 1)

        death_mask = torch.rand(N, device=device) < p_death
        n_deaths = death_mask.sum().item()

        if n_deaths > 0:
            if p_birth.sum() == 0:
                birth_weights = torch.ones(N, device=device) / N
            else:
                birth_weights = p_birth / p_birth.sum()

            parents = torch.multinomial(birth_weights, n_deaths, replacement=True)

            v[death_mask] = v[parents]

        samples[t] = v[:, 1:]

    return samples