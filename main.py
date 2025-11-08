"""
mnist_experiment.py

Small experiment script that loads binary (binarized) MNIST data,
trains the RBM from this repo on a small subset, shows reconstructions
and generates a few samples.

This script tries to use TensorFlow's Keras MNIST loader first, and
falls back to scikit-learn's fetch_openml if TensorFlow isn't available.

The script intentionally uses a small number of epochs / examples by
default so it runs quickly for experimentation. Tune parameters as needed.
"""
from utils import (
    load_mnist_binarized,
    visualize_originals,
    visualize_reconstructions,
    # visualize_daydream_samples,
    # visualize_impaired_reconstruction,
    visualize_hidden_weights,
    visualize_energy_landscape,
    visualize_particle_trajectories,
    visualize_energy_distribution,
    compute_energy,
    show_grid
)
from rbm import RBM
import numpy as np
import matplotlib.pyplot as plt
import os



def main():
    print("Loading and binarizing MNIST...")
    save_dir = './models'
    X_train, X_test, Y_train, Y_test = load_mnist_binarized(n_train=2000, n_test=200, random_binarize=False)

    weights_path = "models/rbm_weights_latest.npy"
    rbm = RBM(num_visible=X_train.shape[1], num_hidden=128)
    
    if os.path.exists(weights_path):
        print(f"find weights path: {weights_path}")
        rbm.load_weights(weights_path)
    else:
        print("train from scratch")
        rbm.train(X_train, max_epochs=200, learning_rate=0.01, batch_size=64, cd_k=5)
        rbm.save_weights()
        rbm.save_weights(weights_path)



    visualize_energy_distribution(rbm, X_test, Y_test, save_dir=save_dir)


    # # --- Visualization Steps ---
    # Save visualizations to this folder (set to None to display instead)
    # mode = "normal"


    visualize_originals(X_test, save_dir=save_dir)
    visualize_reconstructions(rbm, X_test, save_dir=save_dir)
    # visualize_daydream_samples(rbm, mode=mode, save_dir=save_dir)
    # visualize_impaired_reconstruction(rbm, X_test, mode=mode, save_dir=save_dir)
    visualize_hidden_weights(rbm, save_dir=save_dir)
    sample = X_test[0]  # choose a single image
    visualize_energy_landscape(rbm, sample, dim_idx=(100, 150), save_dir=save_dir)

    # Generate multi-particle trajectories and visualize them dynamically.
    # Use multiple particles to see diversity; adjust burn_in/thinning as needed.



    num_particles = 10
    num_samples = 12
    burn_in = 6000
    thinning = 1000



    impaired_X_test = X_test.copy()

    N = impaired_X_test.shape[0]
    impaired_X_test = impaired_X_test.reshape(N, 28, 28)
    impaired_X_test[:, :, 14:] = 0.0 

    impaired_X_test = impaired_X_test.reshape(N, -1)


    n_show = 5
    show_grid(
        [X_test[i].reshape(28, 28) for i in range(n_show)] +
        [impaired_X_test[i].reshape(28, 28) for i in range(n_show)],
        titles=['orig']*n_show + ['impaired']*n_show,
        cols=n_show,
        save_dir=save_dir,
        filename_prefix='impaired_samples'
    )

    # print(f"Generating trajectories from scratch: num_particles={num_particles}, num_samples={num_samples}")
    # trajectories = rbm.daydream(
    #     num_samples=num_samples,
    #     burn_in=burn_in,
    #     thinning=thinning,
    #     mode="bd",
    #     num_particles=num_particles,
    # )

    # # trajectories shape expected: (T, P, V) or (T, V) for single-particle
    # visualize_particle_trajectories(
    #     trajectories,
    #     save_dir="./bd_vis_output",
    #     filename_prefix=f"particle_traj_bd",
    #     shape=(28, 28),
    #     cols=5,
    #     interval=400,
    # )


    # trajectories = rbm.daydream(
    #     num_samples=num_samples,
    #     burn_in=burn_in,
    #     thinning=thinning,
    #     mode="normal",
    #     num_particles=num_particles,
    # )

    # # trajectories shape expected: (T, P, V) or (T, V) for single-particle
    # visualize_particle_trajectories(
    #     trajectories,
    #     save_dir="./normal_vis_output",
    #     filename_prefix=f"particle_traj_normal",
    #     shape=(28, 28),
    #     cols=5,
    #     interval=400,
    # )





    print(f"Generating trajectories from impaired data: num_particles={num_particles}, num_samples={num_samples}")
    trajectories = rbm.daydream(
        initial_visible=impaired_X_test,
        num_samples=num_samples,
        burn_in=burn_in,
        thinning=thinning,
        mode="bd",
        num_particles=num_particles,
    )

    # trajectories shape expected: (T, P, V) or (T, V) for single-particle
    visualize_particle_trajectories(
        trajectories,
        save_dir="./bd_vis_output",
        filename_prefix=f"recon_particle_traj_bd",
        shape=(28, 28),
        cols=5,
        interval=400,
    )


    trajectories = rbm.daydream(
        initial_visible=impaired_X_test,
        num_samples=num_samples,
        burn_in=burn_in,
        thinning=thinning,
        mode="normal",
        num_particles=num_particles,
    )

    # trajectories shape expected: (T, P, V) or (T, V) for single-particle
    visualize_particle_trajectories(
        trajectories,
        save_dir="./normal_vis_output",
        filename_prefix=f"recon_particle_traj_normal",
        shape=(28, 28),
        cols=5,
        interval=400,
    )

    print("Done.")


if __name__ == '__main__':
    main()


