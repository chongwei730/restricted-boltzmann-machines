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
    visualize_daydream_samples,
    visualize_impaired_reconstruction,
    visualize_hidden_weights,
    visualize_energy_landscape,
    visualize_particle_trajectories,
)
from rbm import RBM



def main():
    print("Loading and binarizing MNIST...")
    X_train, X_test = load_mnist_binarized(n_train=2000, n_test=200, random_binarize=False)

    rbm = RBM(num_visible=X_train.shape[1], num_hidden=128)
    rbm.train(X_train, max_epochs=200, learning_rate=0.01, batch_size=64, cd_k=1)

    # # --- Visualization Steps ---
    # Save visualizations to this folder (set to None to display instead)
    # save_dir = './bd_vis_output'
    # # mode = "normal"
    # mode = "bd"
    # print(f"Sample using {mode}")

    # visualize_originals(X_test, save_dir=save_dir)
    # visualize_reconstructions(rbm, X_test, save_dir=save_dir)
    # visualize_daydream_samples(rbm, mode=mode, save_dir=save_dir)
    # visualize_impaired_reconstruction(rbm, X_test, mode=mode, save_dir=save_dir)
    # visualize_hidden_weights(rbm, save_dir=save_dir)
    # sample = X_test[0]  # choose a single image
    # visualize_energy_landscape(rbm, sample, dim_idx=(100, 150), save_dir=save_dir)

    # Generate multi-particle trajectories and visualize them dynamically.
    # Use multiple particles to see diversity; adjust burn_in/thinning as needed.
    num_particles = 10
    num_samples = 12
    burn_in = 2000
    thinning = 1000
    print(f"Generating trajectories: num_particles={num_particles}, num_samples={num_samples}")
    trajectories = rbm.daydream(
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
        filename_prefix=f"particle_traj_bd",
        shape=(28, 28),
        cols=5,
        interval=400,
    )


    trajectories = rbm.daydream(
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
        filename_prefix=f"particle_traj_normal",
        shape=(28, 28),
        cols=5,
        interval=400,
    )



    print("Done.")


if __name__ == '__main__':
    main()


