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
    visualize_hidden_weights,
    visualize_energy_landscape,
    visualize_particle_trajectories,
    visualize_energy_distribution,
)
from rbm import RBM
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--thinning", type=int, default=1000, help="Thinning")
    parser.add_argument("--burn_in", type=int, default=6000, help="Burn in")
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha")

    return parser.parse_args()



def main():
    args = parse_args()
    print("Loading and binarizing MNIST...")
    save_dir = './models'
    X_train, X_test, Y_train, Y_test = load_mnist_binarized(n_train=20000, n_test=200, random_binarize=False)
    num_visible = X_train.shape[1]

    weights_path = "models/rbm_weights_latest.npy"
    rbm = RBM(num_visible=num_visible, num_hidden=256)
    
    if os.path.exists(weights_path):
        print(f"find weights path: {weights_path}")
        rbm.load_weights(weights_path)
    else:
        print("train from scratch")
        rbm.train(X_train, max_epochs=1000, learning_rate=0.001, batch_size=128, cd_k=20, reg_lambda=1e-4)
        rbm.save_weights()
        rbm.save_weights(weights_path)


    visualize_energy_distribution(rbm, X_test, Y_test, save_dir=save_dir)
    visualize_originals(X_test, save_dir=save_dir)
    visualize_reconstructions(rbm, X_test, save_dir=save_dir)
    visualize_hidden_weights(rbm, save_dir=save_dir)
    sample = X_train[0]  # choose a single image
    visualize_energy_landscape(rbm, sample, dim_idx=(100, 150), save_dir=save_dir)

    # Generate multi-particle trajectories and visualize them dynamically.
    # Use multiple particles to see diversity; adjust burn_in/thinning as needed.



    num_particles = 100
    num_samples = 50
    burn_in = args.burn_in
    thinning = args.thinning
    impaired_X_test = X_test[5].copy()


    impaired_X_test = impaired_X_test.reshape(1, 28, 28)
    impaired_X_test[:, :, 14:] = 0.0 

    impaired_X_test = impaired_X_test.reshape(1, -1)
    impaired_X_test = impaired_X_test

    visualize_originals(impaired_X_test, save_dir=save_dir, name="impaired_sample")

    alpha = args.alpha
    np.random.seed(args.seed)
    init_vis = np.random.rand(num_particles, num_visible)
    save_dir = f"seed-{args.seed}/"




    print(f"Generating trajectories from scratch: num_particles={num_particles}, num_samples={num_samples}")
    # trajectories shape expected: (T, P, V) or (T, V) for single-particle
    # trajectories = rbm.daydream(
    #     initial_visible=init_vis,
    #     num_samples=num_samples,
    #     burn_in=burn_in,
    #     alpha=alpha,
    #     thinning=thinning,
    #     mode="bd",
    #     num_particles=num_particles,
    # )
    # visualize_particle_trajectories(
    #     trajectories,
    #     save_dir=f"./bd_vis_output/{save_dir}",
    #     filename_prefix=f"particle_traj_bd",
    #     shape=(28, 28),
    #     cols=5,
    #     interval=400,
    # )




    # trajectories = rbm.daydream(
    #     initial_visible=init_vis,
    #     num_samples=num_samples,
    #     burn_in=burn_in,
    #     thinning=thinning,
    #     mode="normal",
    #     num_particles=num_particles,
    # )
    # visualize_particle_trajectories(
    #     trajectories,
    #     save_dir=f"./normal_vis_output/{save_dir}",
    #     filename_prefix=f"particle_traj_normal",
    #     shape=(28, 28),
    #     cols=5,
    #     interval=400,
    # )




    # print(f"Generating trajectories from impaired data: num_particles={num_particles}, num_samples={num_samples}")
    base_save_dir = save_dir

    bd_vis_dir = os.path.join("bd_vis_output", base_save_dir)
    normal_vis_dir = os.path.join("normal_vis_output", base_save_dir)
    trj_dir = os.path.join("trj_output", base_save_dir)

    os.makedirs(bd_vis_dir, exist_ok=True)
    os.makedirs(normal_vis_dir, exist_ok=True)
    os.makedirs(trj_dir, exist_ok=True)




    bd_trajectories, lst_bd = rbm.daydream(
        initial_visible=impaired_X_test,
        num_samples=num_samples,
        burn_in=burn_in,
        thinning=thinning,
        alpha=alpha,
        mode="bd",
        num_particles=num_particles,
    )

    visualize_particle_trajectories(
        bd_trajectories,
        save_dir=bd_vis_dir,
        filename_prefix="recon_particle_traj_bd",
        shape=(28, 28),
        cols=5,
        interval=400,
    )

    np.save(os.path.join(trj_dir, "bd_trj.npy"), bd_trajectories)


    trajectories, lst_normal = rbm.daydream(
        initial_visible=impaired_X_test,
        num_samples=num_samples,
        burn_in=burn_in,
        thinning=thinning,
        mode="normal",
        num_particles=num_particles,
    )

    visualize_particle_trajectories(
        trajectories,
        save_dir=normal_vis_dir,
        filename_prefix="recon_particle_traj_normal",
        shape=(28, 28),
        cols=5,
        interval=400,
    )

    np.save(os.path.join(trj_dir, "normal_trj.npy"), trajectories)


    min_len = min(len(lst_bd), len(lst_normal))
    bd = np.array(lst_bd[:min_len])
    normal = np.array(lst_normal[:min_len])

    plt.figure(figsize=(7, 5))
    plt.plot(bd, label='Birth-Death MCMC', linewidth=2)
    plt.plot(normal, label='Standard Gibbs', linewidth=2)

    plt.title("MMD vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("MMD² w.r.t. Ground Truth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(trj_dir, "compare.png"))
    print(f"Results saved under {trj_dir}")




    print("Done.")


if __name__ == '__main__':
    main()


