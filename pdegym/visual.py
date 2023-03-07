import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def trisurf(env, trajectory, reference=None):

    xtilde = np.repeat(env.x, env.n_steps).reshape(-1, env.n_steps)
    xtilde = xtilde.T.reshape(-1)

    time = np.linspace(0.0, env.Tmax, env.n_steps)
    ttilde = np.repeat(time, env.N)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca(projection="3d")

    dtilde = trajectory.reshape(-1)
    ax.plot_trisurf(
        xtilde,
        ttilde,
        dtilde,
        linewidth=0.2,
        antialiased=True,
        shade=True,
        alpha=0.7,
        color="lightsteelblue",
    )

    if reference is not None:
        rtilde = reference.reshape(-1)
        ax.plot_trisurf(
            xtilde,
            ttilde,
            rtilde,
            linewidth=0.2,
            antialiased=True,
            shade=True,
            alpha=0.7,
            color="orange",
        )

    ax.set_xlabel("Spatial Domain")
    ax.set_ylabel("Time")
    ax.set_zlabel("PDE Solution")

    ax.view_init(20, -25)

    fig.canvas.draw()
    return Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def pdeplot(target, generated, actions=None):
    bsize = target.shape[0]
    bidx = np.random.randint(0, bsize)

    target = target[bidx]
    generated = generated[bidx]
    error = (target - generated) ** 2

    ncols = 3 + int(actions is not None)
    fig, axes = plt.subplots(ncols=ncols, figsize=(ncols * 4, 3))

    vmin, vmax = np.min(target), np.max(target)

    axes[0].set_title("PDE Solution")
    shw0 = axes[0].imshow(
        target.T, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax
    )
    plt.colorbar(shw0, ax=axes[0])

    axes[1].set_title("Surrogate Solution")
    shw1 = axes[1].imshow(
        generated.T, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax
    )
    plt.colorbar(shw1, ax=axes[1])

    axes[2].set_title("Squared Error")
    shw2 = axes[2].imshow(error.T, interpolation="nearest", aspect="auto")
    plt.colorbar(shw2, ax=axes[2])

    if actions is not None:
        actions = actions[bidx]
        axes[3].set_title("Applied actuation")
        shw3 = axes[3].imshow(actions.T, interpolation="nearest", aspect="auto")
        plt.colorbar(shw3, ax=axes[3])

    fig.canvas.draw()
    return Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
