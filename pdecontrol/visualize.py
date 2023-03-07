import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def hstepplot(hsteploss, **kwargs):

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hsteploss)
    ax.set_xlabel("H-Step")
    ax.set_ylabel("Loss")

    fig.canvas.draw()
    image = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close()
    return image


def pdeplot(obs=None, actions=None, opred=None, rewards=None, rpred=None, **kwargs):

    fig, axes = plt.subplots(ncols=5, figsize=(20, 3))
    axes[0].set_title("Ground Truth")
    axes[1].set_title("Predicted")
    axes[2].set_title("Actions")
    axes[3].set_title("Squared Error")
    axes[4].set_ylabel("Reward")
    axes[4].set_xlabel("Step")

    if obs is not None and opred is not None:
        error = (obs - opred) ** 2

        vmin = min(np.min(obs), np.min(opred))
        vmax = max(np.max(obs), np.max(opred))

        shw0 = axes[0].imshow(
            obs.T, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax
        )
        plt.colorbar(shw0, ax=axes[0])

        shw1 = axes[1].imshow(
            opred.T, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax
        )
        plt.colorbar(shw1, ax=axes[1])

        shw3 = axes[3].imshow(error.T, interpolation="nearest", aspect="auto")
        plt.colorbar(shw3, ax=axes[3])

    if actions is not None:
        shw2 = axes[2].imshow(actions.T, interpolation="nearest", aspect="auto")
        plt.colorbar(shw2, ax=axes[2])

    if rewards is not None and rpred is not None:
        steps = np.arange(rewards.size)
        axes[4].plot(steps, rpred, label="Predicted")
        axes[4].plot(steps, rewards, label="Rewards")

        axes[4].legend()

    fig.canvas.draw()
    image = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close()

    return image


def spatial(keys, **kwargs):
    num_plots = len(keys)
    fig, axes = plt.subplots(ncols=num_plots, figsize=(5 * num_plots, 3))

    kwargs = {key: values for key, values in kwargs.items() if key in keys}

    vmin = min([np.min(values) for values in kwargs.values()])
    vmax = max([np.max(values) for values in kwargs.values()])
    
    for ax, (key, values) in zip(axes, kwargs.items()):
        ax.set_title(key)
        plot = ax.imshow(values.T, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(plot, ax=ax)
    

    fig.canvas.draw()
    image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()

    return image


def epplot(obs, actions, rewards):
    obs, actions, rewards = (
        np.squeeze(obs, axis=1),
        np.squeeze(actions, axis=1),
        np.squeeze(rewards),
    )

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    axes[0].set_title("PDE")
    shw0 = axes[0].imshow(obs.T, interpolation="nearest", aspect="auto")
    plt.colorbar(shw0, ax=axes[0])

    axes[1].set_title("Actions")
    shw1 = axes[1].imshow(actions.T, interpolation="nearest", aspect="auto")
    plt.colorbar(shw1, ax=axes[1])

    steps = np.arange(rewards.size)
    axes[2].plot(steps, rewards)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Timestep")

    plt.tight_layout()

    fig.canvas.draw()
    image = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close()

    return image
