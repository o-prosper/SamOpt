# Plotting functions

import matplotlib.pyplot as plt
import numpy as np

def plot_ctmc_bank_simple(purse, bank, n_show=100):
    """
    Plot up to n_show trajectories of I(t) from a CTMC bank,
    along with the mean trajectory.

    Parameters
    ----------
    purse : SIRSetup
        Contains the reporting-time vector T.
    bank : np.ndarray
        Shape (n_sims, n_times, state_dim).
        This comes from CTMC_bank(purse, n_sims).
    n_show : int, optional
        Number of sample trajectories to plot (default 100).

    Returns
    -------
    None
    """
    T = purse.T
    I_bank = bank[:, :, 1]             # all I(t) trajectories
    I_mean = I_bank.mean(axis=0)       # mean curve

    ns = min(n_show, bank.shape[0])

    plt.figure(figsize=(10, 6))

    # sample trajectories
    for k in range(ns):
        plt.plot(T, I_bank[k], alpha=0.2, color="C1")

    # mean trajectory
    plt.plot(T, I_mean, color="k", lw=2, label="Mean I(t)")

    plt.xlabel("time (days)")
    plt.ylabel("I(t)")
    plt.title(f"{ns} CTMC trajectories and mean I(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# in data_generator.py (or samopt/utils/plotting.py)

import matplotlib.pyplot as plt

def plot_ctmc(purse, X, event_times, event_states, ax=None):
    """
    Plot a single SIR CTMC trajectory with reporting-time samples.

    Parameters
    ----------
    purse : SIRSetup
        The setup object (provides T, etc.).
    X : np.ndarray
        Array of shape (len(T), state_dim) with states at reporting times.
        This is the first return value from CTMC(...).
    event_times : np.ndarray
        1D array of event times (second-to-last return from CTMC(...)).
    event_states : np.ndarray
        2D array of shape (n_events, state_dim) with states at each event.
        This is the last return value from CTMC(...).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    T = purse.T

    # colors
    col_S = "tab:blue"
    col_I = "tab:orange"
    col_R = "tab:green"

    # markers
    m_S = "o"   # circle
    m_I = "s"   # square
    m_R = "D"   # diamond

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # --- Event-time trajectories (step plots)
    ax.step(event_times, event_states[:, 0], where="post",
            alpha=0.8, color=col_S, label="S (events)")
    ax.step(event_times, event_states[:, 1], where="post",
            alpha=0.8, color=col_I, label="I (events)")
    ax.step(event_times, event_states[:, 2], where="post",
            alpha=0.8, color=col_R, label="R (events)")

    # --- Reporting-time samples (dots)
    ax.plot(T, X[:, 0], m_S, color=col_S,
            markersize=5, label="S samples (T)", zorder=5)
    ax.plot(T, X[:, 1], m_I, color=col_I,
            markersize=5, label="I samples (T)", zorder=5)
    ax.plot(T, X[:, 2], m_R, color=col_R,
            markersize=5, label="R samples (T)", zorder=5)

    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title("CTMC SIR trajectory with reporting-time samples")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return ax

