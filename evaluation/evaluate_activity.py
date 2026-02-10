import matplotlib.pyplot as plt
from brian2 import *


def plot_raster(spike_monitor, save_path=None):
    """
    Plots spike raster of hidden neurons.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(spike_monitor.t / ms, spike_monitor.i, '.k', markersize=2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    plt.title("Hidden Layer Spike Raster Plot")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def weight_distribution(input_syn, save_path=None):
    """
    Plots histogram of synaptic weights.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(input_syn.w[:], bins=30)
    plt.xlabel("Synaptic Weight")
    plt.ylabel("Count")
    plt.title("Synaptic Weight Distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def firing_rate_plot(spike_monitor, save_path=None):
    """
    Plots firing rate (Hz) of each neuron.
    """
    duration = spike_monitor.t[-1] / second
    rates = spike_monitor.count / duration

    plt.figure(figsize=(6, 4))
    plt.plot(rates)
    plt.xlabel("Neuron Index")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("Hidden Neuron Firing Rates")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()



