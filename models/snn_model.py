from brian2 import *
from models.lif_neuron import lif_equations

def build_snn(input_neurons, hidden_neurons):
    """
    Builds a simple two-layer Spiking Neural Network.
    """

    input_group = NeuronGroup(
        input_neurons,
        lif_equations(),
        threshold='v > 1',
        reset='v = 0',
        method='exact'
    )

    hidden_group = NeuronGroup(
        hidden_neurons,
        lif_equations(),
        threshold='v > 1',
        reset='v = 0',
        method='exact'
    )

    synapses = Synapses(
        input_group,
        hidden_group,
        model='w : 1',
        on_pre='v_post += w'
    )

    synapses.connect(p=0.1)
    synapses.w = 'rand()'

    return input_group, hidden_group, synapses
