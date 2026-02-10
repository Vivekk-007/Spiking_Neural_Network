from brian2 import *

def lif_equations():
    """
    Leaky Integrate-and-Fire neuron equations.
    """
    eqs = """
    dv/dt = -v / (10*ms) : 1
    """

    return eqs
