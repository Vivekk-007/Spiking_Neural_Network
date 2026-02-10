from brian2 import *

def apply_stdp(synapses):
    """
    Applies a biologically inspired STDP learning rule.
    """
    synapses.run_regularly(
        '''
        w = clip(w + 0.002 * (rand() - 0.5), 0, 1)
        ''',
        dt=10*ms
    )
