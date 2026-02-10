from brian2 import *
from brian2 import prefs
prefs.codegen.target = "numpy"   # avoid compiler warnings

import numpy as np

from data.load_mnist import load_mnist_data
from evaluation.evaluate_activity import (
    plot_raster,
    weight_distribution,
    firing_rate_plot
)

# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------
start_scope()

# Load MNIST
x_train, y_train, _, _ = load_mnist_data()

# Network sizes
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 100

# Simulation parameters
SIM_TIME = 200 * ms
MAX_RATE = 50 * Hz

# --------------------------------------------------
# 2. INPUT: POISSON SPIKE ENCODING
# --------------------------------------------------
poisson_input = PoissonGroup(INPUT_SIZE, rates=np.zeros(INPUT_SIZE) * Hz)

# --------------------------------------------------
# 3. HIDDEN LAYER: LIF NEURONS
# --------------------------------------------------
lif_eqs = '''
dv/dt = -v / (10*ms) : 1
'''

hidden_group = NeuronGroup(
    HIDDEN_SIZE,
    lif_eqs,
    threshold='v > 1',
    reset='v = 0',
    method='exact'
)

# --------------------------------------------------
# 4. SYNAPSES WITH STDP (THIS IS THE TRAINING)
# --------------------------------------------------
input_syn = Synapses(
    poisson_input,
    hidden_group,
    model='''
    w : 1
    dApre/dt = -Apre / (20*ms) : 1 (event-driven)
    dApost/dt = -Apost / (20*ms) : 1 (event-driven)
    ''',
    on_pre='''
    v_post += w
    Apre += 0.01
    w = clip(w + Apost, 0, 1)
    ''',
    on_post='''
    Apost -= 0.01
    w = clip(w + Apre, 0, 1)
    '''
)

input_syn.connect(p=0.1)
input_syn.w = 'rand()'

# --------------------------------------------------
# 5. MONITORS
# --------------------------------------------------
spike_monitor = SpikeMonitor(hidden_group)

# --------------------------------------------------
# 6. TRAINING LOOP
# --------------------------------------------------
print("Starting SNN Training...")

N_TRAIN = 100   # number of training samples

for i in range(N_TRAIN):
    image = x_train[i].reshape(-1)

    # Pixel intensity → firing rate
    poisson_input.rates = image * MAX_RATE

    # Run network (learning happens here)
    run(SIM_TIME)

    # Reset neuron state between samples (important)
    hidden_group.v = 0

print("Training Completed.")
print("Total spikes in hidden layer:", spike_monitor.num_spikes)

# --------------------------------------------------
# 7. EVALUATION & VISUALIZATION
# --------------------------------------------------
plot_raster(spike_monitor, save_path="results/raster_plots.png")
weight_distribution(input_syn, save_path="results/weight_distribution.png")
firing_rate_plot(spike_monitor, save_path="results/firing_rates.png")




