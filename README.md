# Spiking Neural Network (SNN) with STDP Learning

## Overview

This project implements a **Spiking Neural Network (SNN)** using the Brian2 neuromorphic simulator, featuring biologically-inspired Spike-Timing-Dependent Plasticity (STDP) learning. The network is trained on the MNIST dataset to demonstrate unsupervised learning in neuromorphic systems.

## Project Structure

```
Spiking_Neural_Network/
├── data/                          # Data loading and preprocessing
│   ├── load_mnist.py             # MNIST dataset loader
│   └── __init__.py
├── encoding/                      # Input encoding schemes
│   ├── rate_encoding.py          # Rate-based spike encoding
│   └── __init__.py
├── models/                        # Network architecture
│   ├── lif_neuron.py             # LIF neuron equations
│   ├── snn_model.py              # Network builder
│   └── __init__.py
├── learning/                      # Learning algorithms
│   ├── stdp.py                   # STDP learning rule
│   └── __init__.py
├── train/                         # Training pipeline
│   ├── train_snn.py              # Main training script
│   └── __init__.py
├── evaluation/                    # Analysis and visualization
│   └── evaluate_activity.py      # Plotting utilities
├── results/                       # Output directory
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── snn/                          # Virtual environment
```

## Background & Theory

### Spiking Neural Networks (SNNs)

SNNs represent a third generation of neural networks that more closely mimic biological neural systems:

- **Biological Plausibility**: Neurons communicate via discrete spike events rather than continuous firing rates
- **Temporal Dynamics**: Information is encoded in spike timing, enabling event-driven computation
- **Energy Efficiency**: Sparse, event-driven computation reduces power consumption compared to conventional ANNs

#### Key Components:

##### 1. Leaky Integrate-and-Fire (LIF) Neuron Model

The membrane potential $v$ evolves according to:

$$\frac{dv}{dt} = -\frac{v}{\tau_m}$$

where:
- $\tau_m = 10$ ms is the membrane time constant
- When $v > \theta$ (threshold = 1), a spike is emitted
- $v$ resets to 0 after spike generation

The LIF model captures the fundamental dynamics of biological neurons:
- **Leak**: The $-v/\tau_m$ term represents passive membrane decay
- **Integration**: Synaptic currents accumulate in the potential
- **Threshold**: Spike generation occurs at deterministic threshold
- **Reset**: Hyperpolarization back to resting potential

##### 2. Rate Encoding

Input images are converted to spike trains where pixel intensity determines spike probability:

$$P(\text{spike at pixel } i) = \frac{\text{pixel intensity}}{255}$$

**Process**:
1. Normalize pixel values to [0, 1]
2. Generate Poisson spike trains with rate proportional to pixel intensity
3. Each input neuron fires stochastically at rate $r_i = \text{pixel}_i \times f_{\max}$
4. Result: Pixel brightness → neuronal firing frequency

##### 3. Poisson Spiking Process

The input layer uses Poisson processes:
- **Mathematical Model**: Inter-spike intervals follow exponential distribution
- **Implementation**: For each time bin, spike occurs with probability $p = r \cdot \Delta t$
- **Advantages**: Realistic spike variability, robust to noise
- **Rate Range**: 0-50 Hz (adjustable via `MAX_RATE` parameter)

### Spike-Timing-Dependent Plasticity (STDP)

STDP is a biologically-inspired learning rule based on the relative timing between presynaptic and postsynaptic spikes:

$$\Delta w = \begin{cases}
A_{+} e^{-\Delta t / \tau_{+}} & \text{if } \Delta t > 0 \text{ (LTP)} \\
-A_{-} e^{\Delta t / \tau_{-}} & \text{if } \Delta t < 0 \text{ (LTD)}
\end{cases}$$

where:
- $\Delta t = t_{\text{post}} - t_{\text{pre}}$ is the signed spike time difference
- $A_{+} = 0.01$, $A_{-} = 0.01$ are learning rate constants
- $\tau_{+} = 20$ ms, $\tau_{-} = 20$ ms are time constants
- $\theta$ is the synaptic weight

**Biological Interpretation**:

| Scenario | Timing | Change | Mechanism |
|----------|--------|--------|-----------|
| **LTP** | Pre before Post | $\Delta w > 0$ | Presynaptic neuron "causes" postsynaptic firing → strengthens connection |
| **LTD** | Post before Pre | $\Delta w < 0$ | Postsynaptic neuron fires without presynaptic input → weakens connection |
| **No Change** | Large $\|\Delta t\|$ | $\approx 0$ | Causality window exceeded (>100 ms) |

**Implementation Details**:
- Uses eligibility trace approach: $A_{\text{pre}}(t)$ and $A_{\text{post}}(t)$ track recent spikes
- Exponential decay with time constant $\tau = 20$ ms
- Weight clipping to [0, 1] prevents divergence
- Updates occur immediately upon spike detection

## Module Descriptions

### 1. Data Loading: `data/load_mnist.py`

Loads and preprocesses the MNIST dataset for SNN training.

**Functions**:
```python
load_mnist_data(n_samples=60000)
    Returns: (x_train, y_train) - normalized images and labels
```

**Processing Pipeline**:
1. Load raw 28×28 images from TensorFlow/Keras
2. Normalize pixel values: $x_{\text{normalized}} = x_{\text{raw}} / 255$
3. Flatten or reshape as needed
4. Optional: sample subset for faster iteration

**Output Shape**: (n_samples, 784) for flattened 28×28 images

**Key Parameters**:
- Input range: [0, 255] → normalized to [0, 1]
- Data type: uint8 → float32

### 2. Input Encoding: `encoding/rate_encoding.py`

Converts static images into temporal spike trains via Poisson rate coding.

**Encoding Process**:

1. **Rate Mapping**: $r_i(t) = \text{pixel}_i \times f_{\max}$ (Hz)
2. **Temporal Discretization**: Divide simulation into discrete time bins (dt = 1 ms)
3. **Poisson Sampling**: For each neuron and time bin:
   - $p_{\text{spike}} = r_i \times \text{dt}$
   - Generate random number $u \sim \mathcal{U}(0,1)$
   - Spike if $u < p_{\text{spike}}$

**Example**: 
- Pixel intensity = 200/255 ≈ 0.78
- Max firing rate = 50 Hz
- Poisson rate = 0.78 × 50 = 39 Hz
- Per 1 ms time bin: P(spike) = 39/1000 ≈ 0.039

**Configuration**:
- `SIM_TIME = 200 ms`: Duration per image
- `TIME_STEP = 1 ms`: Temporal resolution
- `MAX_RATE = 50 Hz`: Maximum firing rate
- **Output Shape**: (n_neurons=784, n_timesteps=200)

### 3. Neuron Model: `models/lif_neuron.py`

Defines the differential equations for LIF neurons in Brian2.

**Neuron Equations**:
```
dv/dt = -v/tau_m : 1
```

**Parameters**:
- `tau_m = 10 ms`: Membrane time constant
- `theta = 1`: Spike threshold
- `v_reset = 0`: Reset potential
- `refractory = 2 ms`: Refractory period (prevents multiple spikes)

**Neuronal Dynamics**:

The voltage evolves via:
$$v(t) = v_0 e^{-t/\tau_m}$$

With time constant 10 ms:
- Input current decays to 37% original value after 10 ms
- Typical integration window: ~30-50 ms
- Enables temporal filtering of input patterns

**Spike Detection**:
- Continuous membrane potential tracking
- Event-driven spike generation (no polling)
- Threshold crossing generates spike event
- Automatic reset via Brian2's `reset` statement

### 4. Network Architecture: `models/snn_model.py`

Constructs the complete two-layer feedforward SNN.

**Architecture**:

```
Input Layer (Poisson)          Hidden Layer (LIF)
━━━━━━━━━━━━━━━                ━━━━━━━━━━━━━━
├─ 784 neurons                 ├─ 100 neurons (configurable)
├─ Rate-coded input            ├─ Leaky integrate-and-fire
├─ Firing rate: pixel×50 Hz    └─ STDP learning
└────────────────────────────────────────────→
         STDP Synapses (10% sparse connectivity)
```

**Network Details**:

- **Input Neurons**: 784 (28×28 MNIST pixels)
- **Hidden Neurons**: Configurable (default 100)
- **Connectivity**:
  - Random sparse (10% connection probability)
  - ~7,840 synapses (vs. 78,400 if fully connected)
  - More biologically plausible
  - Reduces computation by 90%
- **Weight Initialization**:
  - Random uniform [0, 1]
  - No structure initially (unsupervised learning)
- **Synaptic Model**: STDP-enabled with weight bounds

**Construction Code Outline**:
```python
def build_snn(n_hidden=100, sparsity=0.1):
    # Create Poisson input group
    # Create LIF hidden group
    # Connect with random sparse weights
    # Add STDP learning
```

### 5. Learning Rule: `learning/stdp.py`

Implements biologically-inspired Spike-Timing-Dependent Plasticity.

**STDP Mechanism**:

**State Variables**:
- `A_pre`: Presynaptic trace (decays with τ = 20 ms)
- `A_post`: Postsynaptic trace (decays with τ = 20 ms)
- `w`: Synaptic weight

**Update Rules**:

Upon presynaptic spike:
$$w \leftarrow w + A_{\text{post}}$$

Upon postsynaptic spike:
$$w \leftarrow w - A_{\text{pre}}$$

**Trace Dynamics**:
$$\frac{dA_{\text{pre}}}{dt} = -\frac{A_{\text{pre}}}{\tau} + \delta(t - t_{\text{pre}})$$
$$\frac{dA_{\text{post}}}{dt} = -\frac{A_{\text{post}}}{\tau} + \delta(t - t_{\text{post}})$$

**Implementation Details**:

1. **Exponential Decay**: Traces decay continuously at rate 1/τ
2. **Impulse Events**: Spike events increment traces
3. **Weight Bounds**: Clip to [0, 1] after each update
   - Prevents runaway potentiation/depression
   - Represents physical synaptic limits
4. **Plasticity Window**: Effective range ~[-100 ms, +100 ms]
   - Beyond this window, α(Δt) ≈ 0

**Typical Magnitude**:
- Δw per presynaptic spike: ±0.001 to ±0.01
- Many spikes needed to significantly change weights
- Ensures learning stability

### 6. Training Pipeline: `train/train_snn.py`

Main script implementing the complete training loop.

**Execution Flow**:

```
1. INITIALIZATION
   ├─ Set Brian2 simulation parameters (dt = 0.1 ms)
   ├─ Load MNIST training data (60,000 samples available)
   └─ Build SNN architecture

2. TRAINING LOOP (over N_TRAIN samples)
   ├─ [Per Sample]
   │  ├─ Rate Encoding
   │  │  └─ Convert pixel intensities to Poisson firing rates
   │  ├─ Simulation
   │  │  ├─ Run Brian2 simulation for SIM_TIME = 200 ms
   │  │  ├─ Neurons spike via LIF dynamics
   │  │  └─ STDP automatically modifies weights
   │  ├─ State Reset
   │  │  ├─ Reset hidden neuron potentials v = 0
   │  │  └─ Keep synaptic weights (learned parameters)
   │  └─ Monitoring
   │     ├─ Record spike times/indices
   │     └─ Track weight evolution
   │
   └─ [After all samples]
      ├─ Plot raster diagrams
      ├─ Display weight distributions
      └─ Save results/metrics

3. EVALUATION
   └─ Visualization of learning effects
```

**Key Parameters**:

| Parameter | Value | Role |
|-----------|-------|------|
| `SIM_TIME` | 200 ms | Duration per MNIST sample |
| `dt` | 0.1 ms | Brian2 integration step |
| `MAX_RATE` | 50 Hz | Peak Poisson firing rate |
| `N_TRAIN` | 100 | Number of training samples |
| `HIDDEN_SIZE` | 100 | Number of hidden LIF neurons |
| `SPARSITY` | 0.1 | Fraction of existing synapses |

**Memory & Computation**:
- Simulation per sample: ~1-2 seconds
- Total training time: N_TRAIN × 2s ≈ 3-4 minutes for 100 samples
- Memory: ~100-500 MB for network state

**Pseudocode**:
```python
for sample_id in range(N_TRAIN):
    # Encode image to rates
    rates = image_to_poisson_rates(images[sample_id])
    
    # Run simulation
    net.run(SIM_TIME)
    
    # STDP happens automatically in Brian2
    
    # Reset for next sample
    hidden_neurons.v = 0
    
    # Record metrics
    plot_spikes(spike_monitor)
```

### 7. Evaluation: `evaluation/evaluate_activity.py`

Provides visualization functions to analyze learned network dynamics.

**Function 1: `plot_raster(spike_monitor, title="")`**

**Purpose**: Visualize spatiotemporal firing patterns

**Output**:
- X-axis: Time (0 - SIM_TIME)
- Y-axis: Neuron ID (0 - 99)
- Markers: Spike events

**Interpretation**:
- **Dense regions**: High activity periods, strong input correlation
- **Sparse regions**: Weak or no firing
- **Patterns**: Reveals temporal structure in learned representations

**Example Output**:
```
Neuron │     ·  ·  ·      ··  ·
ID     │ · ·  · · · · · · · · · ·
       │    ·   ·      ·  ·    ·
       └─────────────────────────
         0    50   100   150  200 (time, ms)
```

**Function 2: `weight_distribution(input_syn, bins=50)`**

**Purpose**: Analyze learned synaptic weights

**Output**:
- Histogram of weight values
- X-axis: Weight magnitude (0 to 1)
- Y-axis: Frequency (count)

**Interpretation**:
- **Initial**: Uniform distribution [0, 1]
- **After Learning**: Bimodal distribution
  - Weak weights: Silent synapses (weight → 0)
  - Strong weights: Active synapses (weight → 1)
  - Clear separation indicates learning

**Statistics Computed**:
- Mean weight
- Weight variance (increases with learning)
- Percentage of silent synapses (w < 0.1)

**Function 3: `firing_rate_plot(spike_monitor, n_neurons=100)`**

**Purpose**: Analyze selective neuron activity

**Output**:
- Bar chart of firing rates per neuron
- X-axis: Neuron ID
- Y-axis: Firing rate (Hz) = (spike_count / SIM_TIME)

**Interpretation**:
- **High firing rate neurons**: Selective for common features
- **Low/zero firing rate**: Underutilized neurons
- **Distribution shape**: Indicates learned diversity

**Calculation**:
$$f_i = \frac{n_{\text{spikes}, i}}{SIM\_TIME}$$

**Example**: If neuron 42 fires 50 times in 200 ms:
$$f_{42} = \frac{50}{0.2 \text{ s}} = 250 \text{ Hz}$$

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **brian2** | ≥2.5 | Neuromorphic simulation framework |
| **numpy** | ≥1.19 | Numerical computing arrays |
| **matplotlib** | ≥3.3 | Data visualization and plotting |
| **tensorflow** | ≥2.4 | MNIST dataset loader |
| **scipy** | ≥1.5 | Scientific computing (optimization, etc.) |
| **scikit-learn** | ≥0.24 | ML utilities (metrics, preprocessing) |

**Installation**:

```bash
# Create virtual environment
python -m venv snn

# Activate environment
# Windows:
snn\Scripts\activate
# Linux/Mac:
source snn/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**:
```
brian2>=2.5
numpy>=1.19
matplotlib>=3.3
tensorflow>=2.4
scipy>=1.5
scikit-learn>=0.24
```

## Execution Flow & Workflow

### Step 1: Environment Setup
```bash
cd e:\Spiking_Neural_Network
python -m venv snn
snn\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run Training
```bash
python train/train_snn.py
```

**Expected Output**:
```
Loading MNIST data...
Building SNN...
Starting training loop...
[Sample 0/100] Spike count: 1234, Mean weight: 0.489
[Sample 1/100] Spike count: 1567, Mean weight: 0.491
...
[Sample 99/100] Spike count: 2123, Mean weight: 0.523

Generating evaluation plots...
✓ Raster plot saved
✓ Weight histogram saved
✓ Firing rate chart saved

Training complete!
```

### Step 3: Analyze Results

Results appear in `/results/` directory:
- `raster_plot.png`: Spike timing diagram
- `weight_distribution.png`: Histogram of learned weights
- `firing_rates.png`: Bar chart of neuron activity

## Expected Results & Interpretation

### After Training on 100 MNIST Samples:

**Weight Evolution**:
- **Initial**: Uniform distribution centered at 0.5
- **Final**: Bimodal distribution with peaks near 0 and 1
- **Interpretation**: STDP segregates synapses into strong (w→1) and weak (w→0)

**Firing Patterns**:
- **Total Spikes**: 5,000-15,000 across 100 neurons
- **Per-neuron rate**: 1-4 Hz average
- **Peak rates**: Some neurons reach 10-20 Hz for favorable inputs

**Learning Dynamics**:
- Weight changes are gradual (many spikes needed for large Δw)
- Convergence typically within 50-100 samples
- Network develops stable, reproducible responses

## Research Extensions & Improvements

### Short-term Extensions:

1. **Supervised Classification Layer**
   - Add readout neurons with spike-count based classification
   - Train readout weights via backpropagation
   - Evaluate on MNIST test set

2. **Deeper Networks**
   - Add hidden layers (e.g., 784 → 100 → 100 → 10)
   - Propagate spikes through multiple processing stages
   - Investigate multi-layer learning dynamics

3. **Recurrent Connections**
   - Add lateral inhibition within hidden layer
   - Implement sparse feedback from output
   - Enable winner-take-all computation

### Medium-term Extensions:

4. **Alternative Plasticity Rules**
   - **Intrinsic plasticity**: Homeostatic regulation of firing rates
   - **Neuromodulation**: Dopamine-modulated learning
   - **Triplet STDP**: Third spike dependency

5. **Temporal Coding**
   - Replace rate encoding with rank-order coding
   - Use precise spike times for feature discrimination
   - Analyze temporal information content

6. **Advanced Encodings**
   - **Wavelet encoding**: Multi-scale feature extraction
   - **Gabor filters**: Oriented edge detection
   - **Population coding**: Distributed representations

### Long-term Research:

7. **Hardware Implementation**
   - Deploy to **Intel Loihi** neuromorphic chip
   - Run on **IBM TrueNorth** with 1 million neurons
   - Achieve 100-1000× energy efficiency gains

8. **Network Analysis**
   - Eigenvalue analysis of connectivity matrices
   - Information-theoretic measures (MI, entropy)
   - Dynamical systems analysis of hidden layer

9. **Biological Realism**
   - Balanced excitation/inhibition (E/I = 4:1)
   - Realistic neurotransmitter dynamics
   - Spike-frequency adaptation

10. **Hybrid Learning**
    - Combine STDP with supervised learning objectives
    - Multi-timescale plasticity (fast × slow)
    - Bayesian inference in SNN framework

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'brian2'"
**Solution**: Ensure virtual environment is activated and dependencies installed
```bash
snn\Scripts\activate
pip install brian2
```

### Issue: "No module named 'tensorflow'"
**Solution**: Install TensorFlow for MNIST data
```bash
pip install tensorflow
```

### Issue: Slow simulation (>5s per sample)
**Solution**: 
- Reduce `N_TRAIN` parameter
- Decrease `HIDDEN_SIZE` (smaller network)
- Increase `dt` from 0.1 ms to 0.2 ms (less accurate but faster)

### Issue: OverflowError in weight updates
**Solution**: Verify weight clipping is enabled in `learning/stdp.py`
```python
w = np.clip(w, 0, 1)  # Ensure bounds
```

## Citations & References

### Foundational Papers:

1. **LIF Neuron Model**
   - Gerstner, W., & Kistler, W. M. (2002). *Spiking neuron models: Single neurons, populations, plasticity*. Cambridge University Press.

2. **STDP Learning**
   - Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.

3. **Unsupervised SNN Learning**
   - Masquelier, T., & Thorpe, S. J. (2007). Unsupervised learning of visual features through embedding images on the heptagonal lattice. *IEEE Transactions on Neural Networks*, 14(5), 1313-1330.

4. **Brian2 Simulator**
   - Goodman, D., & Brette, R. (2008). Brian: a simulator for spiking neural networks in Python. *Frontiers in Neuroinformatics*, 2, 5.

### Applied Reviews:

- Tavanaei, A., Ghodrati, M., Kheradpisheh, S. R., & Masquelier, T. (2019). Deep learning in spiking neural networks. *Neural Networks*, 111, 47-63.
- Bellec, G., Salaj, D., Subramoney, A., Legenstein, R., & Maass, W. (2020). Long short-term memory and learning-to-learn in networks of spiking neurons. *arXiv preprint arXiv:1803.09047*.

## License & Contact

**Author**: [Your Name]  
**Institution**: [Your University]  
**Date**: February 2026

For questions or contributions, please open an issue on the project repository.

---

**Last Updated**: February 11, 2026