#!/usr/bin/env python3
"""
HH Cell Firing Rate Modulation Example

This script creates a single-compartment Hodgkin–Huxley (HH) neuron that exhibits
a baseline firing rate. Then, an additional current injection from 100 ms to 300 ms
increases the firing rate. The voltage trace and detected spike times are plotted.

Adjust the IClamp amplitudes as needed.
"""

from neuron import h, gui
import matplotlib.pyplot as plt

# Create a single compartment (soma)
soma = h.Section(name='soma')
soma.L = 20      # microns
soma.diam = 20   # microns
soma.Ra = 100    # axial resistance (ohm*cm)
soma.cm = 1      # membrane capacitance (uF/cm^2)

# Insert the standard Hodgkin-Huxley channels
soma.insert("hh")

# Optionally, you can tweak intrinsic parameters here if needed.
# For example, if you need a more excitable cell, you might adjust:
# soma(0.5).hh.gnabar = 0.12  # default is around 0.12 (S/cm^2)
# soma(0.5).hh.gkbar = 0.036  # default is around 0.036
# soma(0.5).hh.gl = 0.0003    # default is around 0.0003
# soma(0.5).hh.el = -54.3     # leak reversal potential

# Set up recording variables
v_vec = h.Vector()   # Voltage record vector
t_vec = h.Vector()   # Time record vector
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

# Set up spike detection (optional)
spike_times = h.Vector()
nc = h.NetCon(soma(0.5)._ref_v, None)
nc.threshold = 0     # threshold set to 0 mV for spike detection
nc.record(spike_times)

# Create a baseline current injection that is present throughout the simulation.
# This current will drive a modest firing rate.
baseline_stim = h.IClamp(soma(0.5))
baseline_stim.delay = 0
baseline_stim.dur = 400  # entire simulation duration (ms)
baseline_stim.amp = 0.1  # nA; adjust to produce baseline spiking

# Create an additional current injection (extra drive) that is activated from 100 ms to 300 ms.
extra_stim = h.IClamp(soma(0.5))
extra_stim.delay = 100   # start extra injection at 100 ms
extra_stim.dur = 200     # extra injection lasts 200 ms (until 300 ms)
extra_stim.amp = 0.5     # nA; a higher amplitude to increase firing rate during the injection

# Set the simulation time
h.tstop = 400  # ms

# Run the simulation
h.run()

# Print out spike count and spike times
print("Total spikes detected:", len(spike_times))
print("Spike times (ms):", list(spike_times))

# Plot the voltage trace
plt.figure(figsize=(10, 5))
plt.plot(t_vec, v_vec, label='Soma voltage (mV)')
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("HH Neuron: Baseline Firing and Increased Firing with Extra Current")
plt.axvspan(extra_stim.delay, extra_stim.delay+extra_stim.dur, color='yellow', alpha=0.3, label='Extra injection window')
plt.legend()
plt.show()
