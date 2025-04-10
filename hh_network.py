from neuron import h, gui
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

# Load the trained ANN model weights
def load_ann_weights(model_path, input_size=400, hidden_size=16):
    """Load weights from the trained ANN model"""
    # Create a temporary model to load weights
    class GratingNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, 2)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = GratingNN(input_size, hidden_size)
    model.load_state_dict(torch.load(model_path))
    
    # Extract weights and biases
    weights = {
        'input_to_hidden': model.fc1.weight.detach().numpy(),
        'input_to_hidden_bias': model.fc1.bias.detach().numpy(),
        'hidden_to_output': model.fc2.weight.detach().numpy(),
        'hidden_to_output_bias': model.fc2.bias.detach().numpy()
    }
    
    return weights

# Scale ANN weights to biophysical parameters
def scale_weights(weights, input_scaling=0.01, hidden_scaling=0.05):
    """Scale ANN weights to appropriate magnitudes for synaptic conductances (μS)"""
    scaled_weights = {
        'input_to_hidden': weights['input_to_hidden'] * input_scaling,
        'input_to_hidden_bias': weights['input_to_hidden_bias'] * input_scaling,
        'hidden_to_output': weights['hidden_to_output'] * hidden_scaling,
        'hidden_to_output_bias': weights['hidden_to_output_bias'] * hidden_scaling
    }
    return scaled_weights

# Define HH neuron class
class HHNeuron:
    def __init__(self, bias_current=0.0):
        # Create a new section (compartment)
        self.soma = h.Section(name='soma')
        
        # Set geometry
        self.soma.L = 20    # μm
        self.soma.diam = 20 # μm
        
        # Insert HH channels
        self.soma.insert('hh')
        
        # Set up bias current for baseline activity
        self.stim = h.IClamp(self.soma(0.5))
        self.stim.delay = 0
        self.stim.dur = 1e9  # Very long duration
        self.stim.amp = bias_current  # nA
        
        # Recording vectors
        self.v_vec = h.Vector()
        self.t_vec = h.Vector()
        self.v_vec.record(self.soma(0.5)._ref_v)
        self.t_vec.record(h._ref_t)
        
        # Spike counter
        self.nc = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.nc.threshold = 0  # mV
        self.spike_times = h.Vector()
        self.nc.record(self.spike_times)

class RetinaNeuron(HHNeuron):
    def __init__(self, bias_current=0.1):
        super().__init__(bias_current)
        
        # Add additional current clamp for stimulus-driven activity
        self.stim_current = h.IClamp(self.soma(0.5))
        self.stim_current.delay = 0
        self.stim_current.dur = 1e9  # Continuous
        self.stim_current.amp = 0  # Will be set based on pixel value

# Define the HH network class
class HHNetwork:
    def __init__(self, input_size=400, hidden_size=16, output_size=2, 
                 scaled_weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scaled_weights = scaled_weights
        
        # Time parameters
        self.dt = 0.025      # ms
        self.sim_time = 500  # ms
        
        # Create neurons for each layer
        self.setup_network()
        
    def setup_network(self):
        """Set up the network architecture with all neurons and synapses"""
        # Create retina layer neurons (with baseline current)
        self.retina_neurons = [RetinaNeuron(bias_current=0.1) for _ in range(self.input_size)]
        
        # Create hidden layer neurons (with bias from ANN)
        self.hidden_neurons = []
        for i in range(self.hidden_size):
            bias_current = self.scaled_weights['input_to_hidden_bias'][i]
            bias_current = max(bias_current, 0.05)  
            self.hidden_neurons.append(HHNeuron(bias_current=bias_current))
        
        # Create output layer neurons
        self.output_neurons = []
        for i in range(self.output_size):
            bias_current = self.scaled_weights['hidden_to_output_bias'][i]
            bias_current = max(bias_current, 0.05)
            self.output_neurons.append(HHNeuron(bias_current=bias_current))
        
        # Create synaptic connections (retina->hidden)
        self.r2h_connections = []
        for i, retina_neuron in enumerate(self.retina_neurons):
            for j, hidden_neuron in enumerate(self.hidden_neurons):
                weight = self.scaled_weights['input_to_hidden'][j, i]
                
                if weight > 0:
                    syn = h.ExpSyn(hidden_neuron.soma(0.5))
                    syn.tau = 2  # ms - fast AMPA-like synapse
                    syn.e = 0    # mV - excitatory reversal
                    nc = h.NetCon(retina_neuron.soma(0.5)._ref_v, syn, sec=retina_neuron.soma)
                    nc.threshold = 0
                    nc.delay = 1
                    nc.weight[0] = abs(weight)
                    self.r2h_connections.append((syn, nc))
                elif weight < 0:
                    syn = h.ExpSyn(hidden_neuron.soma(0.5))
                    syn.tau = 5  # ms - slower GABA-like synapse
                    syn.e = -80  # mV - inhibitory reversal
                    nc = h.NetCon(retina_neuron.soma(0.5)._ref_v, syn, sec=retina_neuron.soma)
                    nc.threshold = 0
                    nc.delay = 1
                    nc.weight[0] = abs(weight)
                    self.r2h_connections.append((syn, nc))
        
        # Create synaptic connections (hidden->output)
        self.h2o_connections = []
        for i, hidden_neuron in enumerate(self.hidden_neurons):
            for j, output_neuron in enumerate(self.output_neurons):
                weight = self.scaled_weights['hidden_to_output'][j, i]
                
                if weight > 0:
                    syn = h.ExpSyn(output_neuron.soma(0.5))
                    syn.tau = 2
                    syn.e = 0
                    nc = h.NetCon(hidden_neuron.soma(0.5)._ref_v, syn, sec=hidden_neuron.soma)
                    nc.threshold = 0
                    nc.delay = 1
                    nc.weight[0] = abs(weight)
                    self.h2o_connections.append((syn, nc))
                elif weight < 0:
                    syn = h.ExpSyn(output_neuron.soma(0.5))
                    syn.tau = 5
                    syn.e = -80
                    nc = h.NetCon(hidden_neuron.soma(0.5)._ref_v, syn, sec=hidden_neuron.soma)
                    nc.threshold = 0
                    nc.delay = 1
                    nc.weight[0] = abs(weight)
                    self.h2o_connections.append((syn, nc))
    
    def set_stimulus(self, image_array):
        """Set stimulus current for retina neurons based on the image pixel intensities"""
        flat_image = image_array.flatten()
        for i, pixel_value in enumerate(flat_image):
            normalized_value = 1.0 - (pixel_value / 255.0)
            current_magnitude = normalized_value * 0.5  # nA scaling
            self.retina_neurons[i].stim_current.amp = current_magnitude
    
    def run_simulation(self):
        """Run the NEURON simulation"""
        h.dt = self.dt
        h.tstop = self.sim_time
        h.finitialize(-65)
        h.continuerun(self.sim_time)
    
    def get_output_decision(self):
        """Decide orientation based on spike counts in output neurons"""
        # Convention: output_neurons[1] = Horizontal; output_neurons[0] = Vertical
        horizontal_spikes = len(self.output_neurons[1].spike_times)
        vertical_spikes = len(self.output_neurons[0].spike_times)
        if horizontal_spikes > vertical_spikes:
            return "Horizontal", horizontal_spikes, vertical_spikes
        else:
            return "Vertical", horizontal_spikes, vertical_spikes
    
    def plot_results(self, image_array=None, decision=None):
        """Plot simulation results"""
        plt.figure(figsize=(15, 10))
        if image_array is not None:
            plt.subplot(3, 4, 1)
            plt.imshow(image_array, cmap='gray')
            if decision:
                plt.title(f"Input Image:\nPredicted {decision[0]}")
            else:
                plt.title("Input Image")
            plt.axis('off')
        
        layers = [
            ("Retina", self.retina_neurons, 2),
            ("Hidden", self.hidden_neurons, 2),
            ("Output", self.output_neurons, 2)
        ]
        
        plot_idx = 2
        for layer_name, neurons, num_samples in layers:
            sample_indices = np.linspace(0, len(neurons)-1, num_samples, dtype=int)
            for i, idx in enumerate(sample_indices):
                neuron = neurons[idx]
                t = np.array(neuron.t_vec)
                v = np.array(neuron.v_vec)
                plt.subplot(3, 4, plot_idx)
                plt.plot(t, v)
                title = f"{layer_name} Neuron {idx+1}"
                if layer_name == "Output":
                    title += f"\nSpikes: {len(neuron.spike_times)}"
                plt.title(title)
                plt.xlabel('Time (ms)')
                plt.ylabel('Vm (mV)')
                plt.ylim(-80, 50)
                plot_idx += 1
        plt.tight_layout()
        return plt.gcf()

def load_and_preprocess_image(image_path, target_size=(20, 20)):
    """Load a grayscale image and resize it"""
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

def main():
    # Parameters
    model_path = "grating_model_weights.pth"
    image_path = "testing_images/horizontal_90.png"  # Example test image
    input_size = 20 * 20  # 400 pixels
    hidden_size = 16
    
    # Load ANN weights
    weights = load_ann_weights(model_path, input_size, hidden_size)
    
    # Scale weights for biophysical model
    scaled_weights = scale_weights(weights, input_scaling=0.01, hidden_scaling=0.05)
    
    # Load and preprocess test image
    img_array = load_and_preprocess_image(image_path)
    
    # Create HH network
    network = HHNetwork(input_size=input_size, hidden_size=hidden_size, 
                        scaled_weights=scaled_weights)
    
    # Set the stimulus in the retina based on the image
    network.set_stimulus(img_array)
    
    # Run simulation
    network.run_simulation()
    
    # Get decision based on output spikes
    decision = network.get_output_decision()
    print(f"Network decision: {decision[0]}")
    print(f"Spike counts - Horizontal: {decision[1]}, Vertical: {decision[2]}")
    
    # Plot results
    fig = network.plot_results(img_array, decision)
    plt.show()

def test_all_images():
    """Run the HH network on all testing images and calculate accuracy"""
    import matplotlib.pyplot as plt
    
    model_path = "grating_model_weights.pth"
    test_dir = "testing_images"
    input_size = 20 * 20  
    hidden_size = 16
    
    weights = load_ann_weights(model_path, input_size, hidden_size)
    scaled_weights = scale_weights(weights, input_scaling=0.01, hidden_scaling=0.05)
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    print(f"Testing HH network on {len(image_files)} images...")
    
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, filename in enumerate(sorted(image_files)):
        if i >= 25:
            break
        true_orientation = "Horizontal" if filename.startswith("horizontal") else "Vertical"
        angle = filename.split('_')[-1].split('.')[0]
        image_path = os.path.join(test_dir, filename)
        img_array = load_and_preprocess_image(image_path)
        
        network = HHNetwork(input_size=input_size, hidden_size=hidden_size, 
                            scaled_weights=scaled_weights)
        network.set_stimulus(img_array)
        network.run_simulation()
        decision, h_spikes, v_spikes = network.get_output_decision()
        is_correct = decision == true_orientation
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        result = {
            'filename': filename,
            'true_orientation': true_orientation,
            'predicted_orientation': decision,
            'horizontal_spikes': h_spikes,
            'vertical_spikes': v_spikes,
            'correct': is_correct,
            'angle': angle
        }
        results.append(result)
        if i < 25:
            ax = axes[i]
            ax.imshow(img_array, cmap='gray')
            color = 'green' if is_correct else 'red'
            title = f"{angle}°: {decision}\n(H:{h_spikes} vs V:{v_spikes})"
            ax.set_title(title, color=color)
            ax.axis('off')
    
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    angle_results = {}
    for result in results:
        angle = result['angle']
        if angle not in angle_results:
            angle_results[angle] = {'correct': 0, 'total': 0}
        angle_results[angle]['total'] += 1
        if result['correct']:
            angle_results[angle]['correct'] += 1
    
    print("\nAccuracy by Angle:")
    for angle, counts in sorted(angle_results.items(), key=lambda x: int(x[0])):
        acc = (counts['correct'] / counts['total']) * 100 if counts['total'] > 0 else 0
        print(f"  Angle {angle}°: {acc:.2f}% ({counts['correct']}/{counts['total']})")
    
    print("\nIndividual Image Results:")
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"  {status} {result['filename']} -> Predicted: {result['predicted_orientation']} "
              f"(Actual: {result['true_orientation']}) "
              f"[H: {result['horizontal_spikes']}, V: {result['vertical_spikes']}]")
    
    plt.tight_layout()
    plt.show()
    return results, accuracy

if __name__ == "__main__":
    test_all_images()