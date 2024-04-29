import matplotlib.pyplot as plt

# Assuming you have collected spike data for each layer
spike_data = [
    # Layer 1 spike times (example data)
    [0.1, 0.5, 0.6, 1.2, 1.3],
    # Layer 2 spike times
    [0.2, 0.3, 0.8],
    # Layer 3 spike times
    [0.4, 0.7, 1.0],
    # Layer 4 spike times
    [0.9, 1.1]
    # Add spike times for other layers as needed
]

# Plotting raster plot for each layer
plt.figure(figsize=(10, 6))
for layer_idx, spikes in enumerate(spike_data):
    plt.eventplot(spikes, lineoffsets=layer_idx + 1, color='black', linewidths=2)
plt.title('Spike Raster Plot')
plt.xlabel('Time')
plt.ylabel('Layer')
plt.yticks(range(1, len(spike_data) + 1), ['Layer {}'.format(i + 1) for i in range(len(spike_data))])
plt.grid(True)
plt.show()
