import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

# Load the image
image_path = 'original.jpeg'  # Replace with your image path
image = Image.open(image_path)

# Convert image to numpy array
image_array = np.array(image)

# Ensure the image is in RGB format
if image_array.shape[2] == 4:  # If image has an alpha channel
    image_array = image_array[:, :, :3]

# Extract pixel values for each color channel
red_values = image_array[:, :, 0].flatten()
green_values = image_array[:, :, 1].flatten()
blue_values = image_array[:, :, 2].flatten()

# Create the main plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histograms for each color channel
ax.hist(red_values, bins=256, density=True, alpha=0.7, color='r', edgecolor='black', label='Red')
ax.hist(green_values, bins=256, density=True, alpha=0.7, color='g', edgecolor='black', label='Green')
ax.hist(blue_values, bins=256, density=True, alpha=0.7, color='b', edgecolor='black', label='Blue')

# Add the title
ax.text(0.5, 1.05, 'original', transform=ax.transAxes, fontsize=32, ha='center')

# Add the inset image
ax_inset = inset_axes(ax, width="20%", height="20%", loc='upper right')
ax_inset.imshow(image, aspect='auto')
ax_inset.axis('off')  # Hide the axes for the inset image

# Customize axes
ax.set_xlim([0, 256])
ax.set_xlabel('Pixel Value')
ax.set_ylabel('Frequency')
ax.legend(loc='upper left')

# Show the plot
plt.show()
