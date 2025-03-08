import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# Para 180 grados

image = shepp_logan_phantom()
theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

# Simular falla de un detector
sinogram_defectuoso = sinogram.copy()
sinogram_defectuoso[250,:] = 0  # Simulamos un detector defectuoso en la columna 50

# Reconstrucción con y sin defecto
reconstruction = iradon(sinogram, theta=theta, filter_name='ramp')
reconstruction_defectuosa = iradon(sinogram_defectuoso, theta=theta, filter_name='ramp')

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
dx, dy = 0.5 * 360.0 / max(image.shape), 0.5 / sinogram.shape[0]

axes[0].imshow(sinogram_defectuoso, cmap='gray', extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy), aspect='auto')
axes[0].set_title("Sinograma con Falla")
axes[0].set_xlabel("Projection angle (deg)")
axes[0].set_ylabel("Projection position (pixels)")

axes[1].imshow(reconstruction_defectuosa, cmap='gray')
axes[1].set_title("Reconstrucción con Falla")

plt.tight_layout()
plt.show()

# Para 360 grados

image = shepp_logan_phantom()
theta = np.linspace(0., 360., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

# Simular falla de un detector
sinogram_defectuoso = sinogram.copy()
sinogram_defectuoso[250,:] = 0  # Simulamos un detector defectuoso en la columna 50

# Reconstrucción con y sin defecto
reconstruction = iradon(sinogram, theta=theta, filter_name='ramp')
reconstruction_defectuosa = iradon(sinogram_defectuoso, theta=theta, filter_name='ramp')

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
dx, dy = 0.5 * 360.0 / max(image.shape), 0.5 / sinogram.shape[0]

axes[0].imshow(sinogram_defectuoso, cmap='gray', extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy), aspect='auto')
axes[0].set_title("Sinograma con Falla")
axes[0].set_xlabel("Projection angle (deg)")
axes[0].set_ylabel("Projection position (pixels)")

axes[1].imshow(reconstruction_defectuosa, cmap='gray')
axes[1].set_title("Reconstrucción con Falla")

plt.tight_layout()
plt.show()

# Para 360 grados, otro radio y mas fallas

image = shepp_logan_phantom()
theta = np.linspace(0., 360., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

# Simular falla de un detector
sinogram_defectuoso = sinogram.copy()
sinogram_defectuoso[300:305,:] = 0  # Simulamos un detector defectuoso en la columna 50

# Reconstrucción con y sin defecto
reconstruction = iradon(sinogram, theta=theta, filter_name='ramp')
reconstruction_defectuosa = iradon(sinogram_defectuoso, theta=theta, filter_name='ramp')

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
dx, dy = 0.5 * 360.0 / max(image.shape), 0.5 / sinogram.shape[0]

axes[0].imshow(sinogram_defectuoso, cmap='gray', extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy), aspect='auto')
axes[0].set_title("Sinograma con Falla")
axes[0].set_xlabel("Projection angle (deg)")
axes[0].set_ylabel("Projection position (pixels)")

axes[1].imshow(reconstruction_defectuosa, cmap='gray')
axes[1].set_title("Reconstrucción con Falla")

plt.tight_layout()
plt.show()