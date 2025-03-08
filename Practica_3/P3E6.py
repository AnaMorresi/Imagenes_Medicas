import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.transform import radon, iradon

# Crear una imagen en blanco (tamaño 100x100) con un círculo en el centro
size = 100
image = np.zeros((size, size))
rr, cc = disk((size // 2, size // 2), 5/2)  # Círculo con diámetro de 5 píxeles
image[rr, cc] = 1  # Asignar valor 1 al círculo

# Definir los diferentes números de proyecciones
num_projections_list = [8, 16, 32, 64]

fig, axes = plt.subplots(len(num_projections_list)//2, 4, figsize=(10, 5))

for i, num_projections in enumerate(num_projections_list):
    theta = np.linspace(0., 180., num_projections, endpoint=False)  # Ángulos de proyección
    
    # Proyección de Radón (sin filtrar)
    sinogram = radon(image, theta=theta, circle=True)
    # Reconstrucción por retroproyección sin filtro
    reconstruction_no_filter = iradon(sinogram, theta=theta, filter_name=None, circle=True)
    # Reconstrucción con filtro rampa
    reconstruction_filtered = iradon(sinogram, theta=theta, filter_name="ramp", circle=True)
    
    # Graficar resultados
    axes[i//2, (i%2)*2].imshow(reconstruction_no_filter, cmap='gray')
    axes[i//2, (i%2)*2].set_title(f"{num_projections} proyecciones\nsin filtro")
    
    axes[i//2, (i%2)*2+1].imshow(reconstruction_filtered, cmap='gray')
    axes[i//2, (i%2)*2+1].set_title(f"{num_projections} proyecciones\ncon filtro rampa")
    
# Ajustar diseño
plt.tight_layout()
plt.show()
