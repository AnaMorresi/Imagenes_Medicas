#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, resize

from skimage.transform import iradon

image = shepp_logan_phantom()
fig, axes = plt.subplots(2, 4, figsize=(10, 6))

# Variacion cantidad detectores
for i, size in enumerate([100, 200, 300, 400, 500, 600, 700, 800]):
    image = resize(image, output_shape=(size,size), mode='reflect')     # resize vecimos mas cercanos
    axes[i//4,i%4].set_title(f"Cantidad de detectores:\n{size}")
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image,circle=True, theta=theta)
    print(sinogram.shape)
  
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
    error = reconstruction_fbp - image
    print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

    axes[i//4,i%4].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

plt.show()

error_vector=[]
for i in np.arange(100,1001,50):
    image = resize(image, output_shape=(i,i), mode='reflect')     # resize vecimos mas cercanos
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image,circle=True, theta=theta)
  
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
    error = np.sqrt(np.mean(((reconstruction_fbp - image)**2)))
    error_vector.append(error)

plt.plot(np.arange(100,1001,50),error_vector,'o-')
plt.xlabel('Cantidad de detectores')
plt.ylabel('RECM de reconstrucción')
plt.grid()
plt.show()

# Variacion tipo de filtrado

def imagenes_superpuestas_2x3(im_list, row_titles):
    """
    Muestra 3 filas de imágenes con 2 imágenes por fila y títulos para cada fila.
    
    Parámetros:
    - im_list: Lista de 6 imágenes organizadas en filas de 2.
    - row_titles: Lista de 3 títulos, uno por fila.
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

    # Obtener valores mínimo y máximo para la escala uniforme
    vmin = min(np.amin(im) for im in im_list)
    vmax = max(np.max(im) for im in im_list)
    
    def f(x):
        return x*(vmax-vmin)/vmax+vmin
    
    im_list_aux=[]
    for image in im_list:
        image_aux=[f(i) for i in image]
        im_list_aux.append(image_aux)

    # Iterar sobre las filas y columnas para mostrar las imágenes
    for i in range(2):  # Filas
        for j in range(3):  # Columnas
            idx = i * 3 + j  # Índice en la lista de imágenes
            axes[i, j].imshow(im_list_aux[idx], cmap=plt.cm.Greys_r, vmin=vmin, vmax=vmax)
            axes[i, j].axis("off")  # Quitar ejes

            # Agregar título centrado para la fila
            if idx==0:
                axes[i, j].set_title(row_titles[idx], fontsize=16)
            else:
                axes[i, j].set_title('Filtro '+row_titles[idx], fontsize=16)
        
    plt.show()

image = shepp_logan_phantom()
theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image,circle=True, theta=theta, preserve_range=True)

recontruccion_list=[]
error_filtrado_vector=[]
filtros_list=  ['None','ramp','shepp-logan','cosine','hamming','hann']
cont=0
for filtro in filtros_list:
    if cont==0:
        reconstruction_fbp = iradon(sinogram, theta=theta, filter_name=None)
    else:
        reconstruction_fbp = iradon(sinogram, theta=theta, filter_name=filtro)
    recontruccion_list.append(reconstruction_fbp)
    error = reconstruction_fbp - image
    error_rms = np.sqrt(np.mean(error ** 2))
    print(error_rms)
    error_filtrado_vector.append(error_rms)
    cont=cont+1
imagenes_superpuestas_2x3(recontruccion_list, filtros_list)

x_pos=range(len(filtros_list))
plt.scatter(x_pos[1:],error_filtrado_vector[1:],color='red', marker='o')
plt.xticks(x_pos[1:], filtros_list[1:])
#plt.xlabel('Tipo de filtrado')
plt.ylabel('RECM de recontruccion')
plt.grid()
plt.show()
