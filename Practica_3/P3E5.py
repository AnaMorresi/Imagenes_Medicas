#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:23:36 2023
Transformada de Radon y su inversa con scikit-image

@author: matog
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image,circle=True, theta=theta, preserve_range=True)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')

fig.tight_layout()
plt.show()

# agregrego ruido al sinograma:
media=0
std=3
ruido = np.random.normal(media,std,sinogram.shape)
sinogram_ruido = sinogram + ruido
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Sinograma Original")
ax1.set_xlabel("Projection angle (deg)")
ax1.set_ylabel("Projection position (pixels)")
ax1.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')
ax2.set_title("Sinograma con Ruido Gausiano")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram_ruido, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')
fig.tight_layout()
plt.show()

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

#estudiamos efecto filtros diferentes para reconstruccion

recontruccion_list=[image]
filtros_list=  ['ramp','shepp-logan','cosine','hamming','hann']
for filtro in filtros_list:
    reconstruction_fbp = iradon(sinogram_ruido, theta=theta, filter_name=filtro)
    recontruccion_list.append(reconstruction_fbp)
filtros_list=['Imagen Original']+filtros_list
imagenes_superpuestas_2x3(recontruccion_list, filtros_list)
    






