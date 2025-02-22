import cv2
import sys 
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

path=os.getcwd()
print(path)

def read_pgm_file(file_name):

    img = cv2.imread(file_name,flags=cv2.IMREAD_GRAYSCALE)

    if img is not None:
        print('img.size: ', img.size)

    return img


def show_img_hist(im):
    
    vmin = np.amin(im)
    vmax = np.max(im)
    print("Intensity Min: {}   Max:{}".format(vmin,vmax))

    L = vmax - vmin
    print("Number of Levels: {}".format(L))
    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # imgplot = plt.imshow(im/np.amax(im))
    imgplot = ax1.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=ax1)
    # cv2.imshow(infile,img)
    # cv2.waitKey(0)

    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    ax2.bar(bin_edges[:-1], hist)
    plt.savefig(path+'/histogram.png')
    plt.show()

def gaussian(x, mu, sigma, amplitude):
    """Definición de una función gaussiana."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian(x, y):
    """
    Ajusta una gaussiana a los datos x, y.
    
    Parámetros:
        x (array): Datos independientes.
        y (array): Datos dependientes (valores de función).
        
    Retorna:
        mu (float): Media de la gaussiana.
        sigma (float): Desvío estándar de la gaussiana.
        r_squared (float): Coeficiente de determinación R^2.
    """
    # Estimación inicial de parámetros: mu, sigma, amplitude
    initial_guess = [np.mean(x), np.std(x), max(y)]
    
    # Ajuste de la gaussiana a los datos
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=initial_guess)
    except RuntimeError:
        print("El ajuste falló. Verifica los datos.")
        return None, None, None
    
    mu, sigma, amplitude = popt

    # Predicciones con los parámetros ajustados
    y_fit = gaussian(x, mu, sigma, amplitude)
    
    # Cálculo del coeficiente de determinación R^2
    r_squared = r2_score(y, y_fit)
    
    return mu, sigma, r_squared


def region_hist(im1,im2,im3):
    L = 256
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 1, 1)

    hist1, bin_edges1 = np.histogram(im1.ravel(),bins=L)
    hist2, bin_edges2 = np.histogram(im2.ravel(),bins=L)
    hist3, bin_edges3 = np.histogram(im3.ravel(),bins=L)

    print(len(hist1))
    print(len(bin_edges1))
    hist1=hist1[20:101]
    bin_edges1=bin_edges1[20:101]
    hist2=hist2[20:101]
    bin_edges2=bin_edges2[20:101]
    hist3=hist3[20:101]
    bin_edges3=bin_edges3[20:101]

    ax.bar(bin_edges1, hist1, label="AAA0002")
    ax.bar(bin_edges2, hist2, label='AAA0003')
    ax.bar(bin_edges3, hist3, label='AAA0004')
    plt.legend()
    plt.xlabel('Escala de Grises')
    plt.ylabel('Cantidad de pixeles')
    plt.xlim(20,100)
    plt.savefig(r"C:\Users\anapa\Documents\IM\Practica_2\Histograma7.png")
    plt.show()

    # Ajustar la gaussiana
    mu1, sigma1, r_squared1 = fit_gaussian(bin_edges1, hist1)
    mu2, sigma2, r_squared2 = fit_gaussian(bin_edges2, hist2)
    mu3, sigma3, r_squared3 = fit_gaussian(bin_edges3, hist3)
    
    # Imprimir resultados
    print(f"Relacion Senial/Ruido: \n{mu1/sigma1}; \n{mu2/sigma2}; \n{mu3/sigma3}")
    print(f"Coeficiente de determinación (R^2): {r_squared1}; {r_squared2}; {r_squared3}")

def imagenes_superpuestas3(im, im2, im3):

    fig = plt.figure(figsize=(24,6))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    imgplot1 = ax1.imshow(im, cmap='gray', vmin=np.amin(im), vmax=np.max(im))
    fig.colorbar(imgplot1, ax=ax1)
    imgplot2 = ax2.imshow(im2, cmap='gray', vmin=np.amin(im2), vmax=np.max(im2))
    fig.colorbar(imgplot2, ax=ax2)
    imgplot3 = ax3.imshow(im3, cmap='gray', vmin=np.amin(im3), vmax=np.max(im3))
    fig.colorbar(imgplot3, ax=ax3)
    plt.show()

# MAIN

img1 = read_pgm_file(r"C:\Users\anapa\Documents\IM\Practica_2\AAA\AAA0002.pgm")
img2 = read_pgm_file(r"C:\Users\anapa\Documents\IM\Practica_2\AAA\AAA0003.pgm")
img3 = read_pgm_file(r"C:\Users\anapa\Documents\IM\Practica_2\AAA\AAA0004.pgm")
im1 = np.array(img1)
im2 = np.array(img2)
im3 = np.array(img3)

imagenes_superpuestas3(im1, im2, im3)

region_hist(im1,im2,im3)
