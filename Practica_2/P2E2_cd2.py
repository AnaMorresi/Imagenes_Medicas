import cv2
import sys 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path

path=os.getcwd()
print(path)

def read_pgm_file(file_name):

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Test if file exists
    file_path = os.path.join(data_dir, file_name)
    assert os.path.isfile(file_path), 'file \'{0}\' does not exist'.format(file_path)

    img = cv2.imread(file_name,flags=cv2.IMREAD_GRAYSCALE)

    if img is not None:
        print('img.size: ', img.size)
    else:
        print('imread({0}) -> None'.format(file_path))

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

def hist_superpuestos(im, im2):
    vmin = np.amin(im)
    vmax = np.max(im)
    L1 = vmax - vmin
    vmin = np.amin(im2)
    vmax = np.max(im2)
    L2 = vmax - vmin

    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(1, 1, 1)

    hist1, bin_edges1 = np.histogram(im.ravel(), bins=L1, range=(0, 256))
    hist2, bin_edges2 = np.histogram(im2.ravel(), bins=L2, range=(0, 256))
    ax.bar(bin_edges1[:-1], hist1, alpha=0.5, label='Original')
    ax.bar(bin_edges2[:-1], hist2, alpha=0.5, label='Ecualizada')    
    ax.legend(loc='upper right')
    plt.show()

def imagenes_superpuestas(im, im2, im3):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)

    vmin = min(np.amin(im), np.amin(im2), np.amin(im3))
    vmax = max(np.max(im), np.max(im2), np.max(im3))

    # Mostrar imágenes
    imshow1 = axes[0].imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].imshow(im2, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].imshow(im3, cmap='gray', vmin=vmin, vmax=vmax)

    # Quitar ejes para mejor presentación
    for ax in axes:
        ax.axis("off")

    # Agregar subtítulos debajo de cada imagen
    subtitulos = ["(a)", "(b)", "(c)"]
    for i, ax in enumerate(axes):
        ax.text(0.5, -0.15, subtitulos[i], fontsize=16, ha="center", va="top", transform=ax.transAxes)

    # Agregar una sola barra de color a la derecha de todas las imágenes
    cbar = fig.colorbar(imshow1, ax=axes, location="right", shrink=0.9)

    plt.show()


def imagenes_superpuestas_3x2(im_list, row_titles):
    """
    Muestra 3 filas de imágenes con 2 imágenes por fila y títulos para cada fila.
    
    Parámetros:
    - im_list: Lista de 6 imágenes organizadas en filas de 2.
    - row_titles: Lista de 3 títulos, uno por fila.
    """
    fig, axes = plt.subplots(3, 2, figsize=(5, 12), constrained_layout=True)

    # Obtener valores mínimo y máximo para la escala uniforme
    vmin = min(np.amin(im) for im in im_list)
    vmax = max(np.max(im) for im in im_list)

    # Iterar sobre las filas y columnas para mostrar las imágenes
    for i in range(3):  # Filas
        for j in range(2):  # Columnas
            idx = i * 2 + j  # Índice en la lista de imágenes
            axes[i, j].imshow(im_list[idx], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, j].axis("off")  # Quitar ejes

        # Agregar título centrado para la fila
        axes[i, 0].set_title(row_titles[i], fontsize=16, loc='left')

    # Agregar barra de color a la derecha de todas las imágenes
    cbar = fig.colorbar(axes[0, 0].imshow(im_list[0], cmap='gray', vmin=vmin, vmax=vmax), 
                         ax=axes, location="right", shrink=0.5)

    plt.show()

def process_pgm_file_t1(im):
    imout = im.copy()
    #Transformacion Binaria
    def f(x):
        if 0<x<128:
            return 255 #consultar si es 255 o 1
        else:
            return 0         

    for i in range(len(imout)):
        for j in range(len(imout[i])):
            imout[i][j]=f(imout[i][j])
    
    return imout

def process_pgm_file_t23(im,gamma):
    imout = im.copy()
    #Transformacion
    def f(x,gamma):
        c=255/(255**gamma)
        return c*x**gamma
                
    for i in range(len(imout)):
        for j in range(len(imout[i])):
            imout[i][j]=f(imout[i][j],gamma)
    
    return imout

def process_pgm_file_susbtraccion(im,im2):
    imout = np.zeros(im.shape)

    for i in range(len(im)):
        for j in range(len(im[i])):
            imout[i][j]=int( (im[i][j]-im2[i][j])*1/2+255/2 )
    return imout

if __name__ == "__main__":
    
    if(len(sys.argv)<2):
        print("Usage: python P2E2_cd2.py [infile.pgm]")
        exit(1)

    infile = sys.argv[1]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    im_list=[]
    for i in [0.1,0.3,0.5]:
        imout = process_pgm_file_t23(im,i)
        im2 = np.array(imout)

        imout2 = process_pgm_file_susbtraccion(im,im2)
        im3 = np.array(imout2)

        im_list.append(im2)
        im_list.append(im3)
        #imagenes_superpuestas(im, im2, im3)

    imagenes_superpuestas_3x2(im_list, ["γ=0.1", "γ=0.3", "γ=0.5"])

    im_list=[]
    for i in [1.5,3,5]:
        imout = process_pgm_file_t23(im,i)
        im2 = np.array(imout)

        imout2 = process_pgm_file_susbtraccion(im,im2)
        im3 = np.array(imout2)

        im_list.append(im2)
        im_list.append(im3)
        #imagenes_superpuestas(im, im2, im3)

    imagenes_superpuestas_3x2(im_list, ["γ=1.5", "γ=3", "γ=4"])