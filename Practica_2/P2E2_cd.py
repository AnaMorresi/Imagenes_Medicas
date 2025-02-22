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

def process_pgm_file_t1(im):
    imout = im.copy()
    #Transformacion Binaria
    def f(x):
        if 0<x<128:
            return 255 
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
        return int(c*x**gamma)
                
    for i in range(len(imout)):
        for j in range(len(imout[i])):
            imout[i][j]=f(imout[i][j])
    
    return imout

def process_pgm_file_susbtraccion(im,im2):
    imout = np.zeros(im.shape)

    for i in range(len(im)):
        for j in range(len(im[i])):
            imout[i][j]=int( (im[i][j]-im2[i][j])*1/2+255/2 )
    return imout

if __name__ == "__main__":
    
    if(len(sys.argv)<4):
        print("Usage: python P2E2_cd.py [infile.pgm] [outfile_transformacion.pgm] [outfile_sustraccion.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    outfile2 = sys.argv[3]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    imout = process_pgm_file_t1(im)
    im2 = np.array(imout)
    print("Size of image: {}".format(im2.shape))

    imout2 = process_pgm_file_susbtraccion(im,im2)
    im3 = np.array(imout2)
    print("Size of image: {}".format(im3.shape))

    imagenes_superpuestas(im, im2, im3)

    cv2.imwrite(outfile,imout,[cv2.IMWRITE_PXM_BINARY,0])
    cv2.imwrite(outfile2,imout2,[cv2.IMWRITE_PXM_BINARY,0])