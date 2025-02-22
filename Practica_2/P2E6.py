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



def FPB_3x3(im):
    imout=im.copy()
    mask=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]

    for i in range(len(imout)):
        for j in range(len(imout[i])):
            vecinos=[]
            for a in [-1,0,1]:
                for b in [-1,0,1]:
                    ny=i+a
                    nx=j+b
                    if 0<=ny<im.shape[0] and 0<=nx<im.shape[1]:
                        vecinos.append(im[ny,nx]*mask[a][b])
            imout[i,j]=int(sum(vecinos))
    return imout


def Filtro_unsharp_3x3(im):
    #
    #   Escriba su procesamiento de imágenes aquí
    imout=im.copy()
    mask=[[0,-1,0],[-1,5,-1],[0,-1,0]]      #I+FPA

    for i in range(len(imout)):
        for j in range(len(imout[i])):
            vecinos=[]
            for a in [-1,0,1]:
                for b in [-1,0,1]:
                    ny=i+a
                    nx=j+b
                    if 0<=ny<im.shape[0] and 0<=nx<im.shape[1]:
                        vecinos.append(im[ny,nx]*mask[a][b])
            if sum(vecinos)<0:
                imout[i,j]=int(0)
            elif sum(vecinos)>255:
                imout[i,j]=int(255)
            else:
                imout[i,j]=int(sum(vecinos))
    return imout

def Filtro_HighBoost_3x3(im):
    #
    #   Escriba su procesamiento de imágenes aquí
    imout=im.copy()
    A=2
    mask=[[-1,-1,-1],[-1,8+A,-1],[-1,-1,-1]]  # A*I-FPB

    for i in range(len(imout)):
        for j in range(len(imout[i])):
            vecinos=[]
            for a in [-1,0,1]:
                for b in [-1,0,1]:
                    ny=i+a
                    nx=j+b
                    if 0<=ny<im.shape[0] and 0<=nx<im.shape[1]:
                        vecinos.append(im[ny,nx]*mask[a][b])
            if sum(vecinos)<0:
                imout[i,j]=int(0)
            elif sum(vecinos)>255:
                imout[i,j]=int(255)
            else:
                imout[i,j]=int(sum(vecinos))
    return imout

def imagenes_superpuestas(im, im2):

    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    imgplot1 = ax1.imshow(im, cmap='gray', vmin=np.amin(im), vmax=np.max(im))
    fig.colorbar(imgplot1, ax=ax1)
    imgplot2 = ax2.imshow(im2, cmap='gray', vmin=np.amin(im2), vmax=np.max(im2))
    fig.colorbar(imgplot2, ax=ax2)
    plt.show()


def imagenes_superpuestas4(im, im2, im3, im4):
    fig, axes = plt.subplots(1, 4, figsize=(26, 4), constrained_layout=True)

    vmin = min(np.amin(im), np.amin(im2), np.amin(im3), np.amin(im4))
    vmax = max(np.max(im), np.max(im2), np.max(im3), np.max(im4))

    # Mostrar imágenes con títulos
    imshow1 = axes[0].imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Imagen con ruido Gaussiano", fontsize=16)

    axes[1].imshow(im2, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("FPB", fontsize=16)

    axes[2].imshow(im3, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title("Unsharp", fontsize=16)

    axes[3].imshow(im4, cmap='gray', vmin=vmin, vmax=vmax)
    axes[3].set_title("High Boost", fontsize=16)

    # Quitar ejes para mejor presentación
    for ax in axes:
        ax.axis("off")

    # Agregar una sola barra de color a la derecha de todas las imágenes
    cbar = fig.colorbar(imshow1, ax=axes, location="right", shrink=0.9)

    plt.show()

def MSE(original, filtered):
    if original.shape != filtered.shape:
        raise ValueError("Las dimensiones de las imágenes deben coincidir.")
    
    mse = np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
    return mse


if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python P2E6.py [Imagen_con_ruido.pgm] [Imagen_Original.pgm]")
        exit(1)

    infile = sys.argv[1]
    infile2 = sys.argv[2]
    
    img = read_pgm_file(infile)
    img2 = read_pgm_file(infile2)

    im = np.array(img)
    im2 = np.array(img2)
    print("Size of image: {}".format(im.shape))

    imout = FPB_3x3(im)
    print(MSE(im2, imout))

    imout2 = Filtro_unsharp_3x3(im)
    print(MSE(im2, imout2))

    imout3 = Filtro_HighBoost_3x3(im)
    print(MSE(im2, imout3))

    imagenes_superpuestas4(im, imout, imout2, imout3)   