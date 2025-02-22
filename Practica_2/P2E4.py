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


def process_pgm_file3x3(im):
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


def process_pgm_file5x5(im):
    imout=im.copy()
    mask=np.ones((5,5))*1/(5*5)

    for i in range(len(imout)):
        for j in range(len(imout[i])):
            vecinos=[]
            for a in range(-2,2+1):
                for b in range(-2,2+1):
                    ny=i+a
                    nx=j+b
                    if 0<=ny<im.shape[0] and 0<=nx<im.shape[1]:
                        vecinos.append(im[ny,nx]*mask[a][b])
            imout[i,j]=int(sum(vecinos))
    return imout

def process_pgm_file7x7(im):
    imout=im.copy()
    mask=np.ones((7,7))*1/(7*7)

    for i in range(len(imout)):
        for j in range(len(imout[i])):
            vecinos=[]
            for a in range(-3,3+1):
                for b in range(-3,3+1):
                    ny=i+a
                    nx=j+b
                    if 0<=ny<im.shape[0] and 0<=nx<im.shape[1]:
                        vecinos.append(im[ny,nx]*mask[a][b])
            imout[i,j]=int(sum(vecinos))
    return imout


def imagenes_superpuestas4(im, im2, im3, im4):
    fig, axes = plt.subplots(1, 4, figsize=(26, 4), constrained_layout=True)

    vmin = min(np.amin(im), np.amin(im2), np.amin(im3), np.amin(im4))
    vmax = max(np.max(im), np.max(im2), np.max(im3), np.max(im4))

    # Mostrar imágenes con títulos
    imshow1 = axes[0].imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Imagen Original", fontsize=16)

    axes[1].imshow(im2, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("FPB con mascara 3x3", fontsize=16)

    axes[2].imshow(im3, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title("FPB con mascara 5x5", fontsize=16)

    axes[3].imshow(im4, cmap='gray', vmin=vmin, vmax=vmax)
    axes[3].set_title("FPB con mascara 7x7", fontsize=16)

    # Quitar ejes para mejor presentación
    for ax in axes:
        ax.axis("off")

    # Agregar una sola barra de color a la derecha de todas las imágenes
    cbar = fig.colorbar(imshow1, ax=axes, location="right", shrink=0.9)

    plt.show()



if __name__ == "__main__":
    
    if(len(sys.argv)<2):
        print("Usage: python P2E4.py [infile.pgm]")
        exit(1)

    infile = sys.argv[1]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    imout = process_pgm_file3x3(im)
    print("Size of image: {}".format(imout.shape))

    imout2 = process_pgm_file5x5(im)
    print("Size of image: {}".format(imout2.shape))

    imout3 = process_pgm_file7x7(im)
    print("Size of image: {}".format(imout3.shape))

    imagenes_superpuestas4(im, imout, imout2, imout3)   


