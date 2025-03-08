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
    plt.ylabel('Cantidad de pixeles')
    plt.xlabel('Escala de Grises')
    plt.grid()
    plt.show()

def imagenes_superpuestas(im, im2):

    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    imgplot1 = ax1.imshow(im, cmap='gray', vmin=np.amin(im), vmax=np.max(im))
    fig.colorbar(imgplot1, ax=ax1)
    imgplot2 = ax2.imshow(im2, cmap='gray', vmin=np.amin(im2), vmax=np.max(im2))
    fig.colorbar(imgplot2, ax=ax2)
    plt.show()


def process_pgm_file(im):
    imout = im.copy()
    vmin = np.amin(imout)
    vmax = np.max(imout)
    L = vmax - vmin
    hist, bin_edges = np.histogram(imout.ravel(),bins=L)
    
    SA=np.zeros(len(hist)+1)
    SA[0]=0
    for i in range(len(hist)):
        SA[i+1]=sum(hist[0:i])

    SA=SA*vmax/(imout.shape[0]*imout.shape[1]) #normalizar

    for i in range(len(imout)):
        for j in range(len(imout[i])):
            #ecualizar
            imout[i][j]=int(SA[imout[i][j]])
    
    return imout

if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python P3E1.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    imout = process_pgm_file(im)

    im2 = np.array(imout)
    print("Size of image: {}".format(im2.shape))

    hist_superpuestos(im,im2)
    imagenes_superpuestas(im, im2)

    cv2.imwrite(outfile,imout,[cv2.IMWRITE_PXM_BINARY,0])
