import cv2
import sys 
import numpy as np
import matplotlib
#matplotlib.use('Agg')
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

def imagenes_superpuestas(im, im2):

    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    imgplot1 = ax1.imshow(im, cmap='gray', vmin=np.amin(im), vmax=np.max(im))
    fig.colorbar(imgplot1, ax=ax1)
    imgplot2 = ax2.imshow(im2, cmap='gray', vmin=np.amin(im2), vmax=np.max(im2))
    fig.colorbar(imgplot2, ax=ax2)
    plt.show()

def process_pgm_file1(im,sx,sy):
    #
    imout = np.zeros((sy,sx),dtype=np.uint8)
    #prop escala
    py=im.shape[0]/sy
    px=im.shape[1]/sx
    #interpolaciona vecino mas cercano
    for i in range(sy):
        for j in range(sx):
            #coordenadas en imagen original
            x=int(j*px)
            y=int(i*py)
            #vecinos mas cercanos
            vecinos=[]
            for a in [-1,0,1]:
                for b in [-1,0,1]:
                    nx=x+a
                    ny=y+b
                    if 0<=nx<im.shape[1] and 0<=ny<im.shape[0]:
                        vecinos.append(im[ny][nx])
            #promedio
            imout[i][j]=np.mean(vecinos)
    
    return imout

def process_pgm_file2(im,sx,sy):
    #
    imout = np.zeros((sy,sx),dtype=np.uint8)
    #prop escala
    py=im.shape[0]/sy
    px=im.shape[1]/sx
    #interpoalcion bilineal
    for i in range(sy):
        for j in range(sx):
            #coordenadas en imagen original
            x=j*px
            y=i*py
            x_floor=int(x)
            y_floor=int(y)
            a = x - x_floor  # Parte fraccionaria en x
            b = y - y_floor  # Parte fraccionaria en y

            #asegurarse de no salir de los bordes
            x_floor = min(x_floor, im.shape[1] - 2)
            y_floor = min(y_floor, im.shape[0] - 2)

            imout[i][j]=(1-a)*(1-b)*im[y_floor,x_floor]+a*(1-b)*im[y_floor,x_floor+1]+(1-a)*b*im[y_floor+1,x_floor]+a*b*im[y_floor+1,x_floor+1]

    return imout

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

if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python P2E3.py [infile.pgm] [outfile.pgm] [outfile2.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    outfile2 = sys.argv[3]

    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    imout = process_pgm_file1(im,1240,1240)     #interpolacion a vecino mas cercano
    print("Size of image: {}".format(imout.shape))
    imagenes_superpuestas(im, imout)

    imout2 = process_pgm_file2(im,1240,1240)     #interpolacion bilineal
    print("Size of image: {}".format(imout2.shape))
    imagenes_superpuestas(im, imout2)

    imagenes_superpuestas3(im, imout, imout2)

    cv2.imwrite(outfile, imout, [cv2.IMWRITE_PXM_BINARY, 0])
    cv2.imwrite(outfile2, imout2, [cv2.IMWRITE_PXM_BINARY, 0])
