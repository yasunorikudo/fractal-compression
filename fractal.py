import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
from progressbar import ProgressBar
import multiprocessing

def lstsq(img_large, img_small):
    x = np.reshape(img_small, N*N)
    y = np.reshape(img_large, N*N)
    A = np.array([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A,y)

def calc(s, t):
    residues = np.zeros((256-N,256-N))
    for j in range(256-N):
        for i in range(256-N):
            residues[j][i] = lstsq(img1[t:t+N, s:s+N], img2[j:j+N, i:i+N])[1]

    for j in range(256-N):
        for i in range(256-N):
            if residues[j][i] == np.amin(residues):
                x, y = i, j
                break;
        if residues[j][i] == np.amin(residues):
            break

    m, c = lstsq(img1[t:t+N, s:s+N], img2[y:y+N, x:x+N])[0]
    return np.r_[x, y, m, c]

def main(num):
    p = np.zeros((512/N/cpu, 512/N, 4))
    if num == 0: pbar = ProgressBar((512/N/cpu)*(512/N))
    for j in range(512/N/cpu):
        for i in range(512/N):
            p[j][i] = calc(i*N, (j+512/N/cpu*num)*N)
            if num == 0: pbar.update((512/N)*j+i+1)
    return p

if __name__ == '__main__':
    cpu = multiprocessing.cpu_count()
    N = 32
    img1 = np.array(Image.open('Lenna.jpg').convert("L"))
    img2 = cv2.resize(img1,(256,256))

    pool = multiprocessing.Pool(cpu)
    callback = pool.map(main, range(cpu))

    print 'process finished!'

    d = np.zeros((512/N, 512/N, 4))
    for i in range(cpu):
        d[i*512/N/cpu:(i+1)*512/N/cpu] = callback[i]

    img3 = np.zeros((256, 256))
    dst = np.zeros((512, 512))

    for k in range(10):
        for j in range(512/N):
            for i in range(512/N):
                dst[j*N:j*N+N, i*N:i*N+N] = img3[d[j][i][1]:d[j][i][1]+N, d[j][i][0]:d[j][i][0]+N] * d[j][i][2] + d[j][i][3]
        img3 = cv2.resize(dst,(256,256))
        cv2.imwrite(str(N)+"_"+str(k+1)+".jpg", dst)
