import numpy as np
import math
from skimage import feature



def drawline(image, degreelist, edges):
    beammasks = np.zeros((9,128,128))
    for idx, degree in enumerate(degreelist):
        k = math.tan(math.radians(degree))
        b = []
        bound = np.where(edges != 0)
        for i in range(bound[0].size):
            m = bound[0][i]; n = bound[1][i]      
            b.append(k*m + n)

        min_loc = b.index(min(b)); x0 = bound[0][min_loc]; y0 = bound[1][min_loc]
        max_loc = b.index(max(b)); x02 = bound[0][max_loc]; y02 = bound[1][max_loc]
        thick = 5
        for x in range(image.shape[1]):
            y_min = int(-k*(x - x0) + y0 - thick*math.sqrt(k**2+1))
            y_max = int(-k*(x - x02) + y02 + thick*math.sqrt(k**2+1))
            y = (k for k in range(y_min, y_max+1) if k >= 0 and k < image.shape[1])
            for kk in y:
                beammasks[idx,x,kk] = 1

    return beammasks


if __name__ == "__main__":

    """
    you should load your ptv here slice by slice

    """


    # beam angles
    angles = [0,40,80,120,160,200,240,280,320]

    beammasks = np.zeros((9,128,128,128))
    zmin = np.where(ptv[0,...] != 0)[0].min(); zmax = np.where(ptv[0,...] != 0)[0].max()
    for i in range(beammasks.shape[1]):
        if i <= zmin:
            ptvm = ptv[0,zmin,...].copy()
        elif i >= zmax:
            ptvm = ptv[0,zmax,...].copy()
        else:        
            ptvm = ptv[0,i,...].copy() 
        edges = feature.canny(ptvm, low_threshold=1, high_threshold=1)
        beammasks[:,i,:,:] = drawline(beammasks[:,i,:,:], angles, ptvm)

