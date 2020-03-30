
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import morphology
import skimage
from time import time
from scipy import signal
import sys
import os
from PIL import Image
import glob


#%% Load image
# start_time = time()

ruta = '/Users/Enrique/Google Drive Uni/Test/Pruebas morfología/'
ruta_base = ruta + 'color/'
ruta_photos = ruta + 'JPGs/'
ruta_mask = ruta + 'Mask/'
ruta_mix = ruta + 'Mix/'

base_name = glob.glob(ruta_base + '*.npy')
base_name.sort()
base_name = [x.split('/')[7] for x in base_name]

for filename in base_name:

    image = np.load(ruta_base + filename)
    im = Image.fromarray(image)

    photo_name = ruta_photos + 'I_' + filename
    photo_name = photo_name.replace('npy', 'jpg')
    new_image = im.save(photo_name)
    #new_image = cv2.imwrite(photo_name,im)

photo_name = glob.glob(ruta_photos + '*.jpg')
photo_name.sort()
photo_name = [x.split('/')[7] for x in photo_name]

for filename in photo_name:

    new_image = cv2.imread(ruta_photos + filename)
    img = cv2.normalize(new_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # plt.imshow(img, cmap = 'gray')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

    # Preprocessing
    windowSize = 25
    kernel = np.ones((windowSize,windowSize)) / windowSize ** 2
    blurryImage = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_REPLICATE)

        # plt.imshow(blurryImage, cmap = 'gray')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

    # Morphological operations
    diskSize = 10
    grayimg = cv2.cvtColor(blurryImage, cv2.COLOR_BGR2GRAY)
    scale = cv2.convertScaleAbs(grayimg)
    ret2,th = cv2.threshold(scale,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # plt.imshow(th, cmap = 'gray')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
    th2 = th.astype(bool)

    # Removing small objects
    cleaned_mask = morphology.remove_small_objects(th2, min_size=500, connectivity=1)
            
    # Each object is labelled with a number from 0 to number of regions - 1
#    ret, markers = cv2.connectedComponents(cleaned_mask.astype(np.uint8) * 255)
    markers = morphology.label(cleaned_mask,connectivity=1)
    label_count = np.bincount(np.ravel(markers))
    max_lamb = label_count[1:].argmax()+1
    
    if max_lamb.size > 1:
        
        markers = markers + 1  # So we will label background as 0 and regions from 1 to number of regions
        markers[cleaned_mask == 0] = 0  # background is region 0
        # Removing regions touching the borders of the image
        etiquetas_borde_superior = np.unique(markers[0, :])
        etiquetas_borde_inferior = np.unique(markers[-1, :])
        etiquetas_borde_izquierdo = np.unique(markers[:, 0])
        etiquetas_borde_derecho = np.unique(markers[:, -1])
        etiquetas_bordes = np.unique(np.concatenate([etiquetas_borde_superior, etiquetas_borde_inferior, etiquetas_borde_izquierdo, etiquetas_borde_derecho]))
    
        for label in etiquetas_bordes:
            if label > 0:
                markers[markers == label] = 0
    else:
        markers = markers == max_lamb
    cleaned2 = markers.astype(np.uint8)*255

    # Copy the thresholded image.
    im_floodfill = cleaned2.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
    h, w = cleaned2.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = cleaned2 | im_floodfill_inv

        # plt.imshow(im_out, cmap = 'gray')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

    label_img = morphology.label(im_out,connectivity=1)
    size = np.bincount(np.ravel(label_img))
    biggest_label = size[1:].argmax()+1
    clump_mask = label_img == biggest_label

        # plt.imshow(clump_mask, cmap = 'gray')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

    selem = morphology.disk(diskSize)

    clump_maskA = clump_mask.astype(np.uint8)*255
    eroded = cv2.erode(clump_maskA,selem)

        # plt.imshow(eroded, cmap = 'gray')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
#
    label2_img = morphology.label(eroded,connectivity=1)
    size2 = np.bincount(np.ravel(label2_img))
    biggest_label2 = size2[1:].argmax()+1
    clump_mask2 = label2_img == biggest_label2
#
#        # plt.imshow(clump_mask2, cmap = 'gray')
#        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#        # plt.show()
#
    clump_mask2A = clump_mask2.astype(np.uint8)*255
    dilated = cv2.dilate(clump_mask2A,selem)

        # plt.imshow(dilated, cmap = 'gray')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show() 

    mask_name = ruta_mask + 'M_' + filename
    mask_image = dilated.save(mask_name)

    # Prepare image to do the "mix"
    blurryImage = cv2.filter2D(new_image, -1, kernel, borderType = cv2.BORDER_REPLICATE)
    grayimg = cv2.cvtColor(blurryImage, cv2.COLOR_BGR2GRAY)
    scale_mix = cv2.convertScaleAbs(grayimg)

    one = np.concatenate((scale_mix,th), axis=1)
    two = np.concatenate((cleaned2,im_out), axis=1)
    three = np.concatenate((eroded,dilated), axis=1)
    Mix = np.concatenate((one,two, three), axis=0)
        
    mix_name = ruta_mix + 'Mix_' + filename
    mix_image = cv2.imwrite(mix_name,Mix)



#mask_name = glob.glob(ruta_mask + '*.jpg')
#mask_name.sort()
#mask_name = [x.split('/')[7] for x in mask_name]
#
#for filename in mask_name:
##    print(ruta_mask + filename)
#    img = cv2.imread(ruta_mask + filename)
#    img = np.array(img, dtype=np.float32)
##    cv2.imshow('Prueba1', img)
###    
##    plt.imshow(img, cmap = 'gray')
##    plt.xticks([]), plt.yticks([])
##    plt.show()
#    
##    
#    h = img.shape[0]
#    w = img.shape[1]
###
#    average = cv2.mean(img)[0] / 255
#    # print(average)
#    area = average * h * w
#    print(area)

#%%
