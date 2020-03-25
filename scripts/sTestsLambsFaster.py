#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:58:08 2018

@author: Laura
"""

#%% 
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import morphology
from time import time
from scipy import signal
import sys 

#%% Load image
start_time = time()

image_name = 'ta5920.jpg'
# Load a color image #cv2.IMREAD_GRAYSCALE #os.path.join(), glob.glob()
imgColor = cv2.imread(image_name,cv2.IMREAD_COLOR)
img = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)

#img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#Plot image
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(img, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%% Preprocessing
windowSize = 51
kernel = np.ones((windowSize,windowSize)) / windowSize ** 2
blurryImage = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_CONSTANT)

plt.imshow(blurryImage, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#%% Morphological operations

diskSize = 100
ret2,th = cv2.threshold(blurryImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret2,th = cv2.threshold(np.array(blurryImage,dtype='float32'),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(th, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#%%
th2 = th.astype(bool)
cleaned = morphology.remove_small_objects(th2, min_size=500, connectivity=1)
plt.imshow(cleaned, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%%
cleaned2 = cleaned.astype(np.uint8)*255
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

plt.imshow(im_out, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%%

label_img = morphology.label(im_out,connectivity=1)
size = np.bincount(label_img.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = label_img == biggest_label

plt.imshow(clump_mask, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%%
selem = morphology.disk(diskSize)
#eroded = morphology.erosion(clump_mask, selem)
#plt.imshow(eroded, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()

##%%
clump_maskA = clump_mask.astype(np.uint8)*255
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diskSize,diskSize))
#erodedA = cv2.erode(clump_maskA,kernel)

#plt.imshow(erodedA, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
eroded = cv2.erode(clump_maskA,selem)


plt.imshow(eroded, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()



#%%
label2_img = morphology.label(eroded,connectivity=1)
size2 = np.bincount(label2_img.ravel())
biggest_label2 = size2[1:].argmax() + 1
clump_mask2 = label2_img == biggest_label2

plt.imshow(clump_mask2, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%%
#dilated = morphology.dilation(clump_mask2, selem)

#plt.imshow(dilated, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()

clump_mask2A = clump_mask2.astype(np.uint8)*255
dilated = cv2.dilate(clump_mask2A,selem)
#
#
plt.imshow(dilated, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()



#%%
maskSheep = dilated

elapsed_time = time() - start_time
print ("Elapsed time: %0.10f seconds." % elapsed_time)

plt.imshow(maskSheep, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%% Fitting an ellipse

im2,contours,hierarchy = cv2.findContours(maskSheep, 1, 2)
cnt = contours[0]
ellipse = cv2.fitEllipse(cnt)
imgWithEllipse = img.copy()
cv2.ellipse(imgWithEllipse,ellipse,(0,255,0),2)

plt.imshow(imgWithEllipse, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#%%

ellipseMask = np.zeros(img.shape, np.uint8)
cv2.ellipse(ellipseMask,ellipse,(255,255,255),2) #line of three pixels wide


plt.imshow(ellipseMask, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%%
contourMask = np.zeros(maskSheep.shape, np.uint8)
cv2.drawContours(contourMask, contours, -1, (255,255,255), thickness=1)


plt.imshow(contourMask, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

height, width  = maskSheep.shape[:2]


#%%
# center of ellipse
x_c_spatialCoord, y_c_spatialCoord = ellipse[0]
x_c = x_c_spatialCoord # in plot coordinates
y_c = height - 1 - y_c_spatialCoord #in plot coordinates
row_c = np.round(y_c_spatialCoord).astype(int)
col_c = np.round(x_c_spatialCoord).astype(int)
# length of the major axis: MA
#MA, ma = ellipse[1]
MAJ = np.argmax(ellipse[1]) # this is MAJor axis, 1 or 0
MA = ellipse[1][MAJ]
# angle of the ellipse
theta_ellipse = ellipse[2]
theta = 90 - theta_ellipse 
theta_rad = theta * np.pi / 180 

#%%
# rotate countour image
rows,cols = contourMask.astype(int).shape
M = cv2.getRotationMatrix2D((col_c,row_c),-theta,1)
contourRotated_noisy = cv2.warpAffine(contourMask,M,(cols,rows))
#im3,contours_rotated,hierarchy_rotated = cv2.findContours(contourRotated_noisy, 1, 2)
#contourRotated = np.zeros(contourRotated_noisy.shape, np.uint8)
#cv2.drawContours(contourRotated, contours_rotated, -1, (255,255,255), thickness=1)
contourRotated = contourRotated_noisy.copy()
contourRotated[np.where(contourRotated>50)] = 255
contourRotated[np.where(contourRotated<=50)] = 0

plt.imshow(contourRotated, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%% Get the two vertices of the major axis of the ellipse
col_1, row_1 = np.round(col_c - MA/2).astype(int), row_c
col_2, row_2 = np.round(col_c + MA/2).astype(int), row_c

#%%

distances = np.zeros(col_2-col_1+1)
positions_min = np.zeros(col_2-col_1+1)
positions_max = np.zeros(col_2-col_1+1)

for i in range(col_2-col_1+1): 
    values_perpendicular = contourRotated[:, col_1+i]
    positions_true = np.where(values_perpendicular==255)
    if np.size(positions_true) > 0:
        positions_min[i] = np.min(positions_true)
        positions_max[i] = np.max(positions_true)
        distances[i] = positions_max[i] - positions_min[i] + 1
    else:
        distances[i] = 0


plt.plot(distances)
plt.ylabel('distances')
plt.show()


#%% Filter the signal to remove noise

# Create an order 3 lowpass butterworth filter.
b, a = signal.butter(3, 0.05)
# Apply the filter to xn.  Use lfilter_zi to choose the initial condition
# of the filter.
zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, distances, zi=zi*distances[0])
# Apply the filter again, to have a result filtered at an order
# the same as filtfilt.
z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
# Use filtfilt to apply the filter.
distances_smooth = signal.filtfilt(b, a, distances)

# Make the plot.
plt.figure(figsize=(10,5))
plt.plot(distances, 'b', linewidth=1.75, alpha=0.75)
plt.plot(distances_smooth, 'k', linewidth=1.75)
plt.legend(('noisy signal',
        'filtfilt'),
        loc='best')
plt.grid(True)
plt.show()
#plt.savefig('plot.png', dpi=65)

#%% Find local minima
inv_distances_smooth = 1/distances_smooth
max_peakind, _ = signal.find_peaks(inv_distances_smooth)
prominences = signal.peak_prominences(inv_distances_smooth, max_peakind)[0]
plt.plot(distances_smooth)
plt.plot(max_peakind, distances_smooth[max_peakind], "x")
plt.show()

# filter out small distances
idx_to_keep = np.where(distances_smooth[max_peakind] > 50)
max_peakind = max_peakind[idx_to_keep]
prominences = prominences[idx_to_keep]

plt.plot(distances_smooth)
plt.plot(max_peakind, distances_smooth[max_peakind], "x")
plt.show()


if (max_peakind.shape[0]<2):
    print("Error! not enough segments found")
    sys.exit()

#%% take the biggest prominence to each side of the center
idx_center = np.round((col_2-col_1+1) / 2)

prominence_max_left = prominences[max_peakind < idx_center].max()
idx_1 = max_peakind[np.where(prominences==prominence_max_left)][0]
prominence_max_right = prominences[max_peakind > idx_center].max()
idx_2 = max_peakind[np.where(prominences==prominence_max_right)][0]

col_min_1 = col_1 + idx_1 #row_c
col_min_2 = col_1 + idx_2 #row_c

plt.imshow(contourRotated, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.axvline(x=col_min_1)
plt.axvline(x=col_min_2)
plt.show()

#%% take the segments within the contour
row_min_1_top = positions_min[idx_1].astype(int)
row_min_1_bottom = positions_max[idx_1].astype(int)

row_min_2_top = positions_min[idx_2].astype(int)
row_min_2_bottom = positions_max[idx_2].astype(int)

img_coord_a_rotated = np.array([(col_min_1, row_min_1_top)])
img_coord_b_rotated = np.array([(col_min_1, row_min_1_bottom)])
img_coord_c_rotated = np.array([(col_min_2, row_min_2_top)])
img_coord_d_rotated = np.array([(col_min_2, row_min_2_bottom)])
img_coord_middle_segment_point1_rotated = np.array([(col_min_1, row_c)])
img_coord_middle_segment_point2_rotated = np.array([(col_min_2, row_c)])



dist_pixels_segment1 = row_min_1_bottom - row_min_1_top + 1
dist_pixels_segment2 = row_min_2_bottom - row_min_2_top + 1
dist_pixels_between_segments = col_min_2 - col_min_1 + 1

# draw sements on contour
contourOfInterest = contourRotated.copy()
contourOfInterest[row_min_1_top:row_min_1_bottom+1, col_min_1] = 255
contourOfInterest[row_min_2_top:row_min_2_bottom+1, col_min_2] = 255

# erase outer parts of new boundaries
contourOfInterest[:, 0:col_min_1] = 0
contourOfInterest[:, col_min_2+1:] = 0


plt.imshow(contourOfInterest, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#%% obtain perimeter and area
_,countours_contourOfInterest,_ = cv2.findContours(contourOfInterest, 1, 2)
cntRotated = countours_contourOfInterest[0]
perimeter = cv2.arcLength(cntRotated,True)
area = cv2.contourArea(cntRotated)

#%% inverse rotation of the contour of interest
M_inv = cv2.getRotationMatrix2D((col_c,row_c),theta,1)
contourOfInterestOriginal = cv2.warpAffine(contourOfInterest,M_inv,(cols,rows))


plt.imshow(contourOfInterestOriginal, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%% transformation of points to the original orientation

def pointTransformation(points, M_inv): #points as numpy arrays in the shape (col, row)
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    transformed_points = M_inv.dot(points_ones.T).T
    return transformed_points.round()

img_coord_a = pointTransformation(img_coord_a_rotated, M_inv)
img_coord_b = pointTransformation(img_coord_b_rotated, M_inv)
img_coord_c = pointTransformation(img_coord_c_rotated, M_inv)
img_coord_d = pointTransformation(img_coord_d_rotated, M_inv)
img_coord_middle_segment_point1 = pointTransformation(img_coord_middle_segment_point1_rotated, M_inv)
img_coord_middle_segment_point2 = pointTransformation(img_coord_middle_segment_point2_rotated, M_inv)


#%% show representative image - first contour visible and final boundary colored on top
imgColorNorm = cv2.normalize(imgColor.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
maskSheepNorm = cv2.normalize(maskSheep.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
blurred = np.zeros((height,width)) + 0.3
maskSheepBlur = np.maximum(maskSheepNorm,blurred)
maskSheepBlur3D = cv2.merge((maskSheepBlur,maskSheepBlur,maskSheepBlur))
imgColorWithMask = np.multiply(maskSheepBlur3D, imgColorNorm)

selem_draw = morphology.disk(5)
contourOfInterestOriginalDilated = cv2.dilate(contourOfInterestOriginal,selem_draw)

img_coord_contourOfInterestOriginal=cv2.findNonZero(contourOfInterestOriginalDilated)

imgRepresentative = imgColorWithMask.copy()
imgRepresentative[img_coord_contourOfInterestOriginal.T[1], img_coord_contourOfInterestOriginal.T[0],0] = 1.0
imgRepresentative[img_coord_contourOfInterestOriginal.T[1], img_coord_contourOfInterestOriginal.T[0],1] = 0.0
imgRepresentative[img_coord_contourOfInterestOriginal.T[1], img_coord_contourOfInterestOriginal.T[0],2] = 0.0


plt.imshow(imgRepresentative[...,::-1])
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


#%%




#%%
from PIL import Image
imgRepresentativeToSave = np.multiply(imgRepresentative, 255).astype(np.uint8)
imgCont = np.ascontiguousarray(imgRepresentativeToSave)
imgSave = Image.fromarray(imgCont[...,::-1], 'RGB')
imgSave.save('out.png')


#%%
fig, ax = plt.subplots()
ax.imshow(imgColorWithMask[...,::-1])
ax.plot(img_coord_contourOfInterestOriginal.T[0], img_coord_contourOfInterestOriginal.T[1], '.', linewidth=1, color='b')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#%%


















##%%
## Extreme points in ellipse in plot coordinates
#x_1, y_1 = x_c - MA/2 * np.cos(theta_rad), y_c - MA/2 * np.sin(theta_rad)
#x_2, y_2 = x_c + MA/2 * np.cos(theta_rad), y_c + MA/2 * np.sin(theta_rad)
#
## minimun mumber of points (r) to consider in the line of the major axis
#r = np.sqrt( (x_2-x_1)**2 + (y_2-y_1)**2 )
#line = np.arange(0,r,1)
#
## x points of the major axis
#x = x_1 + line * np.cos(theta_rad); 
#x = np.append(x, x_2)
#
## y points of the major axis
#y = y_1 + line * np.sin(theta_rad); 
#y = np.append(y, y_2)
#
#xy = np.column_stack((x,y))
#xy = np.round(xy)
#
## keep only unique pairs of values in the same order. Major axis coordinates
#_, idx = np.unique(xy, return_index=True, axis=0)
#xyUniq = xy[np.sort(idx)]
#
#
## plot the major axis
#fig, ax = plt.subplots()
#ax.imshow(ellipseMask, extent=[0, width, 0, height])
#ax.plot(xyUniq.T[0], xyUniq.T[1], '-', linewidth=5)
#
## draw major axis on the ellipse image
#ellipseMask2 = ellipseMask.copy()
#ellipseMask2[height-1-xyUniq.T.astype(int)[1],xyUniq.T.astype(int)[0]] = 255
#plt.imshow(ellipseMask2, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#
##theta of the perpendiculars
#theta_perp = theta_rad + (90 * np.pi / 180) 
#
## define a line from the coordinates of one point and the slope of the line
#def straightlineFrom1PointSlope(point, m, x):
#  y = m * (x - point[0]) + point[1];
#  return y
#
##%%
#i = 50
#xWholeImage = np.arange(0,width,0.0001)
#yEval = straightlineFrom1PointSlope(xyUniq[i], np.tan(theta_perp), xWholeImage);
#yEval = np.round(yEval)
## clean xEval and xWholeImage out of boundaries of image
#xWholeImage = xWholeImage[yEval>=0]
#yEval = yEval[yEval>=0]
#xWholeImage = xWholeImage[yEval<height]
#yEval = yEval[yEval<height]
#
## plot the perpendicular
#fig, ax = plt.subplots()
#ax.imshow(ellipseMask, extent=[0, width, 0, height])
#ax.plot(xyUniq.T[0], xyUniq.T[1], '-', linewidth=5)
#ax.plot(xWholeImage, yEval, '-', linewidth=5, color='firebrick')
#
#
##%%
#
## convert line to indexes in the image, then find common indexes between line and contour mask, then return coordinates (maybe more than one, cause countour three pixels width, take the extremes) 
#multi_index = np.array([height-1-yEval, xWholeImage]) # height-1-yEval rows,columns
#multi_index = multi_index.astype(int)
#dims = np.array([height, width]) # rows,columns
#
## draw line on the contour mask
#contourMask2 = contourMask.copy()
#contourMask2[multi_index[0],multi_index[1]] = 255
#plt.imshow(contourMask2, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#
#idxLinearLineEval = np.ravel_multi_index(multi_index, dims, order='F')
#idxLinearResp = np.where(contourMask.flatten('F')[idxLinearLineEval]) #contourMask.flatten('F')
#if np.size(idxLinearResp) > 0:
#    rowsResps,colsResps = np.unravel_index(idxLinearLineEval[idxLinearResp], dims, order='F') #comprobado
#
#
#
#
#
##que haya al menos dos separados y coger el primero y último
#
#
##np.savetxt('prueba1.csv', yEval, delimiter=',', fmt='%d')
#

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

#%%

#
#
##%%
#clump_maskA = clump_mask.astype(np.uint8)*255
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diskSize,diskSize))
#erodedA = cv2.erode(clump_maskA,kernel)
#
#
#plt.imshow(erodedA, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#
##%%
#label2_imgA = morphology.label(erodedA,connectivity=1)
#size2A = np.bincount(label2_imgA.ravel())
#biggest_label2A = size2A[1:].argmax() + 1
#clump_mask2A = label2_imgA == biggest_label2A
#
#plt.imshow(clump_mask2A, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#
##%%
#clump_mask2A = clump_mask2A.astype(np.uint8)*255
#dilatedA = cv2.dilate(clump_mask2A,kernel)
#
#
#plt.imshow(dilatedA, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#




#%%
#cleaned2 = cleaned.astype(np.uint8)*255
#contour = cv2.findContours(cleaned2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
#
#for cnt in contour:
#    cv2.drawContours(cleaned2,[ctr],0,255,-1)
#
#
#plt.imshow(cleaned2, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#
##%%
#cleaned2 = cleaned.astype(np.uint8)*255
#
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#res = cv2.morphologyEx(cleaned2,cv2.MORPH_OPEN,kernel)
#
#plt.imshow(res, cmap = 'gray')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()


