import cv2
import os
import numpy as np
import sys

import glob

if len(sys.argv)<4:
    print("not enough arguments, need 3 you gave:", len(sys.argv)-1)
    print("usage: python3 im2mask.py useExact inputDir outputDir")
    print("e.g.:  python3 im2mask.py 1 input output")
    print("useExact = 1 => [255,255,255] is background, everything else is object")
    print("           0 => approximate [255,255,255] is background, everything else is object, spurious points are removed")
    print("           2 => approximate [255,255,255] is background, everything else is object, erode")
    sys.exit(0)

useExact = int(sys.argv[1])

input_dir = sys.argv[2]
output_dir = sys.argv[3]

isdir = os.path.isdir(output_dir)
if (not isdir) :
    os.mkdir(output_dir)
    

image_search = input_dir + '/*.png'
listing = glob.glob(image_search)

count = 0
total = len(listing)

for filename in listing:
    count = count + 1
    text = str(count) + '/' + str(total) + ' converting ' + filename
    print(text)
    tail = os.path.split(filename)[1] 
    name = os.path.splitext(tail)[0]
    name_im_out = output_dir + '/' + name + '.png'
    name_mask   = output_dir + '/' + name + '.pbm'
    
    img = cv2.imread(filename)

    b_channel, g_channel, r_channel = cv2.split(img)

    if useExact==1:
        #  this is for pure white:
        mask = (1-(b_channel/255 + g_channel/255 + r_channel/255)/3)*255
    else:
        #  if you want to allow a bit tolerance:
        mask = (1-(b_channel/245 + g_channel/245 + r_channel/245)/3)*255

    if useExact==0 :
        # dilate and erode to get rid of spurious points, extra erode to get a clean boundary
        kernel1 = np.ones((4,4), np.uint8) 
        kernel2 = np.ones((7,7), np.uint8) 
  
        # The first parameter is the original image, 
        # kernel is the matrix with which image is  
        # convolved and third parameter is the number  
        # of iterations, which will determine how much  
        # you want to erode/dilate a given image.  
        mask = cv2.dilate(mask, kernel1, iterations=1) 
        mask = cv2.erode(mask, kernel2, iterations=1) 
        

    if useExact==2 :
        # erode to get rid of spurious points
        kernel2 = np.ones((2,2), np.uint8) 
  
        # The first parameter is the original image, 
        # kernel is the matrix with which image is  
        # convolved and third parameter is the number  
        # of iterations, which will determine how much  
        # you want to erode/dilate a given image.  
        mask = cv2.erode(mask, kernel2, iterations=1) 
        

    cv2.imwrite(name_mask, mask)    
 

    # convert the white (=background) to black
    imgB = img
    im_mask = cv2.merge((mask,mask,mask))
    im_mask = im_mask + abs(im_mask)

    imgB[np.where((im_mask==[0,0,0]).all(axis=2))] = [0,0,0]
    

    cv2.imwrite(name_im_out,imB)
    
