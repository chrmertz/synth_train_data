import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
import random
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import PIL.ImageOps    
import scipy
from multiprocessing import Pool
from functools import partial
import signal
import time

import cProfile

from defaults import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from pb import *
import math
from pyblur import *
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """ 
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])

def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate 
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    #lineTypeIdx = np.random.randint(0, len(lineTypes))
    # instead of choosing it randomly, we take "full", otherwise the object will shift:
    lineTypeIdx = 2
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    #print(lineType)
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def overlap(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed 
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes 
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    
    if (dx>=0) and (dy>=0) and float(dx*dy) > MAX_ALLOWED_IOU*(a.xmax-a.xmin)*(a.ymax-a.ymin):
        return True
    else:
        return False

def get_list_of_images(root_dir, N=1):
    '''Gets the list of images of objects in the root directory. The expected format 
       is root_dir/<object>/<image>.png. Adds an image as many times you want it to 
       appear in dataset.

    Args:
        root_dir(string): Directory where images of objects are present
        N(int): Number of times an image would appear in dataset. Each image should have
                different data augmentation
    Returns:
        list: List of images(with paths) that will be put in the dataset
    '''
    img_list = glob.glob(os.path.join(root_dir, '*/*.png'))
    img_list_f = []
    for i in range(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f

def get_mask_file(img_file):
    '''Takes an image file name and returns the corresponding mask file. The mask represents
       pixels that belong to the object. Default implentation assumes mask file has same path 
       as image file with different extension only. Write custom code for getting mask file here
       if this is not the case.

    Args:
        img_file(string): Image name
    Returns:
        string: Correpsonding mask file path
    '''
    filename, file_extension = os.path.splitext(img_file)
    mask_file = img_file.replace(file_extension,'.pbm')
    return mask_file

def get_labels(imgs):
    '''Get list of labels/object names. Assumes the images in the root directory follow root_dir/<object>/<image>
       structure. Directory name would be object name.

    Args:
        imgs(list): List of images being used for synthesis 
    Returns:
        list: List of labels/object names corresponding to each image
    '''
    labels = []
    for img_file in imgs:
        label = img_file.split('/')[-2]
        labels.append(label)
    return labels

def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations

    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print("%s not found. Using empty mask instead."%mask_file)
        return -1, -1, -1, -1

def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def write_imageset_file(exp_dir, img_files, anno_files):
    '''Writes the imageset file which has the generated images and corresponding annotation files
       for a given experiment

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        img_files(list): List of image files that were generated
        anno_files(list): List of annotation files corresponding to each image file
    '''
    with open(os.path.join(exp_dir,'train.txt'),'w') as f:
        for i in range(len(img_files)):
            f.write('%s %s\n'%(img_files[i], anno_files[i]))

def write_labels_file(exp_dir, labels):
    '''Writes the labels file which has the name of an object on each line

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    '''
    unique_labels = ['__background__'] + sorted(set(labels))
    with open(os.path.join(exp_dir,'labels.txt'),'w') as f:
        for i, label in enumerate(unique_labels):
            f.write('%s %s\n'%(i, label))

def keep_selected_labels(img_files, labels):
    '''Filters image files and labels to only retain those that are selected. Useful when one doesn't 
       want all objects to be used for synthesis

    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponidng to each imahe in above list
    '''
    with open(SELECTED_LIST_FILE) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    new_img_files = []
    new_labels = []
    for i in range(len(img_files)):
        if labels[i] in selected_labels:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels
 
def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])
    

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    #print('mode, img.size[1], img.size[0]: ',img.mode,' ', img.size[1], ' ',img.size[0])
    if img.mode != 'RGB' :
        print('### WARNING ###: The image mode should be RGB but it is: ',img.mode)
        print('                 If the program crashes, you have an unsuitable image, probably an unsuitable background image in png format')
#    return np.array(img.getdata(),
#                    np.uint8).reshape(img.size[1], img.size[0], 3)
    return np.array(img)

def create_image_anno_wrapper(args, w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=False):
   ''' Wrapper used to pass params to workers
   '''
   return create_image_anno(*args, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=blending_list, dontocclude=dontocclude)


def rotation_inter(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def paste_transparent(background,foreground,mask,mask_org,x,y,kn=3):
    # background = foreground will be pasted on background
    # mask_org   = mask without any blurring. But because of rotation and pixel interpolation
    #              it might have some values between 0 and 255
    # mask       = final mask, i.e.mask_org with gaussian, box, etc. blurring. For 'normal'
    #              mask should be same as mask_org
    # x,y        = location where foreground will be pasted
    # kn         = kernel size, should be at least 3, otherwise same as kernel for gaussian, box etc.
    
    
    if (kn % 2) == 0 :
        kn = kn + 1

    # dilate the foreground by kn.
    foreground_dilate = foreground.filter(ImageFilter.MaxFilter(kn))

    # get the part of the original mask that is 255
    #ret,thresh1 = cv2.threshold(PIL2array1C(mask_org),254,255,cv2.THRESH_BINARY)
    ret,thresh1 = cv2.threshold(mask_org,254,255,cv2.THRESH_BINARY)
    mask_thresh = Image.fromarray(thresh1)

    # paste the original foreground into the dilated forground. We want to keep the original foreground
    # and have the dilation effects only on the border
    foreground_dilate.paste(foreground,(0,0),mask_thresh)

    #cv2.imwrite("mask_org.png", PIL2array1C(mask_org))
    #cv2.imwrite("foreground_dilate.png", PIL2array3C(background) )
    
    background.paste(foreground_dilate, (x, y), mask)

    return background

def fill_holes(im_in):

    # we need to pad the image by one line so that the filling starts from all
    # sides. If you don't do that and the object intersects the side(s) at
    # two locations, the filling will be incorrect.
    # At the end of this function we will remove the extra border.
    im_in = cv2.copyMakeBorder(im_in, 1,1,1,1, cv2.BORDER_CONSTANT)

    gray = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
    gray = ~gray

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # Undo the padding (MakeBorader)
    im_out = im_out[1:-1,1:-1]

    return im_out



def create_image_anno(objects, distractor_objects, img_file, anno_file, bg_file,  w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=False):
    '''Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        distractor_objects(list): List of distractor objects that will be synthesized but whose annotations are not required
        img_file(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path 
        w(int): Width of synthesized image
        h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        dontocclude(bool): Generate images with occlusion
    '''

    if 'none' not in img_file:
        return 
    
    print("Working on %s" % img_file)
    if os.path.exists(anno_file):
        return anno_file
    
    all_objects = objects + distractor_objects

    num_of_objects = len(objects)
    
    while True:
        top = Element('annotation')
        background = Image.open(bg_file)
        print('background file: ',bg_file)
        #print('img_file: ', img_file)
        background = background.resize((w, h), Image.ANTIALIAS)
        backgrounds = []
        for i in range(len(blending_list)+1):
            backgrounds.append(background.copy())
        
        if dontocclude:
            already_syn = []
        for idx, obj in enumerate(all_objects):
           foreground = Image.open(obj[0])
           #print('foreground: ', obj[0])
           xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj[0]))
           if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
               continue
           foreground = foreground.crop((xmin, ymin, xmax, ymax))
           orig_w, orig_h = foreground.size
           mask_file =  get_mask_file(obj[0])
           mask = Image.open(mask_file)
           mask = mask.crop((xmin, ymin, xmax, ymax))
           o_w, o_h = orig_w, orig_h

           # add padding around foreground and mask, we need that to do the blending correctly
           # padding of 8, padding color = black
           
           mask_cv = cv2.copyMakeBorder(PIL2array1C(mask), 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[0,0,0])
           mask = Image.fromarray( mask_cv)
           foreground = Image.fromarray(cv2.copyMakeBorder(PIL2array3C(foreground), 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[0,0,0]))

           factor = random.uniform(MIN_BRIGHTNESS, MAX_BRIGHTNESS)

           if factor > 1:
               factor = 2 - factor
               foreground = PIL.ImageOps.invert(foreground)
               brightor = ImageEnhance.Brightness(foreground)
               foreground = brightor.enhance(factor)
               foreground = PIL.ImageOps.invert(foreground)
           else:
               brightor = ImageEnhance.Brightness(foreground)
               foreground = brightor.enhance(factor)
               
           
           
           if scale_augment:
                while True:
                    scale = random.uniform(MIN_SCALE, MAX_SCALE)
                    o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                    if  w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0:
                        break
                foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
           if rotation_augment:
               max_degrees = MAX_DEGREES  
               while True:
                   rot_degrees = random.randint(-max_degrees, max_degrees)

                   # PIL rotations did not work right for BILINEAR and BICUBIC, let's do it with opencv instead:
                   f_tmp = Image.fromarray(rotation_inter(PIL2array3C(foreground),rot_degrees))
                   foreground_tmp = f_tmp
                   
                   ## Image.fromarray(cv2.blur(PIL2array1C(foreground),(rx,ry))
                   #mask_tmp = mask.rotate(rot_degrees, expand=True)
                   #m_tmp_cv = rotation_inter(mask_cv,rot_degrees)
                   #m_tmp = Image.fromarray(rotation_inter(PIL2array1C(mask,rot_degrees))
                   mask_tmp = mask.rotate(rot_degrees, expand=True)
                   m_tmp = Image.fromarray(rotation_inter(PIL2array1C(mask),rot_degrees))
                   mask_tmp = m_tmp
                   o_w, o_h = foreground_tmp.size
                   if  w-o_w > 0 and h-o_h > 0:
                        break
               mask = mask_tmp
               #mask_cv = m_tmp_cv
               foreground = foreground_tmp
           
           xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
           attempt = 0
           while True:
               attempt +=1
               x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
               y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))
               if dontocclude:
                   found = True
                   for prev in already_syn:
                       ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                       rb = Rectangle(x+xmin, y+ymin, x+xmax, y+ymax)
                       if overlap(ra, rb):
                             found = False
                             break
                   if found:
                      break
               else:
                   break
               if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
                   break
           if dontocclude:
               already_syn.append([x+xmin, x+xmax, y+ymin, y+ymax])

           mask_cv = PIL2array1C(mask)
           for i in range(len(blending_list)+1):
               if i == len(blending_list):
                  if  idx < num_of_objects :
                      background = backgrounds[i]
                      background_black = Image.new("RGB", background.size, (0, 0, 0))
                      foreground_white = Image.new("RGB", foreground.size, (255, 255, 255))
                      backgrounds[i] = paste_transparent(background_black,foreground_white,mask,mask_cv,x,y)
                  #cv2.imwrite("background_black.png", PIL2array3C(backgrounds[i]) )

               elif blending_list[i] == 'none' or blending_list[i] == 'motion':
                   backgrounds[i] = paste_transparent(backgrounds[i],foreground,mask,mask_cv,x,y)
                   
               elif blending_list[i] == 'poisson':
                    # old way of doing the poisson is to use poisson_blend, but it is quite slow 
##                  offset = (y, x)
##                  img_mask = PIL2array1C(mask)
##                  img_src = PIL2array3C(foreground).astype(np.float64)
                  
##                  img_target = PIL2array3C(backgrounds[i])
##                  img_mask, img_src, offset_adj \
##                       = create_mask(img_mask.astype(np.float64),
##                          img_target, img_src, offset=offset)
##                  background_array = poisson_blend(img_mask, img_src, img_target,
##                                    method='normal', offset_adj=offset_adj)
##                  backgrounds[i] = Image.fromarray(background_array, 'RGB')

                  # We will use Clone seamlessly from OpenCV instead of poisson_blend
                  #print("x: ",x," y: ",y)
                  #print("mask_x: ",mask.size[0],' mask_y: ',mask.size[1])
                  #print("background_x: ",backgrounds[i].size[0]," background_y: ",backgrounds[i].size[0])

                  # semalessClone does not work if you place parts of the object outside of the background,
                  # so we need to crop the mask and foreground to fit into the background image:
                  left = 0
                  if x<0 :
                      left = -x

                  top1 = 0
                  if y<0 :
                      top1 = -y

                  right = mask.size[0]
                  if x+mask.size[0] > backgrounds[i].size[0] :
                      right = backgrounds[i].size[0] - x
                  
                  bottom = mask.size[1]
                  if y+mask.size[1] > backgrounds[i].size[1] :
                      bottom = backgrounds[i].size[1] - y

                  #print(" left: ", left, " top1: ", top1, " right: ", right, " bottom: ", bottom)
                  mask_crop = mask.crop((left, top1, right, bottom))
                  
                  foreground_crop = foreground.crop((left, top1, right, bottom))

                  # seamlessClone is weird, it shifts the object unless you take the center of the
                  # bounding rectangle instead of the center of the mask. Even then, it might be off
                  # by a pixel or so
                  mask_crop_cv = PIL2array1C(mask_crop)
                  #mask_crop_cv[:] = 255
                  br = cv2.boundingRect(mask_crop_cv) # bounding rect (x,y,width,height)
                  center = (x+left+ br[0] + br[2] // 2,y+top1+ br[1] + br[3] // 2)

                  #print("center: ", center)
                  #print("br: ", br)
                  
                  #print("mask_crop.size[0]: ", mask_crop.size[0], " mask_crop.size[1]: ", mask_crop.size[1]) 
                  
                  #center = (x+left+int(mask_crop.size[0]/2),y+top1+int(mask_crop.size[1]/2))
                  #center = (x,y)
                  #print(center)
                  #cv2.imwrite("mask_crop.png",mask_crop_cv )
                  output = cv2.seamlessClone(PIL2array3C(foreground_crop),PIL2array3C(backgrounds[i]) , mask_crop_cv , center, cv2.NORMAL_CLONE)
                  backgrounds[i] = Image.fromarray(output, 'RGB')
                  #print(" ")
                  
               elif blending_list[i] == 'gaussian':
                  rx = random.randrange(3, 8,2)
                  ry = random.randrange(3, 8,2)
                  kn = max([rx,ry,3])
                  #mask_gauss_cv = cv2.GaussianBlur(mask_cv,(rx,ry),2))
                  #mask_gauss = Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(rx,ry),2))
                  mask_gauss = Image.fromarray(cv2.GaussianBlur(mask_cv,(rx,ry),2))
                  backgrounds[i] = paste_transparent(backgrounds[i],foreground,mask_gauss,mask_cv,x,y,kn)
                  
               elif blending_list[i] == 'box':
                  rx = random.randint(2, 6)
                  ry = random.randint(2, 6)
                  
                  kn = max([rx,ry,3])
                  #mask_box = Image.fromarray(cv2.blur(PIL2array1C(mask),(rx,ry)))
                  mask_box = Image.fromarray(cv2.blur(mask_cv,(rx,ry)))
                  backgrounds[i] = paste_transparent(backgrounds[i],foreground,mask_box,mask_cv,x,y,kn)
                  

           if idx >= len(objects):
               continue 
           object_root = SubElement(top, 'object')
           object_type = obj[1]
           object_type_entry = SubElement(object_root, 'name')
           object_type_entry.text = str(object_type)
           object_bndbox_entry = SubElement(object_root, 'bndbox')
           x_min_entry = SubElement(object_bndbox_entry, 'xmin')
           x_min_entry.text = '%d'%(max(1,x+xmin))
           x_max_entry = SubElement(object_bndbox_entry, 'xmax')
           x_max_entry.text = '%d'%(min(w,x+xmax))
           y_min_entry = SubElement(object_bndbox_entry, 'ymin')
           y_min_entry.text = '%d'%(max(1,y+ymin))
           y_max_entry = SubElement(object_bndbox_entry, 'ymax')
           y_max_entry.text = '%d'%(min(h,y+ymax))
           difficult_entry = SubElement(object_root, 'difficult')
           difficult_entry.text = '0' # Add heuristic to estimate difficulty later on
        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
           continue
        else:
           break

    for i in range(len(blending_list)):
        #if i == len(blending_list):
        #    print("one time")

        #else :
        if blending_list[i] == 'motion':
            backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))

    # we want to have images with and without jpg compression artifacts
        njpg = random.randrange(80, 120, 1)
    # if njpg is > 100, then we will have no compression artifacts (1/2 of the cases)
    # else we will have random compression between 80% and 100%
    # Normal compression for jpg is 90%



        if njpg<=100 :
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), njpg]
            result, encimg = cv2.imencode('.jpg',PIL2array3C(backgrounds[i]) , encode_param)
            backgrounds[i] = Image.fromarray(cv2.imdecode(encimg, 1))

        backgrounds[i].save(img_file.replace('none', blending_list[i]),quality=100)



        path_list = anno_file.split(os.sep)
        train_file = path_list[0] + ".txt"
        with open(train_file, "a") as ff:
            ff.write(img_file.replace('none', blending_list[i]) + ',' +  x_min_entry.text + ',' +  y_min_entry.text + ',')
            ff.write(x_max_entry.text + ',' +  y_max_entry.text + ',' + object_type_entry.text + '\n')

    xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
    with open(anno_file, "w") as f:
        f.write(xmlstr)

    ii =  len(blending_list)
    im_in = PIL2array3C(backgrounds[ii])
    im_out = fill_holes(im_in)

    pre, ext = os.path.splitext(anno_file)

    label_txt =  objects[0][1]
    
    # the following only works for one object
    for i in range(len(blending_list)):
        anno_im_file = pre + "_" + blending_list[i] + "_" + label_txt + "_1.png"
        cv2.imwrite(anno_im_file,im_out)


    #backgrounds[ii].save(anno_im_file)
    
    #print(anno_file)
    #print(anno_im_file)
    #print("two time")
   
def gen_syn_data(img_files, labels, img_dir, anno_dir, scale_augment, rotation_augment, dontocclude, add_distractors):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image
        img_dir(str): Directory where synthesized images will be stored
        anno_dir(str): Directory where corresponding annotations will be stored
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        dontocclude(bool): Generate images with occlusion
        add_distractors(bool): Add distractor objects whose annotations are not required 
    '''
    w = WIDTH
    h = HEIGHT
    background_dir = BACKGROUND_DIR
    background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))
    print( "background dir: %s" %background_dir)

    print( "Number of background images : %s"%len(background_files) )
    img_labels = list(zip(img_files, labels))
    random.shuffle(img_labels)

    if add_distractors:
        with open(DISTRACTOR_LIST_FILE) as f:
            distractor_labels = [x.strip() for x in f.readlines()]

        distractor_list = []
        for distractor_label in distractor_labels:
            distractor_list += glob.glob(os.path.join(DISTRACTOR_DIR, distractor_label, DISTRACTOR_GLOB_STRING))
            print('Distractors: ', distractor_label)

        distractor_files = list(zip(distractor_list, len(distractor_list)*[None]))
        random.shuffle(distractor_files)

    idx = 0
    img_files = []
    anno_files = []
    params_list = []
    while len(img_labels) > 0:
        # Get list of objects
        objects = []
        n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(img_labels))
        for i in range(n):
            objects.append(img_labels.pop())
        # Get list of distractor objects 
        distractor_objects = []
        if add_distractors:
            n = min(random.randint(MIN_NO_OF_DISTRACTOR_OBJECTS, MAX_NO_OF_DISTRACTOR_OBJECTS), len(distractor_files))
            for i in range(n):
                distractor_objects.append(random.choice(distractor_files))
        idx += 1
        bg_file = random.choice(background_files)
        for blur in BLENDING_LIST:
            img_file = os.path.join(img_dir, '%i_%s.jpg'%(idx,blur))
            anno_file = os.path.join(anno_dir, '%i.xml'%idx)
            params = (objects, distractor_objects, img_file, anno_file, bg_file)
            params_list.append(params)
            img_files.append(img_file)
            anno_files.append(anno_file)
            create_image_anno(objects, distractor_objects, img_file, anno_file, bg_file,  w, h, scale_augment, rotation_augment,BLENDING_LIST , dontocclude)
    
##    partial_func = partial(create_image_anno_wrapper, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dontocclude=dontocclude) 
##    p = Pool(NUMBER_OF_WORKERS, init_worker)
##    try:
##        p.map(partial_func, params_list)
##    except KeyboardInterrupt:
##        print( "....\nCaught KeyboardInterrupt, terminating workers")
##        p.terminate()
##    else:
##        p.close()
##    p.join()
    return img_files, anno_files

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
    ''' Generate synthetic dataset according to given args
    '''
    train_file = args.exp + ".txt"
    print( "base: %s" %train_file)
    fx = open(train_file, "w")
    fx.close()

    img_files = get_list_of_images(args.root, args.num) 
    labels = get_labels(img_files)

    if args.selected:
       img_files, labels = keep_selected_labels(img_files, labels)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)
    
    write_labels_file(args.exp, labels)

    anno_dir = os.path.join(args.exp, 'annotations')
    img_dir = os.path.join(args.exp, 'images')
    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)
    
    syn_img_files, anno_files = gen_syn_data(img_files, labels, img_dir, anno_dir, args.scale, args.rotation, args.dontocclude, args.add_distractors)
    write_imageset_file(args.exp, syn_img_files, anno_files)

def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
    parser.add_argument("exp",
      help="The directory where images and annotation lists will be created.")
    parser.add_argument("--selected",
      help="Keep only selected instances in the test dataset. Default is to keep all instances in the roo directory", action="store_true")
    parser.add_argument("--scale",
      help="Add scale augmentation.Default is to not add scale augmentation.", action="store_true")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is to not add rotation augmentation.", action="store_true")
    parser.add_argument("--num",
      help="Number of times each image will be in dataset", default=1, type=int)
    parser.add_argument("--dontocclude",
      help="Add objects without occlusion. Default is to produce occlusions", action="store_true")
    parser.add_argument("--add_distractors",
      help="Add distractors objects. Default is to not use distractors", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    ##pr = cProfile.Profile()
    ##pr.enable()

    args = parse_args()
    generate_synthetic_dataset(args)

    ##pr.disable()
    #pr.print_stats()
    #pr.sort_stats(SortKey.CUMULATIVE)
    #pr.print_stats(10)
    #pr.print_stats(sort='time')
    ##pr.print_stats(sort=('cumtime'))
