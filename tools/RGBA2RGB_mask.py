import glob
import sys
import os
from PIL import Image, ImageOps

if len(sys.argv)<3:
    print("not enough arguments, need 2 you gave:", len(sys.argv)-1)
    print("usage: python3 RGBA2RGB_mask.py inputDir outputDir")
    print("e.g.:  python3 im2mask.py 1 input output")
    sys.exit(0)


input_dir = sys.argv[1]
output_dir = sys.argv[2]


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


    png = Image.open(filename)
    png.load() # required for png.split()

    background = Image.new("RGB", png.size, (0, 0, 0))
    foreground = Image.new("RGB", png.size, (255, 255, 255))

    
    mask=png.split()[3] # 3 is the alpha channel 
    background.paste(png, mask)
    
#img = Image.open('original-image.png')
#img_with_border = ImageOps.expand(img,border=300,fill='black')
#img_with_border.save('imaged-with-border.png')

    background_ex =  ImageOps.expand(background,border=10,fill='black')
    background_ex.save(name_im_out, 'PNG')
    
    mask_ex =  ImageOps.expand(mask,border=10,fill='black')
    mask_ex.save(name_mask)

    
