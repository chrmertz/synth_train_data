# Paths
# Fill this according to own setup
BACKGROUND_DIR = 'background'
BACKGROUND_GLOB_STRING = '*.png'
POISSON_BLENDING_DIR = '' # if downloaded from S
SELECTED_LIST_FILE = 'selected.txt'
DISTRACTOR_LIST_FILE = 'neg_list.txt' 
DISTRACTOR_DIR = 'distractor'
DISTRACTOR_GLOB_STRING = '*.jpg'

# Parameters for generator
NUMBER_OF_WORKERS = 1 #26
BLENDING_LIST = ['gaussian','poisson', 'none', 'box', 'motion']

# Parameters for images
MIN_NO_OF_OBJECTS = 1
MAX_NO_OF_OBJECTS = 1
MIN_NO_OF_DISTRACTOR_OBJECTS = 1
MAX_NO_OF_DISTRACTOR_OBJECTS = 1
WIDTH = 1024
HEIGHT = 750
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# Parameters for objects in images
#  contrast: factor of 0.0 gives a solid grey image. A factor of 1.0 gives the original image.
MIN_CONTRAST = 0.5 # minium contrast
MAX_CONTRAST = 1.5 # maximum contrast
# brightness: factor of 0.0 gives a black image. A factor of 1.0 gives the original image.
MIN_BRIGHTNESS = 0.5 # minimum brightness
MAX_BRIGHTNESS = 1.5 # maximum brightness

MIN_SCALE = 0.1 # min scale for scale augmentation
MAX_SCALE = 1.0 # max scale for scale augmentation
MAX_DEGREES = 30 # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = 0.25 # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
MAX_ALLOWED_IOU = 0.75 # IOU > MAX_ALLOWED_IOU is considered an occlusion
MIN_WIDTH = 6 # Minimum width of object to use for data generation
MIN_HEIGHT = 6 # Minimum height of object to use for data generation
