import math
from typing import NewType
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy, scipy.ndimage as ndimage
from image_morpher import ImageMorpher

def style_transfer(style_file, style_mask_file, style_lm_file, ex_file, ex_mask_file, ex_lm_file):
    n=7
    original_style = read_file(style_file)
    original_ex = read_file(ex_file)

    style_mask = read_file(style_mask_file)
    ex_mask = read_file(ex_mask_file)

    style = np.copy(original_style)
    ex = np.copy(original_ex)

    style[style_mask == 0] = 0
    ex[ex_mask == 0] = 0

    style_dogs, style_residuals = compute_dog(style, n)
    ex_dogs, _ = compute_dog(ex, n)

    
    style_energies = compute_energy(style_dogs)
    ex_energies = compute_energy(ex_dogs)

    im = ImageMorpher()
    ex_lm = read_lm(ex_lm_file)
    style_lm = read_lm(style_lm_file)
    _, vx, vy = im.run(style, ex, style_lm, ex_lm)

    # Post-process warping style stacks:
    for i in range(len(ex_energies)):
        style_dogs[i] = style_dogs[i][vy, vx]
        style_energies[i] = style_energies[i][vy, vx]

    # Compute Gain Map and Transfer
    eps = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.005
    matched = np.zeros(style_mask.shape)
    for i in range(n):
        gain = np.sqrt(np.divide(style_energies[i], (ex_energies[i] + eps)))
        gain[gain <= gain_min] = 1
        gain[gain > gain_max] = gain_max
        matched += np.multiply(ex_dogs[i], gain)
    matched += style_residuals
    matched = np.abs(matched)
    matched*=255
    
    cv2.imwrite('test.jpg', matched)

    bg = bg_inpainting(style, style_mask)

    matched[ex_mask < 20/255] = bg[ex_mask < 20/255]
    cv2.imwrite('output/temp.jpg', matched)
    '''
    temp = np.zeros(ex.shape, dtype=np.uint8)
    temp[style_mask == 0] = style[style_mask == 0]
    temp[ex_mask == 1] = 0
    # xy = (255 - style_mask).astype(np.uint8)
    # bkg = cv.inpaint(temp, xy[:, :, 0], 10, cv.INPAINT_TELEA)
    # imsave('output/bkg.jpg', bkg.astype(int))
    # TODO: Extrapolate background
    xy = np.logical_not(ex_mask.astype(bool))
    matched[xy] = 0
    output = temp + matched
    output[output > 1] = 1
    output[output <= 0] = 0
    output = output.astype(int)
    cv2.imwrite('output/temp.jpg', output*255)
    # imsave('output/temp.jpg', style.astype(int))
    '''

    return matched

def read_file(file):
    # Reading image
    if isinstance(file, str):
        color_img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    else:
        color_img = file
    # Converting the image to Grayscale
    original_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    color_img = np.transpose(color_img, (2, 0, 1))

    # Normalising values from [0,255] to [0,1]
    if np.max(original_img) > 1:
        img = original_img.astype(np.float64) / 255
    else:
        img = original_img

    return img

def read_lm(lm_file):
    f=open(lm_file, "r")
    lines=f.readlines()
    f.close()
    lm=[]
    for i,line in enumerate(lines):
        elements=line.split(",")
        lm.append([float(elements[0]),float(elements[1])])
    return lm

def compute_dog(img, n=7):
    dogs = []
    for l in range(n):
        
        next_blurred = ndimage.filters.gaussian_filter(img, 2**(l+1))
        
        if l==0:
            dogs.append(img - next_blurred)
        else:
            dogs.append(prev_blurred - next_blurred)
        prev_blurred = next_blurred

    residuals = ndimage.filters.gaussian_filter(img, 2**n)
    return dogs, residuals

def compute_energy(dogs):

    energies = []
    for l, dog in enumerate(dogs):
        energies.append(ndimage.filters.gaussian_filter(dog**2, 2**(l+1)))

    return energies

def bg_inpainting(style, style_mask):

    kernel = np.array([[0,1,0],
                      [1,0,1],
                      [0,1,0]])
    lapKernel = np.array([[0,-0.25,0],
                      [-0.25,1,-0.25],
                      [0,-0.25,0]])
    
    bg = style.copy()
    bg[style_mask > 20/255] = 0

    lapTarget = ndimage.correlate(np.zeros_like(style), lapKernel) 
    
    fixedAreas = style_mask < 20/255
        
    for i in range(1,100):
        bg = ndimage.correlate(bg, kernel)/4 + lapTarget
        bg[fixedAreas] = style[fixedAreas]       
    return bg