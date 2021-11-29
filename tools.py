import math
from typing import NewType
import numpy as np
import cv2
import scipy, scipy.ndimage as ndimage
from image_morpher import ImageMorpher

def style_transfer(style_file, style_mask_file, style_lm_file, ex_file, ex_mask_file, ex_lm_file):
    # On utilise la même valeur de n (nombre d'image dans la pile) que dans le papier.
    n=7
    
    # On récupère les images et les masques associés.
    style = read_file(style_file)
    ex = read_file(ex_file)

    style_mask = read_file(style_mask_file, True)
    ex_mask = read_file(ex_mask_file, True)

    h, w, c = style.shape

    # On va ne travailler que sur les visages et on supprime donc l'arrière plan pour les deux images.
    for channel in range(c):
        style[:,:,channel][style_mask == 0]
        ex[:,:,channel][ex_mask == 0] = 0

    # On calcule ici les différences de gaussiennes ainsi que les résidus (seulement pour l'image stylisée).
    style_dogs, style_residuals = compute_dog(style, n)
    ex_dogs, _ = compute_dog(ex, n)

    # On calcule ici les énergies associées aux différences de gaussienne pour les deux images.
    style_energies = compute_energy(style_dogs)
    ex_energies = compute_energy(ex_dogs)

    # On suit le papier et l'on récupère les valeurs de l'opérateur W(.) à partir des images et de leurs landmarks : Beier and Neely 1992.
    im = ImageMorpher()
    _, vx, vy = im.run(style, ex, read_lm(style_lm_file), read_lm(ex_lm_file))

    # Afin d'appliquer W(.) aux énergies pour l'image stylisée
    for i in range(len(style_energies)):
        style_energies[i] = style_energies[i][vy, vx]

    # On passe au calcul des cartes de gain
    # On reprend les mêmes valeurs d'epsilon et de gain max et min que celle proposées dans l'article
    eps = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.9
    matched = np.zeros(ex.shape)
    for i in range(n):
        # On calcule le gain pour chaque couche suivant la formule définie dans l'article
        gain = (style_energies[i] / (ex_energies[i] + eps))**0.5
        gain[gain < gain_min] = gain_min
        gain[gain > gain_max] = gain_max
        # On construit notre image finale à partir des produits entre les différences de gaussiennes et les gains
        matched += ex_dogs[i] * gain
    # On oublie pas d'ajouter les résidus à notre image
    matched += style_residuals[vy, vx]
    
    
    cv2.imwrite('output/no_bg.jpg', matched*255)
    # On cherche désormais à récupérer l'arrière plan de l'image stylisée de on l'étend par inpainting avant de l'appliquer à notre exemplaire
    bg = bg_inpainting(style, style_mask)
    thres = 20/255
    matched[ex_mask < thres] = bg[ex_mask < thres]
    cv2.imwrite('output/good_bg.jpg', matched*255)
  
    return matched

def read_file(file, grey = False):
    # Nous nous sommes fortement inspirée d'une fonctions déjà existantes dans l'un des TPs
    # Reading image
    if isinstance(file, str):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    else:
        img = file
    # Converting the image to Grayscale
    if grey:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalising values from [0,255] to [0,1]
    if np.max(img) > 1:
        img = img.astype(np.float64) / 255

    return img

# Cette fonction stocke dans une liste des coordonnées des landmarks de lm_file
def read_lm(lm_file):
    f=open(lm_file, "r")
    lines=f.readlines()
    f.close()
    lm=[]
    for line in lines:
        elements=line.split(",")
        lm.append([float(elements[0]),float(elements[1])])
    return lm

# Cette fonction calcule des différences deux à deux des images floutées en suivant la méthode décrite dans l'article
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

# Cette fonction calcule les énergies d'une image à partir de ses différences de gaussienne
def compute_energy(dogs):

    energies = []
    for l, dog in enumerate(dogs):
        energies.append(ndimage.filters.gaussian_filter(dog**2, 2**(l+1)))

    return energies

# En s'inspirant du TP sur l'inpainting, cette fonction comble l'arrière plan de l'image stylisée par la méthode de l'inpainting
def bg_inpainting(style, style_mask):
    kernel = np.array([[0,1,0], [1,0,1], [0,1,0]])

    _, _, c = style.shape

    bg = style.copy()
    bg[style_mask > 20/255] = 0
    
    fixedAreas = style_mask < 20/255
        
    for _ in range(1,100):
        for channel in range(c):
            # On fait la convolution de notre arrière plan par notre noyau avant de rétablir les valeurs des pixels fixe i.e. de l'arrière plan intial
            bg[:,:,channel] = ndimage.correlate(bg[:,:,channel], kernel)/4
            bg[:,:,channel][fixedAreas] = style[:,:,channel][fixedAreas]       
    return bg