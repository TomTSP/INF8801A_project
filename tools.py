import numpy as np
import cv2
import scipy.ndimage as ndimage
from image_morpher import ImageMorpher
import matplotlib.pyplot as plt

def style_transfer(style_file, ex_file):
    # On utilise la même valeur de n (nombre d'image dans la pile) que dans le papier.
    n=6
    
    # On récupère les images et les masques associés.
    style = read_file('input/'+style_file+'.jpg')
    ex = read_file('input/'+ex_file+'.png')
    style_mask = read_file('input/'+style_file+'_mask.png', True)
    ex_mask = read_file('input/'+ex_file+'_mask.png', True)

    _, _, c = style.shape

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
    _, vx, vy = im.run(style, ex, read_lm('input/'+style_file+'.lm'), read_lm('input/'+ex_file+'.lm'))

    # Afin d'appliquer W(.) aux énergies pour l'image stylisée
    for i in range(len(style_energies)):
        style_energies[i] = style_energies[i][vy, vx]

    # On passe au calcul des cartes de gain
    # On reprend les mêmes valeurs d'epsilon et de gain max et min que celle proposées dans l'article
    epsilon = 0.01**2
    gain_max = 2.8
    gain_min = 0.9
    beta = 3
    matched = np.zeros(ex.shape)
    for l in range(n):
        # On calcule le gain pour chaque couche suivant la formule définie dans l'article
        gain = (style_energies[i] / (ex_energies[i] + epsilon))**0.5
        gain[gain < gain_min] = gain_min
        gain[gain > gain_max] = gain_max
        gain = ndimage.filters.gaussian_filter(gain, beta * 2**l)
        plt.imshow(gain)
        plt.show()

        # On construit notre image finale à partir des produits entre les différences de gaussiennes et les gains
        matched += ex_dogs[i] * gain
    # On oublie pas d'ajouter les résidus à notre image
    matched += style_residuals[vy, vx]
    
    
    cv2.imwrite('output/no_bg.jpg', matched)
    # On cherche désormais à récupérer l'arrière plan de l'image stylisée de on l'étend par inpainting avant de l'appliquer à notre exemplaire
    bg = bg_inpainting(style, style_mask)
    cv2.imwrite('output/bg.jpg', bg)
    thres = 20
    matched[ex_mask < thres] = bg[ex_mask < thres]
    cv2.imwrite('output/good_bg.jpg', matched)
  
    return matched

def read_file(file, grey = False, normalize = False):
    # Nous nous sommes fortement inspirée d'une fonctions déjà existantes dans l'un des TPs
    
    # Reading image
    img = cv2.imread(file)
    # Converting the image to Grayscale if asked
    if grey:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalising values from [0,255] to [0,1]
    if np.max(img) > 1 and normalize:
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
def compute_dog(img, n):
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
    bg[style_mask > 20] = 0

    cv2.imwrite('output/bg_before.jpg', bg)
    
    fixedAreas = style_mask < 20
        
    for _ in range(1,100):
        for channel in range(c):
            # On fait la convolution de notre arrière plan par notre noyau avant de rétablir les valeurs des pixels fixe i.e. de l'arrière plan intial
            bg[:,:,channel] = ndimage.correlate(bg[:,:,channel], kernel)
            bg[:,:,channel][fixedAreas] = style[:,:,channel][fixedAreas]       
    return bg