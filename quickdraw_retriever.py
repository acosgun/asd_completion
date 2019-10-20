import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob

num_pics = 124000

filename = "smiley_face"

img_array = np.load(filename+".npy")
print(img_array.shape)

count = 0
for i in range(0,num_pics):    
    img = img_array[i,:].reshape(28,28)        
    img_name = filename + str(i) + ".png"
    plt.imsave("./"+ filename + "/"+ img_name, img, cmap='gray')
