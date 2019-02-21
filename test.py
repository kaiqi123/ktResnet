import numpy as np
import matplotlib.pyplot as plt

a = np.load("temp/temp_origin.npy")
print(a.shape)
plt.imshow(a)
plt.show()

b = np.load("temp/temp_whiten.npy")
print(b.shape)
plt.imshow(b)
plt.show()