# MODULE - 8 
# EXPLORE AND ANALYZE DATA WITH PYTHON 

data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
print(data) #python list

import numpy as np

grades = np.array(data)
print(grades) #numpy array

print (type(data),'x 2:', data * 2) # it will write the data list two time
print('---')
print (type(grades),'x 2:', grades * 2) # it will multiply the 2 with grades array elements

print(grades.shape)
print(grades[0])

print("Mean of grades: ",grades.mean())