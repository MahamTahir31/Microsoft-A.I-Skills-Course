# MODULE - 8 
# EXPLORE AND ANALYZE DATA WITH PYTHON 

import pandas as pd
import numpy as np
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]


grades = np.array(data)
# Define an array of study hours
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]

# Create a 2D array (an array of arrays)
student_data = np.array([study_hours, grades])

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
                            'StudyHours':student_data[0],
                            'Grade':student_data[1]})

print(df_students)

# Get the data for index value 5
print(df_students.loc[5])

# Get the rows with index values from 0 to 5
print(df_students.loc[0:5])

# Get data in the first five rows
print(df_students.iloc[0:5])

print(df_students.iloc[0,[1,2]])

print(df_students.loc[0,'Grade']) # get only grade value of 0th index

print(df_students.loc[df_students['Name']=='Aisha']) # extracting record of only the student named as Aisha

print(df_students[df_students['Name']=='Aisha']) # same result as above

print(df_students.query('Name=="Aisha"')) # same result as above

print(df_students[df_students.Name == 'Aisha']) # same result as above