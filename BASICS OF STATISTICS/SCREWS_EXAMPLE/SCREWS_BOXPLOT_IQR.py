################################################################################################################
#
# Determinatio of the Interquartile Range (IQR) and Boxplot presentation
# 
# A box plot is a type of plot that displays the five number summary of a dataset, which includes:

# The minimum value
# The first quartile (the 25th percentile)
# The median value
# The third quartile (the 75th percentile)
# The maximum value
# We use the following process to draw a box plot:

# Draw a box from the first quartile (Q1) to the third quartile (Q3)
# Then draw a line inside the box at the median
# Then draw “whiskers” from the quartiles to the minimum and maximum values

# When the median is closer to the bottom of the box 
# and the whisker is shorter on the lower end of the box, 
# the distribution is right-skewed (or “positively” skewed).

# When the median is closer to the top of the box 
# and the whisker is shorter on the upper end of the box, 
# the distribution is left-skewed (or “negatively” skewed).

# When the median is in the middle of the box 
# and the whiskers are roughly equal on each side, 
# the distribution is symmetrical (or “no” skew).
#################################################################################################################

import pandas as pd
import numpy as np


# Lengths_of_the_Screws_From_Supplier_A = [81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79]   # Array A

# Lengths_of_the_Screws_From_Supplier_B = [80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81]  # Array B

# # Create a dataframe before the construction of boxplots
# df = pd.DataFrame({'screws From supplier A': Lengths_of_the_Screws_From_Supplier_A,
#                    'Screws From supplier B': Lengths_of_the_Screws_From_Supplier_B})

# # View dataframe
# print(df)

# df.boxplot(column=['Suppliers'], grid = False, color = 'black')


df = pd.DataFrame({'conference': ['A', 'A', 'A', 'B', 'B', 'B'],
                   'points': [5, 7, 7, 9, 12, 9],
                   'assists': [11, 8, 10, 6, 6, 5],
                   'rebounds': [4, 2, 5, 8, 6, 11],})

df.boxplot(column=['points'], grid=False, color='black')
