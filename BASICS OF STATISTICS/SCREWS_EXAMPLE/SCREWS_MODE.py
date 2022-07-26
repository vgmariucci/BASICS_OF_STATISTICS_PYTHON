#######################################################################################################################################

# Computation of the Mode ("Moda") for the data on the tables above:

# Recall that the Mode is the value which occurs most often in an array (table, sample or finite population).

# Note that it is possible for an array to contain one or more modes.

#######################################################################################################################################

import pandas as pd
import numpy as np

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B

Average_Length_for_Screws_From_Supplier_A = np.mean(Lengths_of_the_Screws_From_Supplier_A)

Average_Length_for_Screws_From_Supplier_B = np.mean(Lengths_of_the_Screws_From_Supplier_B)


# Find unique values in array for supplier A along with their counts 
vals, counts = np.unique(Lengths_of_the_Screws_From_Supplier_A, return_counts = True)

#find mode for supplier A
mode_value_A = np.argwhere(counts == np.max(counts))

# Find how often the mode for supplier A occurs
Mode_A = vals[mode_value_A]

Number_of_Occurrences_for_Mode_A = np.max(counts).flatten().tolist()

print("\n Mode for supplier A = ", Mode_A)

print("\n Number of occurrences of the mode for supplier A = ", Number_of_Occurrences_for_Mode_A)

# Find unique values in array for supplier B along with their counts 
vals, counts = np.unique(Lengths_of_the_Screws_From_Supplier_B, return_counts = True)

#find mode for supplier B
mode_value_B = np.argwhere(counts == np.max(counts))

# Find how often the mode for supplier A occurs
Mode_B = vals[mode_value_B]

Number_of_Occurrences_for_Mode_B = np.max(counts).flatten().tolist()

print("\n Mode for supplier B = ", Mode_B)

print("\n Number of occurrences of the mode for supplier B = ", Number_of_Occurrences_for_Mode_B)