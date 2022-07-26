#######################################################################################################################################

# Example 1: Screws from two suppliers

# Suppose you order screws from two suppliers, ten screws from each vendor. The packages of each supplier specifies the length 
# of the screws to be L = 80 mm. To verify which supplier has the screws nearest to the nominal value you decide measure the length
# of each screw from each supplier with the aid of a ruler graded with a millimetric scale and resolution of 1 mm. Once all screws were 
# measured, you organize the data in two tables, one for each supplier, as shown below:



# Supplier A
# Screw Number    1       2       3       4       5       6       7       8       9       10

# Length (mm)     81      70      83      72      78      81      81      80      80      79

# Supplier B
# Screw Number    1       2       3       4       5       6       7       8       9       10

# Length (mm)     80      80      80      79      79      78      81      81      80      81

# Questions:
# a) How well do the screws fulfil the spec? 

# b) Can we compare the two suppliers?

# c) Can we infer to the whole supply (population)?
#######################################################################################################################################

# Computation os the Arithmetic Mean of Average for the data of the tales above:

import pandas as pd
import numpy as np

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)

Average_Length_for_Screws_From_Supplier_A = np.mean(Lengths_of_the_Screws_From_Supplier_A)

Average_Length_for_Screws_From_Supplier_B = np.mean(Lengths_of_the_Screws_From_Supplier_B)

print ("<a_A> = ", Average_Length_for_Screws_From_Supplier_A)

print ("<a_B> = ", Average_Length_for_Screws_From_Supplier_B)


