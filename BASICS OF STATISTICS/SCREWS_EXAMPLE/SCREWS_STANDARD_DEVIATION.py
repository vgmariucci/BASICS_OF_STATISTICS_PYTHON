#######################################################################################################################################

# Computation of the Standard Deviation (STD)  for the data on tables above:

# Because the standard deviation es the square root of the variance. There is two different ways to calculate:

# 1. Population Standard Deviation;
# 2. Sample Standard Deviation.

# The rules of thumb for variances applies to the standard deviations too:

#  * Calculate the Population STD if our datasets is the entire population (as we are considering since de beginning for the suppliers' screwes example) 

#  * Calculate the Sample STD if our datasets represents a sample taken from a great population.
# 
#     NOTE: 
#     The Sample STD of a given array of data will always be larger than the Population STD for 
#     the same array of a data because there is more uncertainty when calculating the Sample STD, 
#     thus our estimate of the STD will be larger.
###############################################################################################################
import statistics as stat
import numpy as np

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B


# STD unsing numpy library
def Calc_Population_Standard_Deviation_Numpy(a):
    
    Population_STD_Numpy = np.std(a)
    
    return Population_STD_Numpy

def Calc_Sample_Standard_Deviation_Numpy(a):
    
    Sample_STD_Numpy = np.std(a, ddof = 1)
    
    return Sample_STD_Numpy


# STD unsing statistics library
def Calc_Population_Standard_Deviation_Stat(a):
    
    Population_STD_Stat = stat.pstdev(a)
    
    return Population_STD_Stat


def Calc_Sample_Standard_Deviation_Stat(a):
    
    Sample_STD_Stat = stat.stdev(a)
    
    return Sample_STD_Stat



print("\n Population STD for supplier A (done with numpy): ", Calc_Population_Standard_Deviation_Numpy(Lengths_of_the_Screws_From_Supplier_A))

print("\n Population STD for supplier B (done with numpy): ", Calc_Population_Standard_Deviation_Numpy(Lengths_of_the_Screws_From_Supplier_B))



print("\n Sample STD for supplier A (done with numpy): ", Calc_Sample_Standard_Deviation_Numpy(Lengths_of_the_Screws_From_Supplier_A))

print("\n Sample STD for supplier B (done with numpy): ", Calc_Sample_Standard_Deviation_Numpy(Lengths_of_the_Screws_From_Supplier_B))




print("\n Population STD for supplier A (done with statistics): ", Calc_Population_Standard_Deviation_Stat(Lengths_of_the_Screws_From_Supplier_A))

print("\n Population STD for supplier B (done with statistics): ", Calc_Population_Standard_Deviation_Stat(Lengths_of_the_Screws_From_Supplier_B))


print("\n Sample STD for supplier A (done with statistics): ", Calc_Sample_Standard_Deviation_Stat(Lengths_of_the_Screws_From_Supplier_A))

print("\n Sample STD for supplier B (done with statistics): ", Calc_Sample_Standard_Deviation_Stat(Lengths_of_the_Screws_From_Supplier_B))
