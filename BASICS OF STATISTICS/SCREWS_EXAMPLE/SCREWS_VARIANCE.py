#######################################################################################################################################

# Computation of the Variances for the data on tables above:

# OBS: To interpret the two types of variances below, we must consider the following rule of thumb:  
    
#  * Calculate the Population Variance if our datasets is the entire population (as we are considering since de beginning for the suppliers' screwes example) 

#  * Calculate the Sample Variance if our datasets represents a sample taken from a great population.
# 
#     NOTE: 
#     The Sample Variance of a given array of data will always be larger than the Population Variance for 
#     the same array of a data because there is more uncertainty when calculating the Sample Variance, 
#     thus our estimate of the variance will be larger.
#######################################################################################################################################
from statistics import variance, pvariance

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B

#Funtion do calulate the population variance  of the values on array a
def Calc_Population_Variance(a):
    
    population_variance = pvariance(a)
    
    return population_variance

#Funtion do calulate the sample variance of the values on array a
def Calc_Sample_Variance(a):
    
    Sample_Variance = variance(a)
    
    return Sample_Variance


print("\n Population Variance for supplier A: ", Calc_Population_Variance(Lengths_of_the_Screws_From_Supplier_A))
print("\n Sample Variance for supplier A: ", Calc_Sample_Variance(Lengths_of_the_Screws_From_Supplier_A))

print("\n Population Variance for supplier B: ", Calc_Population_Variance(Lengths_of_the_Screws_From_Supplier_B))
print("\n Sample Variance for supplier B: ", Calc_Sample_Variance(Lengths_of_the_Screws_From_Supplier_B))

