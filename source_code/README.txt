Reconciling conflicting selection pressures in the plant collaborative non-self recognition self-incompatibility system
Amit Jangid (1), Keren Erez (1), Ohad Noy Feldheim (2) and Tamar Friedlander (1)

    (1) The Robert H. Smith Institute of Plant Sciences and Genetics in Agriculture
        Faculty of Agriculture, The Hebrew University of Jerusalem,
        P.O. Box 12 Rehovot 7610001, Israel
    (2) The Einstein Institute of Mathematics, Faculty of Natural Sciences,
        The Hebrew University of Jerusalem, Jerusalem 9190401, Israel.
       
    Correspondence: tamar.friedlander@mail.huji.ac.il.

###############################################################################################################

Details of the Python scripts to regenerate the data.

1. main_simulation_run.py
   This python script generates the raw data that need to be analyzed later. Saved data sets are distinct RNases, distinct SLFs, distinct pollens, 
   the interaction of distinct SLFs with RNases, number of ancestors, etc.

2. z_class_prior.py
   This script generates the compatibility matrix of between distinct haplotypes that is used to find the compatibility classes. 

3. z_class_post_main.py
   This script gives the compatibility classes with details of which class has which haplotypes, Hamming distance within and between classes, etc.

###############################################################################################################  




