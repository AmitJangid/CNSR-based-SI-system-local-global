Reconciling conflicting selection pressures in the plant collaborative non-self recognition self-incompatibility system
Amit Jangid (1), Keren Erez (1), Ohad Noy Feldheim (2) and Tamar Friedlander (1)

		https://www.biorxiv.org/content/10.1101/2024.11.11.622984v1

    (1) The Robert H. Smith Institute of Plant Sciences and Genetics in Agriculture
        Faculty of Agriculture, The Hebrew University of Jerusalem,
        P.O. Box 12 Rehovot 7610001, Israel
    (2) The Einstein Institute of Mathematics, Faculty of Natural Sciences,
        The Hebrew University of Jerusalem, Jerusalem 9190401, Israel.
       
    Correspondence: tamar.friedlander@mail.huji.ac.il.

###############################################################################################################

Source Code:
    The folder 'source_code' contains the Python scripts which are required to generate the data used in the figures.

Simulated dataset:
    The folder 'simulated_dataset' contains final output datasets and Python script to regenerate the figures in the main manuscript.


The following are the requirements for the Python scripts:

1. System requirements
    a) All the scripts are written in Python and are simulated in Python 3.9.19
    b) Following are the main libraries used in the simulation in Python
        numpy 
        pickle 
        matplotlib 
        seaborn 
        os Miscellaneous operating system interfaces [https://docs.python.org/3/library/os.html#]
    c) The scripts were simulated on a local server with a clock speed of 2.4 GHz. It does not require any non-standard hardware.

2. Installation guide:
    a) Python and other libraries that are used to simulate the code are standard and can be installed in many different ways.
       One of the simplest ways is to install via ANACONDA [https://www.anaconda.com/] --- strongly suggested.
    b) We did not use any specific software for simulation.

3. Demo:
    a) Here we provide the data and the Python scripts to regenerate the figures in the main manuscript in the folder 'simulated_dataset'. 
       Data and corresponding scripts are provided in the different folders with python scripts.
    b) Running the scripts will provide the figures which are in the main manuscript.
    c) The run time for only these scripts (for given datasets) is under 3 minutes. 
       To regenerate the whole data for a single run, it takes around 11 hrs on a clock speed of 2.4 GHz for 100000 generations.
    d) We also provide the Mathematica notebook for the calculation of frequency of AA in the reduced model "numerical_minimization_Sanov.nb".  

4. Instructions for use:
    a) No specific software has been used. We use custom codes which are explained above.


###############################################################################################################  




