#
#		SLF-RNAse interaction in Non-self recognition self-incompatible (SI) System
#
##########################################################################################

#from mpl_toolkits.mplot3d import Axes3D
#import pylab
#import sympy as sp
#from operator import itemgetter
#from scipy import linalg
from pylab import genfromtxt
#import fit
#import networkx as nx
from multiprocessing import Pool#, TimeoutError, Process, Queue, Pipe, Value, Array
import os, math, pdb, time, random as ra, numpy as np, matplotlib.pyplot as plt, sys
import collections, pickle
from datetime import datetime
from numba import jit
import pickle
import glob
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick
from operator import itemgetter
#from scipy.integrate import odeint
#from scipy.signal import argrelextrema, find_peaks
#from scipy import signal
import seaborn as sns
import matplotlib as mpl
########################################################################################################
start_time = datetime.now()
print ('Starts|--------|', start_time)

energy_interaction = np.array([[-1,0,0,0],[0,-0.75,-0.25,-0.25],[0,-0.25,1,-1.25],[0,-0.25,-1.25,1]])
#@jit(nopython=True)
def energyOfIntPair(RNAse, FBoxg):  return sum([energy_interaction[x,y] for x, y in zip(RNAse, FBoxg)])

weight = [0.5, 0.265, 0.113, 0.122]

def find_IPR(input_Var):
    total_Sum = sum(input_Var)
    input_Var = input_Var/total_Sum
    IPR = 1.0/sum([fre**2 for fre in input_Var])
    return IPR

def HammingDistance(inputX, inputY):
    hDistance = 0
    for x, y in zip(inputX, inputY):
        if x != y: hDistance += 1
    return hDistance

def frequency_Count(Input_Seq):
    KEY, VALUES = np.unique(Input_Seq, axis=None, return_counts=True)
    Values, indexY = [], 0
    for AA in [0, 1, 2, 3]:
        if AA in KEY:
            Values.append(VALUES[indexY])
            indexY += 1
        else:   Values.append(0)
    return np.array(Values)/sum(Values)

def HammingDistanceWeight(inputX, inputY):
    hDistance = 0
    for x, y in zip(inputX, inputY):
        #if x != y: hDistance += 1
        if x != y:
            hDistance += weight[int(x)]*weight[int(y)]
    return hDistance

def HammingDistanceWeightRealTime(inputX, inputY):
    hDistance = 0
    weight_01 = frequency_Count(inputX)
    weight_02 = frequency_Count(inputY)
    for x, y in zip(inputX, inputY):
        #if x != y: hDistance += 1
        if x != y:
            hDistance += weight_01[int(x)]*weight_02[int(y)]
    return hDistance


#
#
#
# plot for Unique RNAse
def fertilize_Class(input_Var):
    SampleNo, energyThreshold, pathToPick, pathToSave, typeOfGene = input_Var

    distinct_RNAse = 10   # number of initial distinct haplotype
    length_RF = 18   # length of RNAse and F box genes
    window_Size = 500   # if generations == 100000 else 50
    Samples = 15
    tot_Population = 2000
    E_Th = energyThreshold

    list_file_name = glob.glob(pathToPick + '/ancestral_Array_*.dat')

    if len(list_file_name) > 100:
        indices = [i for i, x in enumerate(list_file_name[0]) if x == '_']
        _index = indices[-1]
        gen_temp = [int(each_file[_index+1:-4]) for each_file in list_file_name]
        gen_temp.sort()

        gen_Time_Value = gen_temp[-3:] # change the index here to save the number of data points 
        gen_Time_Value_index = list(range(len(gen_Time_Value)))

        class_prior_haps = {time_iter:0 for time_iter in gen_Time_Value}
        for gen_Time, index in zip(gen_Time_Value, gen_Time_Value_index):
            #
            # indirect RNases  - as ids 0, 1, 2, 3 ...
            unique_RNAse_with_Fre = genfromtxt(pathToPick+'/unique_RNAse_gen_{}.dat'.format(gen_Time)) # unique rnase with counts
            unique_RNAse_inv = genfromtxt(pathToPick+'/inverse_of_RNAse_{}.dat'.format(gen_Time)).astype(int) # inverse of the rnase
            indirect_RNAse = np.array(list(range(len(unique_RNAse_with_Fre)))).astype(int)[unique_RNAse_inv]

            #
            # about all pollen and unique SLFs
            unique_SLF_with_Fre = genfromtxt(pathToPick+'/unique_SLF_gen_{}.dat'.format(gen_Time)) # unique slfs with counts
            unique_SLF_inter = unique_SLF_with_Fre[:,length_RF+2:]

            SLF_nature = pickle.load(open(pathToPick+'/SLF_nature_gen_{}.pkl'.format(gen_Time), 'rb')) # unique pollen and counts
            SLF_nature = np.array(SLF_nature)[:,0] # unique pollen slfs and counts
            SLF_nature_inv = np.genfromtxt(pathToPick+'/SLF_nature_inv_gen_{}.dat'.format(gen_Time)).astype(int)# inverse of the pollen

            #
            # about haps - discarded SC haplotypes
            all_haps = np.concatenate((np.array([indirect_RNAse]).T, SLF_nature[SLF_nature_inv]), axis=1)


            # ----------------------------------------------------------------------------------------------------------------
            # calculations nand data saving
            unique_Hap, unique_Hap_inverse, unique_Hap_count = np.unique(all_haps, axis=0, return_inverse=True, return_counts=True)
            sorted_Index = np.argsort(unique_Hap_count)
            unique_Hap = unique_Hap[sorted_Index][::-1]
            unique_Hap_count = unique_Hap_count[sorted_Index][::-1]

            #np.savetxt(pathToSave+'/hap_count_sam_{}_gen_{}.dat'.format(SampleNo, gen_Time), np.array(unique_Hap_count), fmt='%i')
            #np.savetxt(pathToSave+'/hap_sam_{}_gen_{}.dat'.format(SampleNo, gen_Time), np.array(unique_Hap), fmt='%i')

            fertilize_Group = []
            RNAse_index = unique_Hap[:,0]
            for each_Hap, each_Hap_count in zip(unique_Hap, unique_Hap_count):

                temp_fertilize_Group = []
                SLF_index = each_Hap[1:]
                for each_RNAse_index, each_RNAse_count in zip(RNAse_index, unique_Hap_count):
                    fertilize = 0
                    if each_RNAse_index in unique_SLF_inter[SLF_index][:,each_RNAse_index]:
                        fertilize = 1
                    temp_fertilize_Group.append(fertilize)
                fertilize_Group.append(np.array(temp_fertilize_Group))

            fertilize_Group = np.array(fertilize_Group)
            #np.savetxt(pathToSave+'/fertilize_sam_{}_gen_{}.dat'.format(SampleNo, gen_Time), fertilize_Group, fmt='%i')
            class_prior_haps[gen_Time] = [fertilize_Group, unique_Hap.astype(int), unique_Hap_count.astype(int), unique_RNAse_with_Fre.astype(int), unique_SLF_with_Fre[:,length_RF+1].astype(int)]
            #print (np.shape(fertilize_Group), gen_Time, SampleNo)

        class_prior_haps_Save = open(pathToSave+'/class_prior_haps.pkl', "wb");
        pickle.dump(class_prior_haps, class_prior_haps_Save)
        class_prior_haps_Save.close()


    print (SampleNo, energyThreshold)

if __name__ == '__main__':

    N_Proces = 2
    typeOfGene  = 'SLF' # SLF/RNAse
    Sample_Size = range(1)
    alpha, delta = 0.90, 0.90 # self pollen (selfing), inbreeding depression

    threshold   = [-10, -9, -8, -7, -6, -5, -4, -3, -2]
    main_path   = './variable_ene_al_dl'

    pick_path   = ['/E'+str(th_Value) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100)) for th_Value in threshold]
    save_path   = ['/fig_E'+str(th_Value) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100)) for th_Value in threshold]

    Samples     = [i for th_Value in threshold for i in Sample_Size]
    Threshold   = [th_Value for th_Value in threshold for SampleNo in Sample_Size]

    PathToPick  = [main_path + each_pick_path + '/data_{}'.format(SampleNo) for each_pick_path in pick_path for SampleNo in Sample_Size]
    PathToSave  = [main_path + each_save_path + '/data_{}'.format(SampleNo) for each_save_path in save_path for SampleNo in Sample_Size]

    TypeOfGene  = [typeOfGene for th_Value in threshold for SampleNo in Sample_Size]
    print (len(pick_path), len(save_path), len(Samples), len(Threshold), len(PathToPick), len(PathToSave), len(TypeOfGene))

    Pool(N_Proces).map(fertilize_Class, zip(Samples, Threshold, PathToPick, PathToSave, TypeOfGene))

print ('Ends|----|H:M:S|{}'.format(datetime.now() - start_time), '\n')
###############################################################################################################
###############################################################################################################
