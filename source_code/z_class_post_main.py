#
#		SLF-RNAse interaction in Non-self recognition self-incompatible (SI) System
#
##########################################################################################

#from mpl_toolkits.mplot3d import Axes3D
#import pylab
#import sympy as sp
#from operator import itemgetter
#from scipy import linalg
from pylab import genfromtxt # type: ignore
#import fit
#import networkx as nx
from multiprocessing import Pool#, TimeoutError, Process, Queue, Pipe, Value, Array
import os, math, pdb, time, random as ra, numpy as np, matplotlib.pyplot as plt, sys # type: ignore
import collections, pickle
from datetime import datetime
from numba import jit # type: ignore
import pickle
import glob
from scipy.optimize import curve_fit # type: ignore
import matplotlib.ticker as mtick # type: ignore
from operator import itemgetter
#from scipy.integrate import odeint
#from scipy.signal import argrelextrema, find_peaks
#from scipy import signal
import seaborn as sns # type: ignore
import matplotlib as mpl # type: ignore
########################################################################################################
start_time = datetime.now()
print ('Starts|--------|', start_time)

energy_interaction = np.array([[-1,0,0,0],[0,-0.75,-0.25,-0.25],[0,-0.25,1,-1.25],[0,-0.25,-1.25,1]])
#@jit(nopython=True)
def energyOfIntPair(RNAse, FBoxg):  return sum([energy_interaction[x,y] for x, y in zip(RNAse, FBoxg)])

weight = [0.5, 0.265, 0.113, 0.122]

def find_ipr(input_Var):
    total_Sum = sum(input_Var)
    input_Var = input_Var/total_Sum
    IPR = 1.0/sum([fre**2 for fre in input_Var])
    return IPR

def HammingDistance(inputX, inputY):
    hDistance = 0
    for x, y in zip(inputX, inputY):
        if x != y: hDistance += 1
    return int(hDistance)

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
# plot for Unique RNAse
def fertilize_Class(input_Var):
    SampleNo, energyThreshold, pathToPick, pathToSave, typeOfGene = input_Var

    distinct_RNAse = 10   # number of initial distinct haplotype
    length_RF = 18   # length of RNAse and F box genes
    window_Size = 500   # if generations == 100000 else 50
    Samples = 25
    tot_Population = 2000
    E_Th = energyThreshold


    if (os.path.exists(pathToPick + '/class_prior_haps.pkl')):
        class_prior_haps = pickle.load(open(pathToPick + '/class_prior_haps.pkl', 'rb')) # structure {iter_gen: [hap_inter_matrix, unique_haplotypes, counts]}

        gen_Time_Value = list(class_prior_haps.keys())
        gen_Time_Value_index = range(len(gen_Time_Value))

        gen_Time_Value = gen_Time_Value[:]

        hd_within = np.zeros(length_RF+1)
        hd_between = np.zeros(length_RF+1)

        IPR_save = np.zeros(len(gen_Time_Value))
        size_subGroup = [0 for i in range(len(gen_Time_Value))]

        remained_non_sc_hap = np.zeros(len(gen_Time_Value))
        sc_hap = np.zeros(len(gen_Time_Value))
        all_remained_hap = np.zeros(len(gen_Time_Value))

        class_size = []

        rnase_ipr_non_sc_all = np.zeros(len(gen_Time_Value))
        rnase_ipr_classified = np.zeros(len(gen_Time_Value))

        slf_ipr_non_sc_all = np.zeros(len(gen_Time_Value))
        slf_ipr_classified = np.zeros(len(gen_Time_Value))

        sc_haplotypes_dict = {} # indices are based on the original haplotypes which include the sc haps
        non_sc_unclass_haps_dict = {} # indices are based on the original haplotypes which include the sc haps


        for gen_Time, index_top in zip(gen_Time_Value, gen_Time_Value_index):

            #   initial -
            #####################################
            fertilize_Group = class_prior_haps[gen_Time][0]
            hap_rnase_pollen = class_prior_haps[gen_Time][1]
            hap_count = class_prior_haps[gen_Time][2]
            unique_rnases_all = class_prior_haps[gen_Time][3][:,:length_RF]
            unique_slfs_degree_all = class_prior_haps[gen_Time][4]

            
            plt.figure(figsize = (16,7.5))
            cmap = sns.color_palette("deep", 2)
            plt.subplot(1,2,1)
            ax = sns.heatmap(fertilize_Group, cmap=['k','c'], rasterized=True, cbar=False, cbar_kws={"ticks": [i for i in range(2)]})#, linewidth=1.0)
            plt.ylabel('Male (ticks: number of distinct haplotypes)', fontsize=14)
            plt.xlabel('Female (ticks: number of distinct haplotypes)', fontsize=14)
            plt.yticks(range(len(hap_count)), hap_count.astype(int), rotation='horizontal', fontsize=5)
            plt.xticks(range(len(hap_count)), hap_count.astype(int), rotation='vertical', fontsize=5)
            ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, right=True, labelright=True)

            ax.set_title('a', loc='left', fontweight='bold')
            ax.set_title('before classification (cyan: compatible, black: incompatible)', loc='center', fontweight='bold')
            
            #plt.show()
            

            # unclassified SC haplotypes:
            #####################################
            sc_haplotypes = []
            for temp_i in range(len(hap_count)):
                if fertilize_Group[temp_i, temp_i] == 1:
                    sc_haplotypes.append(temp_i)#;   print ('yes')

            sc_hap[index_top] = sum(hap_count[sc_haplotypes])  # ----------------------------------------------- save data


            # unclassified non sc haplotypes [rnases and slfs]
            non_sc_unclassified_haplotypes = []   # ----------------------------------------------- save data

            ###
            non_sc_haplotypes = np.setdiff1d(range(len(hap_count)), sc_haplotypes) #list(range(25,len(hap_count_Temp))))

            if len(hap_count) > 0: # else all are SC
                # classification
                #####################################
                classified_haps, classified_haps_count, initial_class_id = {},  {}, 0   # ----------------------------------------------- save data
                for hap_count_top, hap_ind_top in zip(hap_count[non_sc_haplotypes], non_sc_haplotypes):

                    if hap_ind_top == 0:
                        classified_haps[initial_class_id] = [hap_ind_top]
                        classified_haps_count[initial_class_id] = [hap_count_top]

                    else:
                        earlier_classes = list(classified_haps.keys())

                        compatible_over_class = 0
                        incompatible_over_class = 0
                        #class_index_incompatible_with = 0
                        for each_class in list(classified_haps.keys()):

                            compatible_over_class_haps, incompatible_over_class_haps = 0, 0

                            for each_class_hap in classified_haps[each_class]:
                                if fertilize_Group[hap_ind_top, each_class_hap] == 1 \
                                                        and fertilize_Group[each_class_hap, hap_ind_top] == 1:
                                    compatible_over_class_haps += 1

                                elif fertilize_Group[hap_ind_top, each_class_hap] == 0 \
                                                        and fertilize_Group[each_class_hap, hap_ind_top] == 0:
                                    incompatible_over_class_haps += 1
                                else:
                                    'do nothing'

                            if compatible_over_class_haps == len(classified_haps[each_class]):
                                compatible_over_class += 1

                            elif incompatible_over_class_haps == len(classified_haps[each_class]):
                                incompatible_over_class += 1
                                class_index_incompatible_with = each_class

                            else:
                                'do nothing'

                        if compatible_over_class == len(list(classified_haps.keys())):
                            initial_class_id += 1 # new class
                            classified_haps[initial_class_id] = [hap_ind_top]
                            classified_haps_count[initial_class_id] = [hap_count_top]

                        elif compatible_over_class == len(list(classified_haps.keys())) - 1 and incompatible_over_class == 1: # part of one of the class
                            classified_haps[class_index_incompatible_with].append(hap_ind_top)
                            classified_haps_count[class_index_incompatible_with].append(hap_count_top)

                        else:
                            'this haplotype is unclassified'
                            non_sc_unclassified_haplotypes.append(hap_ind_top)

                #print (classified_haps, '\n', classified_haps_count, '\n')
                #####################################
                all_classifiec_haps = [each_hap for each_class_haps in list(classified_haps.values()) for each_hap in each_class_haps]
                hap_count_classified = hap_count[all_classifiec_haps]
                fertilize_Group_classified = fertilize_Group[np.ix_(all_classifiec_haps, all_classifiec_haps)]

                # HD-within and between
                #####################################
                for each_class_id_top in list(classified_haps.keys()):
                    all_haps_id_top = classified_haps[each_class_id_top]
                    all_rnase_in_this_class_top = hap_rnase_pollen[all_haps_id_top][:,0]

                    for each_rnase_top in all_rnase_in_this_class_top:

                        for each_class_id_bot in list(classified_haps.keys()):
                            all_haps_id_bot = classified_haps[each_class_id_bot]
                            all_rnase_in_this_class_bot = hap_rnase_pollen[all_haps_id_bot][:,0]


                            if each_class_id_top == each_class_id_bot: # HD-within ----------

                                for each_rnase_bot in all_rnase_in_this_class_bot:

                                    if each_rnase_top == each_rnase_bot:
                                        hd_within[0] += 1
                                    else:
                                        rnase_1 = unique_rnases_all[each_rnase_top]
                                        rnase_2 = unique_rnases_all[each_rnase_bot]
                                        hd_within[HammingDistance(rnase_1, rnase_2)] += 1   # ----------------------------------------------- save data

                            else:  # HD-between ----------

                                for each_rnase_bot in all_rnase_in_this_class_bot:

                                    rnase_1 = unique_rnases_all[each_rnase_top]
                                    rnase_2 = unique_rnases_all[each_rnase_bot]
                                    hd_between[HammingDistance(rnase_1, rnase_2)] += 1   # ----------------------------------------------- save data


                # unclassified non-SC haplotypes:
                #####################################
                unclassified_non_sc_haplotypes = np.setdiff1d(list(range(len(hap_count))), all_classifiec_haps)
                unclassified_non_sc_haplotypes = np.setdiff1d(unclassified_non_sc_haplotypes, sc_haplotypes)
                remained_non_sc_hap[index_top] = sum(hap_count[unclassified_non_sc_haplotypes])   # ----------------------------------------------- save data
                all_remained_hap[index_top] = remained_non_sc_hap[index_top] + sc_hap[index_top]

                sc_haplotypes_dict[gen_Time] = sc_haplotypes   # ----------------------------------------------- save data
                non_sc_unclass_haps_dict[gen_Time] = unclassified_non_sc_haplotypes   # ----------------------------------------------- save data


                # save data
                ####################################
                input_Var = np.array([sum(hap_count[each_class_haps]) for each_class_haps in list(classified_haps.values())]);
                #print (input_Var, sum(input_Var))
                for each_class_size in input_Var:   class_size.append(each_class_size)
                IPR_save[index_top] = find_ipr(input_Var)
                size_subGroup[index_top] = len(list(classified_haps.keys()))


                # unclassified rnases and slfs ipr
                fun_rnases_all = {} # sc hapotypes are removed
                fun_slfs_all = {} # sc hapotypes are removed

                ######
                # section for all slfs and rnases but sc are being removed
                all_hap_rnase_pollen  =  hap_rnase_pollen[non_sc_haplotypes]
                all_hap_rnase_pollen_count = hap_count[non_sc_haplotypes]

                for over_each_hap, over_each_hap_count in zip(all_hap_rnase_pollen, all_hap_rnase_pollen_count):
                    each_rnase_id = over_each_hap[0]
                    all_slfs_id = over_each_hap[1:]

                    if each_rnase_id in fun_rnases_all.keys():
                        fun_rnases_all[each_rnase_id] += over_each_hap_count
                    else:
                        fun_rnases_all[each_rnase_id] = over_each_hap_count


                    for over_each_slf in all_slfs_id:

                        if unique_slfs_degree_all[over_each_slf] != 0: # should be functional

                            if over_each_slf in fun_slfs_all.keys():
                                fun_slfs_all[over_each_slf] += over_each_hap_count
                            else:
                                fun_slfs_all[over_each_slf] = over_each_hap_count

                        else:
                            'this slfs in dyfunctional'

                sum_of_fun_rnases_all = sum(list(fun_rnases_all.values()))
                fun_rnase_all_ipr = np.round(find_ipr(np.array(list(fun_rnases_all.values()))), decimals=2)
                rnase_ipr_non_sc_all[index_top] = fun_rnase_all_ipr

                sum_of_fun_slfs_all = sum(list(fun_slfs_all.values()))
                fun_slf_all_ipr = np.round(find_ipr(np.array(list(fun_slfs_all.values()))), decimals=2)
                slf_ipr_non_sc_all[index_top] = fun_slf_all_ipr


                # classified  rnases and slfs ipr
                classified_fun_slfs ={} # sc hapotypes are removed
                classified_fun_rnases ={} # sc hapotypes are removed

                ######
                # section for all slfs and rnases but sc are being removed
                classified_hap_rnase_pollen  =  hap_rnase_pollen[all_classifiec_haps]
                classified_hap_rnase_pollen_count = hap_count[all_classifiec_haps]

                for over_each_hap, over_each_hap_count in zip(classified_hap_rnase_pollen, classified_hap_rnase_pollen_count):
                    each_rnase_id = over_each_hap[0]
                    all_slfs_id = over_each_hap[1:]

                    if each_rnase_id in classified_fun_rnases.keys():
                        classified_fun_rnases[each_rnase_id] += over_each_hap_count
                    else:
                        classified_fun_rnases[each_rnase_id] = over_each_hap_count


                    for over_each_slf in all_slfs_id:

                        if unique_slfs_degree_all[over_each_slf] != 0: # should be functional

                            if over_each_slf in classified_fun_slfs.keys():
                                classified_fun_slfs[over_each_slf] += over_each_hap_count
                            else:
                                classified_fun_slfs[over_each_slf] = over_each_hap_count

                        else:
                            'this slfs in dyfunctional'

                sum_of_classified_fun_rnases = sum(list(classified_fun_rnases.values()))
                #classified_rnase_ipr = np.round(find_ipr(np.array(list(classified_fun_rnases.values()))), decimals=2)
                classified_rnase_ipr = np.round(fun_rnase_all_ipr*sum_of_classified_fun_rnases/sum_of_fun_rnases_all, decimals=2)
                rnase_ipr_classified[index_top] = classified_rnase_ipr

                sum_of_classified_fun_slfs = sum(list(classified_fun_slfs.values()))
                #classified_slf_ipr = np.round(find_ipr(np.array(list(classified_fun_slfs.values()))), decimals=2)
                classified_slf_ipr = np.round(fun_slf_all_ipr*sum_of_classified_fun_slfs/sum_of_fun_slfs_all, decimals=2)
                slf_ipr_classified[index_top] = classified_slf_ipr

                #input_Var = np.array([sum(hap_count[each_class_haps]) for each_class_haps in list(classified_haps.values())]);

                
                cmap = sns.color_palette("deep", 2)
                plt.subplot(1,2,2)
                ax = sns.heatmap(fertilize_Group_classified, cmap=['k','c'], rasterized=True, cbar=False, cbar_kws={"ticks": [i for i in range(2)]})#, linewidth=1.0)
                plt.ylabel('Male (ticks: number of distinct haplotypes)', fontsize=14)
                plt.xlabel('Female (ticks: number of distinct haplotypes)', fontsize=14)
                plt.yticks(range(len(hap_count_classified)), hap_count_classified.astype(int), rotation='horizontal', fontsize=5)
                plt.xticks(range(len(hap_count_classified)), hap_count_classified.astype(int), rotation='vertical', fontsize=5)
                ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, right=True, labelright=True)
                ax.set_title('b', loc='left', fontweight='bold')
                ax.set_title('after classification', loc='center', fontweight='bold')

                plt.tight_layout()
                plt.savefig(pathToSave + '/classified_haplotypes_{}.pdf'.format(gen_Time))
                plt.show()
                plt.close()
                

        if len(gen_Time_Value) > 1:
            np.savetxt(pathToSave + '/ipr_class_no.dat', IPR_save, fmt='%0.2f')
            np.savetxt(pathToSave + '/class_no.dat', size_subGroup, fmt='%i')
            np.savetxt(pathToSave + '/unclass_non_sc_haps.dat', remained_non_sc_hap, fmt='%i')
            np.savetxt(pathToSave + '/unclass_sc_haps.dat', sc_hap, fmt='%i')
            np.savetxt(pathToSave + '/unclass_all_haps.dat', all_remained_hap, fmt='%i')
            np.savetxt(pathToSave + '/class_size.dat', class_size, fmt='%i')
            np.savetxt(pathToSave + '/hd_within.dat', hd_within, fmt='%i')
            np.savetxt(pathToSave + '/hd_between.dat', hd_between, fmt='%i')

            sc_haplotypes_dict_Save = open(pathToSave+'/unclass_sc_haps_dict_indices.pkl', "wb");
            pickle.dump(sc_haplotypes_dict, sc_haplotypes_dict_Save);
            sc_haplotypes_dict_Save.close()

            non_sc_unclass_haps_dict_Save = open(pathToSave+'/unclass_non_sc_haps_dict_indices.pkl', "wb");
            pickle.dump(non_sc_unclass_haps_dict, non_sc_unclass_haps_dict_Save);
            non_sc_unclass_haps_dict_Save.close()

            np.savetxt(pathToSave + '/RNAse_ipr_non_sc_all.dat', rnase_ipr_non_sc_all, fmt='%0.2f')
            np.savetxt(pathToSave + '/RNAse_ipr_classified.dat', rnase_ipr_classified, fmt='%0.2f')
            np.savetxt(pathToSave + '/SLF_ipr_non_sc_all.dat', slf_ipr_non_sc_all, fmt='%0.2f')
            np.savetxt(pathToSave + '/SLF_ipr_classified.dat', slf_ipr_classified, fmt='%0.2f')

    else:
        print (pathToPick, ', data did not find')


    print ('\n')
    #print (rnase_ipr_non_sc_all)
    #print (slf_ipr_non_sc_all)


    #print ('\n')
    #print (rnase_ipr_classified)
    #print (slf_ipr_classified, '\n')

    print (SampleNo, energyThreshold)

    return 'done'

if __name__ == '__main__':

    N_Proces = 1
    typeOfGene  = 'SLF' # SLF/RNAse
    Sample_Size = range(1)
    alpha, delta = 0.90, 0.90 # self pollen (selfing), inbreeding depression

    threshold   = [-10, -9, -8, -7, -6, -5, -4, -3, -2][0:1]
    main_path = './variable_ene_al_dl'

    pick_path  = ['/fig_E'+str(th_Value) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100)) for th_Value in threshold]
    save_path  = ['/fig_E'+str(th_Value) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100)) for th_Value in threshold]

    Samples     = [i for th_Value in threshold for i in Sample_Size]
    Threshold   = [th_Value for th_Value in threshold for SampleNo in Sample_Size]

    PathToPick  = [main_path + each_pick_path + '/data_{}'.format(SampleNo) for each_pick_path in pick_path for SampleNo in Sample_Size]
    PathToSave  = [main_path + each_save_path + '/data_{}'.format(SampleNo) for each_save_path in save_path for SampleNo in Sample_Size]

    TypeOfGene  = [typeOfGene for th_Value in threshold for SampleNo in Sample_Size]

    Pool(N_Proces).map(fertilize_Class, zip(Samples, Threshold, PathToPick, PathToSave, TypeOfGene))





def merge_fertilize_Class(input_Var):
    SampleNo, energyThreshold, pathToPick, pathToSave, typeOfGene = input_Var
    print (SampleNo, energyThreshold, pathToPick, pathToSave, typeOfGene, '\n')


    if energyThreshold in [-10, -9, -8, -7, -6, -5]:
        tot_Samples = 1
    if energyThreshold in [-4, -3, -2]:
        tot_Samples = 1
    
    tot_Samples = 1

    dict_ipr_class = {}
    dict_class_no = {}
    dict_unclass_non_sc_haps = {}
    dict_unclass_sc_haps = {}
    dict_unclass_all_haps = {}
    dict_class_size = {}

    hd_within = []
    hd_between = []

    rnase_ipr_non_sc_all = []
    rnase_ipr_classified = []
    slf_ipr_non_sc_all = []
    slf_ipr_classified = []

    for SampleNo in range(tot_Samples):

        if (os.path.exists(pathToPick + '/data_{}'.format(SampleNo) + '/ipr_class_no.dat')):
            data = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/ipr_class_no.dat')
            x_data, y_data = np.unique(data, axis=None, return_counts=True)
            for each_key, each_value in zip(x_data, y_data):
                if each_key in dict_ipr_class.keys():
                    dict_ipr_class[each_key] += each_value
                else:
                    dict_ipr_class[each_key] = each_value

            data = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/class_no.dat')
            x_data, y_data = np.unique(data, axis=None, return_counts=True)
            for each_key, each_value in zip(x_data, y_data):
                if each_key in dict_class_no.keys():
                    dict_class_no[each_key] += each_value
                else:
                    dict_class_no[each_key] = each_value

            data = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/unclass_non_sc_haps.dat')
            x_data, y_data = np.unique(data, axis=None, return_counts=True)
            for each_key, each_value in zip(x_data, y_data):
                if each_key in dict_unclass_non_sc_haps.keys():
                    dict_unclass_non_sc_haps[each_key] += each_value
                else:
                    dict_unclass_non_sc_haps[each_key] = each_value

            data = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/unclass_sc_haps.dat')
            x_data, y_data = np.unique(data, axis=None, return_counts=True)
            for each_key, each_value in zip(x_data, y_data):
                if each_key in dict_unclass_sc_haps.keys():
                    dict_unclass_sc_haps[each_key] += each_value
                else:
                    dict_unclass_sc_haps[each_key] = each_value

            data = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/unclass_all_haps.dat')
            x_data, y_data = np.unique(data, axis=None, return_counts=True)
            for each_key, each_value in zip(x_data, y_data):
                if each_key in dict_unclass_all_haps.keys():
                    dict_unclass_all_haps[each_key] += each_value
                else:
                    dict_unclass_all_haps[each_key] = each_value

            data = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/class_size.dat')
            x_data, y_data = np.unique(data, axis=None, return_counts=True)
            for each_key, each_value in zip(x_data, y_data):
                if each_key in dict_class_size.keys():
                    dict_class_size[each_key] += each_value
                else:
                    dict_class_size[each_key] = each_value

            hd_w = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/hd_within.dat')
            hd_within.append(hd_w)

            hd_b = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/hd_between.dat')
            hd_between.append(hd_b)


            data_rnase_ipr_non_sc_all = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/RNAse_ipr_non_sc_all.dat')
            data_rnase_ipr_classified = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/RNAse_ipr_classified.dat')
            data_slf_ipr_non_sc_all = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/SLF_ipr_non_sc_all.dat')
            data_slf_ipr_classified = np.genfromtxt(pathToPick + '/data_{}'.format(SampleNo) + '/SLF_ipr_classified.dat')


            rnase_ipr_non_sc_all.append(data_rnase_ipr_non_sc_all)
            rnase_ipr_classified.append(data_rnase_ipr_classified)
            slf_ipr_non_sc_all.append(data_slf_ipr_non_sc_all)
            slf_ipr_classified.append(data_slf_ipr_classified)




        else:
            print (pathToPick + '/data_{}'.format(SampleNo), 'did not find the file')


    #
    ### ----------------
    dict_data = [dict_ipr_class, \
                dict_class_no, \
                dict_unclass_non_sc_haps, \
                dict_unclass_sc_haps, \
                dict_unclass_all_haps, \
                dict_class_size]

    file_data = ['ipr_class_no',\
                'class_no',\
                'unclass_non_sc_haps',\
                'unclass_sc_haps',\
                'unclass_all_haps', \
                'class_size']

    #
    ##
    for each_dict_data, each_file_data in zip(dict_data, file_data):
        dict_keys, dict_values = np.array(list(each_dict_data.keys())), np.array(list(each_dict_data.values()))
        sorted_index = np.argsort(dict_keys)
        dict_keys_sorted, dict_values_sorted = dict_keys[sorted_index], dict_values[sorted_index]
        if each_file_data == 'ipr_class_no':
            fmt_save = '%0.2f'
        else:
            fmt_save = '%i'
        np.savetxt(pathToSave + '/{}.dat'.format(each_file_data), np.concatenate(([dict_keys_sorted], [dict_values_sorted]), axis=0).T, fmt=fmt_save)

    hd_within = np.array(hd_within)
    hd_between = np.array(hd_between)

    np.savetxt(pathToSave + '/hd_within.dat', np.mean(hd_within, axis=0)/sum(np.mean(hd_within, axis=0)), fmt='%0.7f')
    np.savetxt(pathToSave + '/hd_between.dat', np.mean(hd_between, axis=0)/sum(np.mean(hd_between, axis=0)), fmt='%0.7f')

    np.savetxt(pathToSave + '/RNAse_ipr_non_sc_all.dat', np.array(rnase_ipr_non_sc_all).flatten(), fmt='%0.2f')
    np.savetxt(pathToSave + '/RNAse_ipr_classified.dat', np.array(rnase_ipr_classified).flatten(), fmt='%0.2f')
    np.savetxt(pathToSave + '/SLF_ipr_non_sc_all.dat', np.array(slf_ipr_non_sc_all).flatten(), fmt='%0.2f')
    np.savetxt(pathToSave + '/SLF_ipr_classified.dat', np.array(slf_ipr_classified).flatten(), fmt='%0.2f')


    print (np.shape(hd_within))

    return 'done'

if __name__ == '__main__':

    typeOfGene = 'RNAse' # RNAse/RNAse_hap/SLF/Hap
    Sample_Size = range(1)
    alpha, delta = 0.90, 0.90 # self pollen (selfing), inbreeding depression

    threshold   = [-10, -9, -8, -7, -6, -5, -4, -3, -2][:]
    main_path = './variable_ene_al_dl'

    pick_path  = ['/fig_E'+str(th_Value) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100)) for th_Value in threshold]
    save_path  = ['/fig_E'+str(th_Value) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100)) for th_Value in threshold]

    Samples     = [i for th_Value in threshold for i in Sample_Size]
    Threshold   = [th_Value for th_Value in threshold for SampleNo in Sample_Size]

    PathToPick  = [main_path + each_pick_path for each_pick_path in pick_path for SampleNo in Sample_Size]
    PathToSave  = [main_path + each_save_path for each_save_path in save_path for SampleNo in Sample_Size]

    TypeOfGene  = [typeOfGene for th_Value in threshold for SampleNo in Sample_Size]

    #Pool(N_Proces).map(merge_fertilize_Class, zip(Samples, Threshold, PathToPick, PathToSave, TypeOfGene))









def temp_plot(input_Var):
    k   = ['E-2_a90_d75', 'E-2_a90_d90', 'E-6_a90_d90', 'E-6_a90_d75', 'E-10_a90_d90', 'E-10_a90_d75'][0:2]
    k = ['E-2_a65_d65', 'E-2_a60_d60', 'E-6_a65_d65', 'E-6_a60_d60'][0:2]
    PathToSave  = './fertilize/weight_hd'
    color = ['b', 'g', 'r', 'y', 'k', 'm']
    title = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i']
    plt.figure(figsize=(10,8.5))


    ax = plt.subplot(2,2,3)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    Samples = 15
    for p_k, index in zip(k, range(6)):
        data_to_plot = []
        for i in range(Samples):
            data = np.genfromtxt(PathToSave + '/class_size_k{}_sam_{}.dat'.format(p_k, i))
            for each_data in data:  data_to_plot.append(each_data)
        data_unique, data_count = np.unique(data_to_plot, axis=None, return_counts=True)
        plt.plot(data_unique, data_count/sum(data_count), '-o', color=color[index], label=k[index])
        average = np.round(sum(data_unique*(data_count/sum(data_count))), decimals=3)
        plt.plot([average, average], [0.0, 0.010], '--', color=color[index])

    plt.ylabel('Fraction', fontsize=10);    plt.xlabel(r'Class size', fontsize=10)
    plt.yticks(fontsize=10);    plt.xticks(fontsize=10)
    plt.legend(ncols=1, frameon=False, fontsize=8)
    plt.title(title[2], loc='left', fontweight='bold')


    ax = plt.subplot(2,2,4)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    Samples = 15
    for p_k, index in zip(k, range(6)):
        data_to_plot = []
        for i in range(Samples):
            data = np.genfromtxt(PathToSave + '/unclass_sc_haps_k{}_sam_{}.dat'.format(p_k, i))
            data_to_plot.append(data)
        data_unique, data_count = np.unique(data_to_plot, axis=None, return_counts=True)
        plt.plot(data_unique/2000, data_count/sum(data_count), '-o', color=color[index], label=k[index])
        average = np.round(sum((data_unique/2000)*(data_count/sum(data_count))), decimals=3)
        plt.plot([average, average], [0.0, 0.010], '--', color=color[index])

    plt.ylabel('Fraction', fontsize=10);    plt.xlabel(r'fraction of unclassified SC haplotypes', fontsize=10)
    plt.yticks(fontsize=10);    plt.xticks(fontsize=10)
    plt.legend(ncols=1, frameon=False, fontsize=8)
    plt.title(title[2], loc='left', fontweight='bold')


    ax = plt.subplot(2,2,1)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    Samples = 15
    for p_k, index in zip(k, range(6)):
        data_to_plot = []
        for i in range(Samples):
            data = np.genfromtxt(PathToSave + '/class_no_k{}_sam_{}.dat'.format(p_k, i))
            data_to_plot.append(data)
        data_unique, data_count = np.unique(data_to_plot, axis=None, return_counts=True)
        plt.plot(data_unique, data_count/sum(data_count), '-o', color=color[index], label=k[index])

    plt.ylabel('Fraction', fontsize=10);    plt.xlabel(r'# classes', fontsize=10)
    plt.yticks(fontsize=10);    plt.xticks(fontsize=10)
    plt.legend(ncols=1, frameon=False, fontsize=8)
    plt.title(title[0], loc='left', fontweight='bold')

    ax = plt.subplot(2,2,2)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    Samples = 15
    for p_k, index in zip(k, range(6)):
        data_to_plot = []
        for i in range(Samples):
            data1 = np.genfromtxt(PathToSave + '/unclass_sc_haps_k{}_sam_{}.dat'.format(p_k, i))
            data2 = np.genfromtxt(PathToSave + '/unclass_non_sc_haps_k{}_sam_{}.dat'.format(p_k, i))
            data_to_plot.append(data1+data2)
        data_unique, data_count = np.unique(data_to_plot, axis=None, return_counts=True)
        plt.plot(data_unique/2000, data_count/sum(data_count), '-o', color=color[index], label=k[index])
        average = np.round(sum((data_unique/2000)*(data_count/sum(data_count))), decimals=3)
        plt.plot([average, average], [0.0, 0.025], '--', color=color[index])

    plt.ylabel('Fraction', fontsize=10);    plt.xlabel(r'Fraction of unclassified haplotypes', fontsize=10)
    plt.yticks(fontsize=10);    plt.xticks(fontsize=10)
    plt.legend(ncols=1, frameon=False, fontsize=8)
    plt.title(title[1], loc='left', fontweight='bold')

    plt.tight_layout()
    #plt.savefig('./figures/class_structure_eth{}.pdf'.format(k[0][1:3]))
    plt.show()

    return 'done'

#print (temp_plot(1))


print ('Ends|----|H:M:S|{}'.format(datetime.now() - start_time), '\n')
###############################################################################################################
###############################################################################################################
