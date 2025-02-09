#
#		SLF-RNAse interaction in Non-self based self-incompatible (SI) System
#
#
##########################################################################################

#from mpl_toolkits.mplot3d import Axes3D # type: ignore
#import pylab
#import sympy as sp
#from operator import itemgetter # type: ignore
#from scipy import linalg # type: ignore
#from pylab import genfromtxt # type: ignore
#import fit
#import networkx as nx
from multiprocessing import Pool#, TimeoutError, Process, Queue, Pipe, Value, Array # type: ignore
import os, math, pdb, time, random as ra, numpy as np, sys
#import collections, pickle
from datetime import datetime # type: ignore
#from numba import jit
import pickle 
#import seaborn as sns; sns.set()
#from scipy.integrate import odeint
#from scipy.signal import argrelextrema, find_peaks
#from scipy import signal
########################################################################################################
start_time = datetime.now()
print ('Starts--------', start_time)


#
#
###############################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------
def cnsr_evolution(input_Values):
    start_time_01 = datetime.now()
    # index, mutation probability, to save the file, total population
    start_time_here, generations, generations_window_change, window_Size_Last, \
                    SampleNo, energyThreshold, prob_Mut, folder_to_Save, \
                            alpha_input, delta_input, random_seed, numpy_seed, population_in, no_attempts_in = input_Values

    ra.seed(random_seed)
    np.random.seed(numpy_seed)

    mainPath = folder_to_Save
    pathToSave = mainPath+'/data_{}'.format(SampleNo)
    #
    #
    #--------------------------------------------------------
    distinct_RNAse = 10                     # number of initial distinct haplotype
    length_RF = 18                          # length of RNAse and F box genes
    weight = [0.5, 0.265, 0.113, 0.122]     # weight of four types of Amino Acid (AA) occurance
    fourTypesAA = [0,1,2,3]
    window_Size = 500
    alpha = alpha_input # self pollen (selfing)
    delta = delta_input # inbreeding depression
    window_Size_Last = window_Size_Last
    tot_Population = population_in
    no_attempts = no_attempts_in
    # energy of interacting pair; one RNAse and another F box gene
    energy_interaction = np.array([[-1,0,0,0],[0,-0.75,-0.25,-0.25],[0,-0.25,1,-1.25],[0,-0.25,-1.25,1]])
    def energyOfIntPair(RNAse, FBoxg):  return sum([energy_interaction[x,y] for x, y in zip(RNAse.astype(int), FBoxg.astype(int))])

    #
    #
    #
    # initialization of the population                                                                                          BLOCK-I--------------------
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    popRNAse, popFbox = [0 for i in range(tot_Population)], [0 for i in range(tot_Population)]         # parental population
    tempRNAse, tempFbox = [0 for i in range(distinct_RNAse)], [0 for i in range(distinct_RNAse)]
    SLF_on_Each_Hap = []


    #
    #
    # measures ------------------------------------------------------------------------------------------
    #RNAse_dict_all_gen  = {}
    if start_time_here == 0:
        for line, index_ini in zip(open('./{}/initial/SLF_initial_sam_{}.dat'.format(folder_to_Save, SampleNo)), range(distinct_RNAse)):
            check = [int(value) for value in line.split()]
            tempFbox[index_ini] = np.array(check)

        for line, index_ini in zip(open('./{}/initial/RNAse_initial_sam_{}.dat'.format(folder_to_Save, SampleNo)), range(distinct_RNAse)):
            check = [int(value) for value in line.split()]
            tempRNAse[index_ini] = np.array(check)

        data = np.genfromtxt('./{}/initial/SLF_no_initial_sam_{}.dat'.format(folder_to_Save, SampleNo))
        no_Of_SLF_Each_haplotype = [int(value) for value in data]
        print (no_Of_SLF_Each_haplotype, SampleNo, '\n')

        Ancestral_Array_initial = []
        integerCal = int(tot_Population/distinct_RNAse)
        for i, SLF_Hap in zip(range(distinct_RNAse), no_Of_SLF_Each_haplotype):
            for population in range(integerCal):
                popRNAse[i*integerCal+population] = tempRNAse[i]
                popFbox[i*integerCal+population] = tempFbox[i]
                SLF_on_Each_Hap.append(SLF_Hap)
                Ancestral_Array_initial.append(i)

        # measure -------------- unique RNAse and SLF frequency --- for every generations
        #KEY, KEY_index, KEY_inverse, VALUES = np.unique(popRNAse, axis=0, return_index=True, return_counts=True, return_inverse=True)
        unique_RNAse_value, unique_RNAse_inverse, unique_RNAse_fre = np.unique(popRNAse, axis=0, return_inverse=True, return_counts=True)
        #RNAse_dict_all_gen[start_time_here] = [unique_RNAse_value, unique_RNAse_inverse, unique_RNAse_fre]
        np.savetxt(pathToSave+'/unique_RNAse_gen_{}.dat'.format(start_time_here), np.concatenate((unique_RNAse_value, np.array([unique_RNAse_fre]).T), axis=1), fmt='%0.0f')   # initial unique RNAse
        np.savetxt(pathToSave+'/inverse_of_RNAse_{}.dat'.format(start_time_here), np.array(unique_RNAse_inverse).T, fmt='%0.0f')   # initial RNAse inverse
        np.savetxt(pathToSave+'/ancestral_Array_{}.dat'.format(start_time_here), np.array(Ancestral_Array_initial).T, fmt='%0.0f')  # initial ancestar

        all_SLF = [each_Pollen[index*length_RF:length_RF*(index+1)] for each_Pollen in popFbox for index in range(int(len(each_Pollen)/length_RF))]
        unique_SLF_value, unique_SLF_fre = np.unique(all_SLF, axis=0, return_counts=True)

        no_of_Unique_RNAse, no_of_Unique_SLF = len(unique_RNAse_value), len(unique_SLF_value)

        Functionality_SLF = [0 for i in range(no_of_Unique_SLF)]
        index_Interacting_RNAse_with_Unique_SLF = []
        for Each_unique_SLF_value, Each_unique_SLF_value_index in zip(unique_SLF_value, range(no_of_Unique_SLF)):
            SLF_degree = 0
            index_Interacting_RNAse_with_each_SLF = []
            for Each_unique_RNAse_value, Each_unique_RNAse_fre, Each_unique_RNAse_index in zip(unique_RNAse_value, unique_RNAse_fre, range(no_of_Unique_RNAse)):
                if energyOfIntPair(Each_unique_RNAse_value, Each_unique_SLF_value) < energyThreshold:
                    SLF_degree += Each_unique_RNAse_fre
                    index_Interacting_RNAse_with_each_SLF.append(Each_unique_RNAse_index)
                else:
                    index_Interacting_RNAse_with_each_SLF.append(-1)
            index_Interacting_RNAse_with_Unique_SLF.append(np.array(index_Interacting_RNAse_with_each_SLF))
            Functionality_SLF[Each_unique_SLF_value_index] = SLF_degree
        final_SLF_To_Save = np.concatenate((unique_SLF_value, np.array([unique_SLF_fre]).T, np.array([np.array(Functionality_SLF)]).T), axis=1)
        final_SLF_To_Save = np.concatenate((final_SLF_To_Save, np.array(index_Interacting_RNAse_with_Unique_SLF)), axis=1)
        np.savetxt(pathToSave+'/unique_SLF_gen_{}.dat'.format(start_time_here), final_SLF_To_Save, fmt='%0.0f')   # initial unique SLF

        # measure ------
        popFbox_unique, popFbox_unique_inverse, popFbox_unique_count = np.unique(popFbox, axis=0, return_inverse=True, return_counts=True)
        SLF_nature_Each_Hap_to_Save = [0 for i in range(len(popFbox_unique_count))]
        for pollen_Here, pollen_Here_index in zip(popFbox_unique, range(len(popFbox_unique_count))):
            no_Of_SLFgenes = int(len(pollen_Here)/length_RF) #+ 1 # number of SLF genes on that haplotype
            SLF_at_pollen = [pollen_Here[ii*length_RF:(ii+1)*length_RF] for ii in range(no_Of_SLFgenes)]
            #SLF_value, SLF_inverse, SLF_fre = np.unique(SLF_at_pollen, axis=0, return_inverse=True, return_counts=True)
            SLF_nature, SLF_index = [], []
            #for SLF_value_current in SLF_value:
            for SLF_value_current in SLF_at_pollen:
                for Each_unique_SLF_value, Each_unique_SLF_value_index in zip(unique_SLF_value, range(no_of_Unique_SLF)):
                    if np.array_equal(SLF_value_current, Each_unique_SLF_value) is True:
                        SLF_nature.append(Functionality_SLF[Each_unique_SLF_value_index]);
                        SLF_index.append(Each_unique_SLF_value_index);
                        break
            #SLF_nature_Each_Hap_to_Save[pollen_Here_index] = [SLF_index, SLF_inverse, SLF_fre, SLF_nature]
            SLF_nature_Each_Hap_to_Save[pollen_Here_index] = [SLF_index, SLF_nature]

        SLF_Save = open(pathToSave+'/SLF_nature_gen_{}.pkl'.format(start_time_here), "wb");   pickle.dump(SLF_nature_Each_Hap_to_Save, SLF_Save); SLF_Save.close()
        np.savetxt(pathToSave+'/SLF_nature_inv_gen_{}.dat'.format(start_time_here), np.array(popFbox_unique_inverse).T, fmt='%i')
        np.savetxt(pathToSave+'/SLF_nature_counts_gen_{}.dat'.format(start_time_here), np.array(popFbox_unique_count).T, fmt='%i')

        haplotype_Nat = np.array([0 for i in range(tot_Population)]) # 0 means SI, 1 means SC --- in each generation it will be updated
        for each_temp_rnase, each_fbox, ind_i in zip(popRNAse, popFbox, range(tot_Population)):
            all_current_SLFs = [each_fbox[ii*length_RF:(ii+1)*length_RF] for ii in range(no_Of_SLFgenes)]
            sc_test = 0
            for each_current_SLF in all_current_SLFs:
                if energyOfIntPair(each_temp_rnase, each_current_SLF) < energyThreshold:
                    sc_test = 1
                    print ('yes', energyOfIntPair(each_temp_rnase, each_current_SLF))
                    break
            haplotype_Nat[ind_i] = sc_test

    else:

        'do nothing'
        # the section to resatrt the code where it was stooped


    #
    #
    #
    #
    #
    #
    #
    # ----------------------------------------------------------------------------------------------------------------------------
    tempPopRNAse, tempPopFbox, tempSLF_on_Each_Hap = [0 for i in range(tot_Population)], [0 for i in range(tot_Population)], [0 for i in range(tot_Population)]
    current_ancesteral = Ancestral_Array_initial
    is_all_sc = False
    is_all_sc_gen_Time = 0
    is_single_ancestor = False



    def female_probability(input_Var):
        alpha, delta, fs, fi = input_Var
        fs, fi = fs/(fs+fi), fi/(fs+fi)

        fssp = fs * alpha * (1 - delta) / (1 - fs * alpha * delta)
        fsnp = fs * (1 - alpha) / (1 - fs * alpha * delta)
        fip  = fi / (1 - fs * alpha * delta)

        output = [fssp, fsnp, fip]
        output = np.array(output)/sum(output)

        return output

    np.savetxt(pathToSave+'/hap_Nature_{}.dat'.format(start_time_here), np.array(haplotype_Nat).T, fmt='%0.0f')

    gen_Time = start_time_here + 1
    while gen_Time < generations+1:

                                                                                 # Mutation------------------------------------------
        # mutation ----------- Input non-mutated population and output as mutated population ----------------------------------------
        tempPopRNAse, tempPopFbox, tempSLF_on_Each_Hap = [0 for i in range(tot_Population)], [0 for i in range(tot_Population)], [0 for i in range(tot_Population)]

        SC_Attempts = [-1 for i in range(tot_Population)]
        haplotype_Nat_temp = np.array([0 for i in range(tot_Population)])
        for pop_Index in range(tot_Population):
            no_Of_SLFgenes = SLF_on_Each_Hap[pop_Index] #int(length_Of_AA/length_RF) # + 1 # number of SLF genes on that haplotype
            length_Of_AA = no_Of_SLFgenes*length_RF # by default it will be integer
            '''
            RNAse_prob_Seq = np.array([1 if random_AA < prob_Mut else 0 for random_AA in np.random.random(length_RF)])
            temp_Mut_RNAse = RNAse_prob_Seq*np.random.choice(4, length_RF, p=weight) + popRNAse[pop_Index]*(1-RNAse_prob_Seq)
            Fbox_prob_Seq  = np.array([1 if random_AA < prob_Mut else 0 for random_AA in np.random.random(length_Of_AA)])
            temp_Mut_Fbox  = Fbox_prob_Seq*np.random.choice(4, length_Of_AA, p=weight) + popFbox[pop_Index]*(1-Fbox_prob_Seq)
            '''
            tracker_RNAse_mut, tracker_SLF_mut = 0, 0
            RNAse_prob_Seq = np.array([1 if random_AA < prob_Mut else 0 for random_AA in np.random.random(length_RF)])
            if np.any(RNAse_prob_Seq==1):
                temp_Mut_RNAse = RNAse_prob_Seq*np.random.choice(4, length_RF, p=weight) + popRNAse[pop_Index]*(1-RNAse_prob_Seq)
                popRNAse[pop_Index] = temp_Mut_RNAse
                tracker_RNAse_mut = 1
            else:
                temp_Mut_RNAse = popRNAse[pop_Index]
                popRNAse[pop_Index] = temp_Mut_RNAse
                'basically do nothing'

            Fbox_prob_Seq  = np.array([1 if random_AA < prob_Mut else 0 for random_AA in np.random.random(length_Of_AA)])
            if np.any(Fbox_prob_Seq == 1):
                temp_Mut_Fbox = Fbox_prob_Seq*np.random.choice(4, length_Of_AA, p=weight) + popFbox[pop_Index]*(1-Fbox_prob_Seq)
                popFbox[pop_Index] = temp_Mut_Fbox
                tracker_SLF_mut = 1
            else:
                temp_Mut_Fbox = popFbox[pop_Index]
                popFbox[pop_Index]  = temp_Mut_Fbox
                'basically do nothing'

            # check self-compatibility
            check_self_compatibility = 0 # 0: self incompatible (SI), 1: self-compatible (SC)
            for SLF_mutated in [temp_Mut_Fbox[ii*length_RF:(ii+1)*length_RF] for ii in range(no_Of_SLFgenes)]:
                if energyOfIntPair(temp_Mut_RNAse, SLF_mutated) < energyThreshold:
                    check_self_compatibility = 1
                    break
            haplotype_Nat[pop_Index] = check_self_compatibility

                                                                          # Selection------------------------------------------

        # selection -------------Input non-selected population and output as slelected population -----------------------------------
        next_gen_ancesteral = [-1 for i in range(tot_Population)]
        no_of_Attempts = [0 for i in range(tot_Population)]
        hap_Replacement = [ix for ix in range(tot_Population)]
        #alpha_array = np.random.random(tot_Population)
        #alpha_array = np.where(alpha_array < alpha, 1, 0)
        #alpha_haplotype = alpha_array*haplotype_Nat  # 0 means there is outcrossing for every SI and SC,,,, 1 means selfing
        #selfing_hap_Index = np.where(alpha_haplotype==1)[0] # index where selfing will occur -- these are SC haplotype always

        SC_hap_indices = np.where(haplotype_Nat==1)[0] # sc
        SI_hap_indices = np.where(haplotype_Nat==0)[0] # si

        hap_nat_unique, hap_nat_unique_count = np.unique(haplotype_Nat, axis=None, return_counts=True)
        if len(hap_nat_unique) == 2:
            fraction_SI = hap_nat_unique_count[0]/sum(hap_nat_unique_count)
            fraction_SC = hap_nat_unique_count[1]/sum(hap_nat_unique_count)

        else:
            if hap_nat_unique[0] == 0:
                fraction_SC = 0.0
                fraction_SI = 1.0
            else:
                fraction_SC = 1.0
                fraction_SI = 0.0

        choosing_female_hap_prob = female_probability([alpha, delta, fraction_SC, fraction_SI]) #alpha, delta, fs, fi
        haps_to_choose = np.random.choice([0,1,2], tot_Population, p=choosing_female_hap_prob)

        for pop_Index, hap_type_to_choose in zip(range(tot_Population), haps_to_choose):

            is_female_pollinated = False # 0 not is_female_pollinated, 1 is_female_pollinated
            while is_female_pollinated == False:

                if hap_type_to_choose == 0:
                    female_new_pop_Index = np.random.choice(SC_hap_indices)
                if hap_type_to_choose == 1:
                    female_new_pop_Index = np.random.choice(SC_hap_indices)
                if hap_type_to_choose == 2:
                    female_new_pop_Index = np.random.choice(SI_hap_indices)

                if hap_type_to_choose == 1 or hap_type_to_choose == 2:

                    #is_female_pollinated = 0 # 0 not is_female_pollinated, 1 is_female_pollinated
                    for ij in range(no_attempts):
                        random_hap_pollen = ra.randint(0,tot_Population-1)
                        while random_hap_pollen == female_new_pop_Index:  random_hap_pollen = ra.randint(0,tot_Population-1)
                        RNAse_current, pollen_Selected = popRNAse[female_new_pop_Index], popFbox[random_hap_pollen] # current RNA from popRNA and random SLF from popFbox (parent)

                        no_Of_SLFgenes_Selected = SLF_on_Each_Hap[random_hap_pollen] # number of SLF genes on that haplotype
                        #length_Of_AA_Selected = int(no_Of_SLFgenes_Selected*length_RF)

                        # checking the pollination
                        for SLF_selected in [pollen_Selected[iix*length_RF:(iix+1)*length_RF] for iix in range(no_Of_SLFgenes_Selected)]:
                            if energyOfIntPair(RNAse_current, SLF_selected) < energyThreshold:
                                is_female_pollinated = True; break

                        if is_female_pollinated == True:
                            no_of_Attempts[pop_Index] = ij+1
                            if ra.random() < 0.5:
                                tempPopRNAse[pop_Index] = popRNAse[random_hap_pollen]   # redraw the population from previous population
                                tempPopFbox[pop_Index] = popFbox[random_hap_pollen]     # redraw the population from previous population
                                tempSLF_on_Each_Hap[pop_Index] = SLF_on_Each_Hap[random_hap_pollen]
                                next_gen_ancesteral[pop_Index] = current_ancesteral[random_hap_pollen]
                                hap_Replacement[pop_Index] = random_hap_pollen
                                haplotype_Nat_temp[pop_Index] = haplotype_Nat[random_hap_pollen]
                            else:
                                tempPopRNAse[pop_Index] = popRNAse[female_new_pop_Index]  # redraw the population from previous population
                                tempPopFbox[pop_Index] = popFbox[female_new_pop_Index]     # redraw the population from previous population
                                tempSLF_on_Each_Hap[pop_Index] = SLF_on_Each_Hap[female_new_pop_Index]
                                next_gen_ancesteral[pop_Index] = current_ancesteral[female_new_pop_Index]
                                hap_Replacement[pop_Index] = female_new_pop_Index
                                haplotype_Nat_temp[pop_Index] = haplotype_Nat[female_new_pop_Index]
                            break # to break the if is_female_pollinated == 1 and for loop of attempts

                if hap_type_to_choose == 0:
                    is_female_pollinated = True # pollinated by self, just for the while loop
                    no_of_Attempts[pop_Index] = 0 # no attempts means selfing
                    tempPopRNAse[pop_Index] = popRNAse[female_new_pop_Index]   # redraw the population from previous population
                    tempPopFbox[pop_Index] = popFbox[female_new_pop_Index]    # redraw the population from previous population
                    tempSLF_on_Each_Hap[pop_Index] = SLF_on_Each_Hap[female_new_pop_Index]
                    next_gen_ancesteral[pop_Index] = current_ancesteral[female_new_pop_Index]
                    hap_Replacement[pop_Index] = female_new_pop_Index
                    haplotype_Nat_temp[pop_Index] = haplotype_Nat[female_new_pop_Index]

                if is_female_pollinated == False:
                    hap_type_to_choose = np.random.choice([0,1,2], 1, p=choosing_female_hap_prob)

        # population loop is ending here --- Save popRNAse or tempPopRNAse both are same

        # mutation and selection loop ends here
        popRNAse, popFbox = tempPopRNAse, tempPopFbox
        SLF_on_Each_Hap, haplotype_Nat = tempSLF_on_Each_Hap, haplotype_Nat_temp
        current_ancesteral = next_gen_ancesteral
        #print (np.unique(next_gen_ancesteral), hap_nat_unique, hap_nat_unique_count, gen_Time)

        if np.any(haplotype_Nat == 0) == False and is_all_sc == False: # only once
            is_all_sc = True
            is_all_sc_gen_Time = gen_Time
            generations = gen_Time + 10000
            generations_window_change = gen_Time

        #if len(np.unique(next_gen_ancesteral)) == 1 and is_single_ancestor == False and is_all_sc == False:
        #    is_single_ancestor = True
        #    generations = gen_Time + 15000
        #    generations_window_change = gen_Time

        window_Size = window_Size if gen_Time <= generations_window_change else window_Size_Last

        #  --- for every t generation in multiple of some interger
        if gen_Time % window_Size == 0:
            #
            #
            # measure -------------- unique RNAse and SLF frequency --- for every generations
            #KEY, KEY_index, KEY_inverse, VALUES = np.unique(popRNAse, axis=0, return_index=True, return_counts=True, return_inverse=True)
            unique_RNAse_value, unique_RNAse_inverse, unique_RNAse_fre = np.unique(tempPopRNAse, axis=0, return_inverse=True, return_counts=True)
            #RNAse_dict_all_gen[gen_Time] = [unique_RNAse_value, unique_RNAse_inverse, unique_RNAse_fre]
            np.savetxt(pathToSave+'/unique_RNAse_gen_{}.dat'.format(gen_Time), np.concatenate((unique_RNAse_value, np.array([unique_RNAse_fre]).T), axis=1), fmt='%0.0f')
            np.savetxt(pathToSave+'/inverse_of_RNAse_{}.dat'.format(gen_Time), np.array(unique_RNAse_inverse).T, fmt='%0.0f')

            all_SLF = [each_Pollen[index*length_RF:length_RF*(index+1)] for each_Pollen in tempPopFbox for index in range(int(len(each_Pollen)/length_RF))]
            unique_SLF_value, unique_SLF_fre = np.unique(all_SLF, axis=0, return_counts=True)

            no_of_Unique_RNAse, no_of_Unique_SLF = len(unique_RNAse_value), len(unique_SLF_value)

            Functionality_SLF = [0 for i in range(no_of_Unique_SLF)]
            index_Interacting_RNAse_with_Unique_SLF = []
            for Each_unique_SLF_value, Each_unique_SLF_value_index in zip(unique_SLF_value, range(no_of_Unique_SLF)):
                SLF_degree = 0
                index_Interacting_RNAse_with_each_SLF = []
                for Each_unique_RNAse_value, Each_unique_RNAse_fre, Each_unique_RNAse_index in zip(unique_RNAse_value, unique_RNAse_fre, range(no_of_Unique_RNAse)):
                    if energyOfIntPair(Each_unique_RNAse_value, Each_unique_SLF_value) < energyThreshold:
                        SLF_degree += Each_unique_RNAse_fre
                        index_Interacting_RNAse_with_each_SLF.append(Each_unique_RNAse_index)
                    else:
                        index_Interacting_RNAse_with_each_SLF.append(-1)
                index_Interacting_RNAse_with_Unique_SLF.append(np.array(index_Interacting_RNAse_with_each_SLF))
                Functionality_SLF[Each_unique_SLF_value_index] = SLF_degree
            final_SLF_To_Save = np.concatenate((unique_SLF_value, np.array([unique_SLF_fre]).T, np.array([np.array(Functionality_SLF)]).T), axis=1)
            final_SLF_To_Save = np.concatenate((final_SLF_To_Save, np.array(index_Interacting_RNAse_with_Unique_SLF)), axis=1)
            np.savetxt(pathToSave+'/unique_SLF_gen_{}.dat'.format(gen_Time), final_SLF_To_Save, fmt='%0.0f')

            # measure ------
            popFbox_unique, popFbox_unique_inverse, popFbox_unique_count = np.unique(popFbox, axis=0, return_inverse=True, return_counts=True)
            SLF_nature_Each_Hap_to_Save = [0 for i in range(len(popFbox_unique_count))]
            for pollen_Here, pollen_Here_index in zip(popFbox_unique, range(len(popFbox_unique_count))):
                no_Of_SLFgenes = int(len(pollen_Here)/length_RF) #+ 1 # number of SLF genes on that haplotype
                SLF_at_pollen = [pollen_Here[ii*length_RF:(ii+1)*length_RF] for ii in range(no_Of_SLFgenes)]
                #SLF_value, SLF_inverse, SLF_fre = np.unique(SLF_at_pollen, axis=0, return_inverse=True, return_counts=True)
                SLF_nature, SLF_index = [], []
                #for SLF_value_current in SLF_value:
                for SLF_value_current in SLF_at_pollen:
                    for Each_unique_SLF_value, Each_unique_SLF_value_index in zip(unique_SLF_value, range(no_of_Unique_SLF)):
                        if np.array_equal(SLF_value_current, Each_unique_SLF_value) is True:
                            SLF_nature.append(Functionality_SLF[Each_unique_SLF_value_index]);
                            SLF_index.append(Each_unique_SLF_value_index);
                            break
                #SLF_nature_Each_Hap_to_Save[pollen_Here_index] = [SLF_index, SLF_inverse, SLF_fre, SLF_nature]
                SLF_nature_Each_Hap_to_Save[pollen_Here_index] = [SLF_index, SLF_nature]

            SLF_Save = open(pathToSave+'/SLF_nature_gen_{}.pkl'.format(gen_Time), "wb");   pickle.dump(SLF_nature_Each_Hap_to_Save, SLF_Save); SLF_Save.close()
            np.savetxt(pathToSave+'/SLF_nature_inv_gen_{}.dat'.format(gen_Time), np.array(popFbox_unique_inverse).T, fmt='%i')
            np.savetxt(pathToSave+'/SLF_nature_counts_gen_{}.dat'.format(gen_Time), np.array(popFbox_unique_count).T, fmt='%i')
            np.savetxt(pathToSave+'/attempts_{}.dat'.format(gen_Time), np.array(no_of_Attempts).T, fmt='%0.0f')
            np.savetxt(pathToSave+'/ancestral_Array_{}.dat'.format(gen_Time), np.array(next_gen_ancesteral).T, fmt='%0.0f')
            #np.savetxt(pathToSave+'/SC_Attempts_{}.dat'.format(gen_Time), np.array(SC_Attempts).T, fmt='%0.0f')
            np.savetxt(pathToSave+'/hap_Replacement_{}.dat'.format(gen_Time), np.array(hap_Replacement).T, fmt='%0.0f')
            np.savetxt(pathToSave+'/hap_Nature_{}.dat'.format(gen_Time), np.array(haplotype_Nat).T, fmt='%0.0f')

            print (gen_Time)

        gen_Time += 1

    print (('%s %0.0f %s') % ('completed', SampleNo, datetime.now() - start_time_01))



if __name__ == '__main__':

    print ('\n')
    print ('-----------------------------------------------------------------------')
    print ('\x1b[31m All the scripts of the current manuscript were run on the following version of Conda and Python \x1b[0m')
    print ('\x1b[31m                 conda version : 4.14.0 \x1b[0m')
    print ('\x1b[31m                 conda-build version : 3.21.5 \x1b[0m')
    print ('\x1b[31m                 python version : 3.9.19.final.0 \x1b[0m \n \n')
    print ('-----------------------------------------------------------------------')
    print ('\x1b[31m                 The current code generates the raw data \x1b[0m                \n')
    print ('\x1b[31m Here, in this sample code we have reduced the simulation time duration to finish it quickly \x1b[0m')
    print ('\x1b[31m To run it for sufficient time, please change the time duration as mentioned in the manuscript \x1b[0m \n')
    print ('\x1b[31m We are also running only for one independent run. To make the changes please change the value of "tot_independent_runs" to any desired value \x1b[0m')
    print ('-----------------------------------------------------------------------')
    print ('\n \n \n')

    tot_independent_runs = 1 # number of independent runs for each eth value, here max 25

    end_sim_T = {-10:150000, -9:150000, -8:150000, -7:150000, -6:200000, -5:275000, -4:450000, -3:450000, -2:450000} # original used in the manuscript
    end_sim_T = {-10:3000, -9:3000, -8:3000, -7:3000, -6:3000, -5:3000, -4:3000, -3:3000, -2:3000} # only in the sample code
    save_sim_T = {eth_temp:int(end_sim_T[eth_temp] - 25000) for eth_temp in [-10, -9, -8, -7, -6, -5, -4, -3, -2]} # will be saving the data for every 25 generations for the last 25000 generations
    seed_data = np.genfromtxt('./random_numpy_seed.dat').astype(int)

    N_Proces = 1 #int(tot_independent_runs*3) # number of CPUs
    Samples = [i for i in range(tot_independent_runs)];
    mutation, alpha, delta = 10**(-4), 0.90, 0.90
    population_main, no_attempts_main = 2000, 2000

    energy_variable = [-10, -9, -8, -7, -6, -5, -4, -3, -2][-1:]

    all_folder = ['./variable_ene_al_dl/E' + str(eth_temp) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100)) for eth_temp in energy_variable]
    #all_folder = ['./folder_temp']
    #folder = '' 'E'+str(threshold) + '_a'+str(int(alpha*100))+'_d'+str(int(delta*100))

    #Start_Time      = [215000 for i in Samples]
    Start_Time      = [0 for folder in all_folder for i in Samples] # if want to start from generation ZERO
    End_Time        = [end_sim_T[eth_temp] for eth_temp in energy_variable for i in Samples]  # stop time of simulation end_sim_T[eth_temp]
    Change_Window   = [save_sim_T[eth_temp] for eth_temp in energy_variable for i in Samples]  # time from saving data in the window size of 10, 20 or 25, check the main code
    Window_Size     = [25 for folder in all_folder for i in Samples] # to save the data size after change window time
    Sample_Index    = [i for folder in all_folder for i in Samples]  # index of sample number
    Folder          = [folder for folder in all_folder for i in Samples] # path to save the output

    Threshold       = [eth_temp for eth_temp in energy_variable for i in Samples] # threshold value
    Mutation        = [mutation for folder in all_folder for i in Samples] # mutation rate
    alpha_Input     = [alpha for folder in all_folder for i in Samples] # alpha
    delta_Input     = [delta for folder in all_folder for i in Samples] # delta
    Population_In   = [population_main for folder in all_folder for i in Samples]
    Attempts_In     = [no_attempts_main for folder in all_folder for i in Samples]

    Random_seed     = [int(i + seed_data[eth_temp+10][0]) for eth_temp in energy_variable for i in Samples]
    Numpy_seed      = [int(i + seed_data[eth_temp+10][1]) for eth_temp in energy_variable for i in Samples]

    #Pool(N_Proces).map(geneExchange, zip(Sample_Index, Threshold, Mutation, Folder))
    Pool(N_Proces).map(cnsr_evolution, \
                                        zip(Start_Time, End_Time, Change_Window, Window_Size, \
                                            Sample_Index, Threshold, Mutation, Folder, alpha_Input, \
                                            delta_Input, Random_seed, Numpy_seed, Population_In, Attempts_In))


#data = pickle.load(open(folder+'/data_0'+'/SLF_nature_gen_1.pkl', 'rb'))
#print (len(data))

#print (geneExchange([0, 0, '10_0', 10]))
print ('Ends----H:M:S--{}'.format(datetime.now() - start_time), '\n')
###############################################################################################################
###############################################################################################################
