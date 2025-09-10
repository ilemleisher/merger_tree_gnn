import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
import h5py

import dreams

auxtag  = 'MW_zooms' 
savetag = 'DREAMS_WDM_zoom'
basedir = '/standard/DREAMS/'

## Tree Walking Function
def walk_tree(box_to_process, counter, args):
    """
    This function walks the merger tree within a box. Starts at snapshot 90/redshift 0 with milky way subhalo and 
    works top to bottom following all branches.
    Takes in...
    box_to_process: box of interest within SB4 directory.
    Returns...
    Unique data containers for each data tag.
    """
    _s_ = f'Box {box_to_process}'
    print('#'*(len(_s_) + 6))
    print('## ' + _s_ + ' ##')
    print('#'*(len(_s_) + 6))
    
    # Continue established path to box of interest
    datadir  = basedir + 'Sims/WDM/'+auxtag+'/SB4/'    
    grpdir   = basedir + 'FOF_Subfind/WDM/'+auxtag+'/SB4/'
    datadir += 'box_%s/' %box_to_process
    grpdir  += 'box_%s/' %box_to_process
    
    # Constants used
    snapnr = 90
    h      = 0.6909
    
    # Establish MW index
    keysel  = ['SubhaloMass','GroupMassType','GroupFirstSub','GroupMass','GroupPos','SubhaloMassType']
    cat     = dreams.load_group_data(f'{grpdir}/fof_subhalo_tab_{snapnr:03}.hdf5', keysel)
    cat_idx = dreams.get_MW_idx(cat)
    
    first_sub    = cat['GroupFirstSub'][cat_idx]
    target_mass  = cat['GroupMass'][cat_idx] * 1.00E+10/h
    subfind_mass = cat['SubhaloMassType'][first_sub,4] * 1.00E+10/h
    print('Subfind Mass:',np.log10(target_mass))
    
    keys  = ['FirstSubhaloInFOFGroupID','NextSubhaloInFOFGroupID','SubfindID',
             'DescendantID','MainLeafProgenitorID','SnapNum','SubhaloMassType',
             'SubhaloPos','SubhaloGrNr','TreeID','NextProgenitorID','Group_R_Crit200',
             'SubhaloID','GroupMassType','GroupPos', 'FirstProgenitorID', 'GroupFirstSub',
             'GroupMass']
    
    sublink = dreams.load_slink(grpdir + 'tree_extended.hdf5', keys)
    
    try:
        mw_idx = dreams.get_MW_idx(sublink, tree=True)
    except ValueError as e:
        print(e)
        print('❌❌❌❌❌❌❌ THIS BOX FAILED ❌❌❌❌❌❌❌')
        return counter
        
    sublink_mass = np.log10(sublink['GroupMass'][mw_idx]*1.00E+10/h)
    print('SubLink Mass:',sublink_mass)
    
    sf_pos = cat['GroupPos'][cat_idx,:]/h
    sl_pos = sublink['GroupPos'][mw_idx,:]/h
    
    distance = np.sqrt(
        (sf_pos[0] - sl_pos[0])**2 + 
        (sf_pos[1] - sl_pos[1])**2 + 
        (sf_pos[2] - sl_pos[2])**2 
    )
    
    print('Subfind Position:',cat['GroupPos'][cat_idx,:]/h)
    print('SubLink Position:',sublink['GroupPos'][mw_idx,:]/h)
    print('✅ Subfind and SubLink Positions Agree' if distance < 1 else '❌ Subfind and SubLink Positions Disagree','\n')
    target = sublink['FirstSubhaloInFOFGroupID'][mw_idx]
    print('Target Index',target)
    
    # Load merger tree
    with h5py.File( grpdir + 'tree_extended.hdf5', 'r' ) as tree_file:
        # Locate MW index within box
        all_snaps = np.array(tree_file["SnapNum"])
        
        subhaloid = np.array(tree_file["FirstSubhaloInFOFGroupID"])[all_snaps == snapnr]        
        matching_index = np.argmin( np.abs(subhaloid - target) )
        
        file_index = np.arange(len(all_snaps))[all_snaps==snapnr][matching_index]

        print('Location in tree file:',file_index)
        
        check_mass = np.log10(tree_file['GroupMass'][file_index]*1.00E+10/h)
        print(f'✅ Target Found Successfully ({check_mass:0.3f})' \
              if np.isclose(check_mass, sublink_mass) else '❌ Target Incorrect','\n')
        
        subfind_stellar_mass = np.log10(subfind_mass)
        sublink_stellar_mass = np.log10(tree_file['SubhaloMassType'][file_index,4]*1.00E+10/h+1)
        
        print(f'✅ Subfind and SubLink Agree on Stellar Mass ({sublink_stellar_mass:0.3f})' \
              if np.isclose(subfind_stellar_mass, sublink_stellar_mass) else\
              f'❌ Subfind and SubLink Agree on Stellar Mass (SF: {subfind_stellar_mass:0.3f}, SL: {sublink_stellar_mass:0.3f})','\n')
        
        if np.isclose(subfind_stellar_mass, sublink_stellar_mass):
            counter += 1
        
        return counter
        
        rootID       = tree_file["SubhaloID"][file_index] # ID in the tree
        fpID         = tree_file["FirstProgenitorID"][file_index] # ID of the first progenitor (next most massive subhalo)
        current_npID = tree_file["NextProgenitorID"][file_index] # ID of the next progenitor (most massive subhalo after FP)
     
        # Create containers for data
        containers = [[] for i in args]
        for i in range(len(args)):
            containers[i].append(args[i])
            containers[i].append([tree_file[args[i]][file_index]])

        # Create a container with initial subhalo indices to walk from.
        # This currently includes MW index and will include as all 
        # MW next progenitors as well.
        initial_subhalo_indices = [file_index]

        # While loop over all next progenitors of MW
        # Not necessary to do the same for FP since get_progenitors loop begins with the initial subhalo's
        # first progenitor, and there is a max of one FP.
        while current_npID != -1:
    
            current_npIndex = file_index + (current_npID - rootID)

            for i in range(len(args)):
                containers[i][1].append(tree_file[args[i]][current_npIndex])
            
            initial_subhalo_indices.append(current_npIndex)

            
            current_npID = tree_file['NextProgenitorID'][current_npIndex]

        # Defining a recursive function call within tree walking function
        # This will allow us to follow each first and next progenitor from each
        # previous first and next progenitor, walking the tree top down
        # To do this we go to the first progenitor, then all of the FPs next progenitors,
        # then call the function again on each of those NPs. 
        def get_progenitors(index,file_index,containers):
            """
            Takes in...
            Index: the subhalo index to begin walking the tree with. Function will save all data from all subhalos following this one.
            File_index: MW index, used to find progenitor indices.
            Lists: data containers to be defined in walking function.
            Returns...
            Lists: same data containers.
            """
        
            fpID = tree_file["FirstProgenitorID"][index] # fpID of the first subhalo.
        
            # While loop over fpID
            # Documentation says if fpID = -1, there is no more FP.
            # Each subhalo should have either 1 or 0 FPs.
            while fpID != -1:
        
                fpIndex = file_index + (fpID - rootID) # "Calculating" the index of the FP given its subhalo ID
                fpDesIndex = file_index+(tree_file["DescendantID"][index]-rootID)
                
                # Get data for each argument based on the FP index and append it to the specific containers

                for i in range(len(args)):
                    containers[i][1].append(tree_file[args[i]][fpIndex])
                            
                # Get NP ID based on the FP to set up next while loop.
                npID = tree_file['NextProgenitorID'][fpIndex]
                
                # While loop over npID. Each subhalo can any number of NPs.
                # The recursive function will continue walking down this branch as far as
                # it goes, and then will come back to the next NP based on the same FP. 
                # Then it will walk down that branch and so on.
                while (npID != -1):
        
                    npIndex = file_index + (npID - rootID) # "Calculating" the index of the NP given its subhalo ID.
                    npDesIndex = file_index + (tree_file["DescendantID"][fpIndex] - rootID)
                                       
                    # Get data for each argument based on the NP index and append it to the specific containers.
                    for i in range(len(args)):
                        containers[i][1].append(tree_file[args[i]][npIndex])
        
                    # Calling the function to make it recursive.
                    containers = get_progenitors(npIndex,file_index,containers)
                    
                    # Updating the npID to continue while loop.
                    npID = tree_file['NextProgenitorID'][npIndex]
        
                # Updating the fpID to continue while loop.
                fpID = tree_file['FirstProgenitorID'][fpIndex]
                
            return containers

        # Calling recursive function for each index in the initial indices container
        # Important to do it in this way to keep the structure of the
        # data the same. This way the data is always organized by first progenitor to next progenitor.
        for index in initial_subhalo_indices:
            
            get_progenitors(index,file_index,containers)

    data = dict(containers)
    
    return data



#Always need DescendantID, SubhaloID, SnapNum
boxes = range(1024)
datalist = []
indices_to_remove = [1, 2, 4, 5, 6, 9, 10, 11, 14, 17, 18, 20]
to_run = [item for i, item in enumerate(boxes) if i not in indices_to_remove]
counter = 0
for box in to_run:
    counter = walk_tree(box,counter,["SubhaloSFR","SubhaloMass","SnapNum","SubhaloGrNr", "DescendantID", 
                         "SubhaloID", "MainLeafProgenitorID", "LastProgenitorID","RootDescendantID",
                         "TreeID","FirstProgenitorID","NextProgenitorID","FirstSubhaloInFOFGroupID",
                         "NextSubhaloInFOFGroupID","NumParticles","MassHistory","GroupNsubs","SubhaloMassType",
                         "SubhaloBHMass","SubhaloBHMdot","SubhaloBfldDisk","SubhaloBfldHalo","SubhaloCM","SubhaloGasMetalFractions",
                         "SubhaloGasMetallicity","SubhaloHalfmassRad","SubhaloLen","SubhaloParent",
                         "SubhaloPos","SubhaloSpin","SubhaloStarMetalFractions","SubhaloStarMetallicity",
                         "SubhaloStellarPhotometrics","SubhaloVel","SubhaloVelDisp","SubhaloVmax","SubhaloWindMass"])
    datalist.append(-1)
    print('')
_s_ = f"Total Success Rate: {counter}/{len(to_run)}"
print('!'*(len(_s_) + 10))
print('!!!  '+_s_+'  !!!')
print('!'*(len(_s_) + 10))

with open('31data','wb') as f:
    pickle.dump(datalist, f)
