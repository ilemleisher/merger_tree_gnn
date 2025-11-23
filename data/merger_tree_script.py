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
def walk_tree(box_to_process, args):
    """
    This function walks the merger tree within a box. Starts at snapshot 90/redshift 0 with milky way subhalo and 
    works top to bottom following all branches.
    Takes in...
    box_to_process: box of interest within SB4 directory.
    args: the features you want to save from each subhalo. Follow docmentation on: https://www.tng-project.org/data/docs/specifications/#sec4a
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
    
    keys  = ['FirstSubhaloInFOFGroupID','NextSubhaloInFOFGroupID','SubfindID',
             'DescendantID','MainLeafProgenitorID','SnapNum','SubhaloMassType',
             'SubhaloPos','SubhaloGrNr','TreeID','NextProgenitorID','Group_R_Crit200',
             'SubhaloID','GroupMassType','GroupPos', 'FirstProgenitorID', 'GroupFirstSub',
             'GroupMass']
    
    sublink = dreams.load_slink(grpdir + 'tree_extended.hdf5', keys)
    mw_idx = dreams.get_MW_idx(sublink, tree=True)
    target = sublink['FirstSubhaloInFOFGroupID'][mw_idx]
    
    # Load merger tree
    with h5py.File( grpdir + 'tree_extended.hdf5', 'r' ) as tree_file:
        # Locate MW index within box
        all_snaps = np.array(tree_file["SnapNum"])
        
        subhaloid = np.array(tree_file["FirstSubhaloInFOFGroupID"])[all_snaps == snapnr]        
        matching_index = np.argmin( np.abs(subhaloid - target) )
        
        file_index = np.arange(len(all_snaps))[all_snaps==snapnr][matching_index]

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

if __name__ == "__main__":
  
    #These boxes should always be removed, they are corrupted (as of 11/25). The corresponding box parameters have been removed from the parameters file.
    boxes_to_remove = [1, 2, 4, 5, 6, 9, 10, 11, 14, 17, 18, 19, 20, 21, 30, 42, 88] 
    
    datalist = []
  
    first_box = sys.argv[1]
    last_box = sys.argv[2]
    boxes = range(int(first_box),int(last_box))
    to_run = [item for i, item in boxes if i not in boxes_to_remove]

    for box in to_run:
        print(box)
        try:
            data = walk_tree(box, sys.argv[3:])
            datalist.append(data)
        except Exception as e:
            print(box)
            print(e)
    
    with open('raw_merger_tree_data.pkl','wb') as f:
        pickle.dump(datalist, f)
