## Standard Imports

import matplotlib
matplotlib.use('agg') # Used on HPC machine (suppresses plt.show)
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
import h5py

# Read in hdf5 files
import h5py

sys.path.insert(1,'/home/aku7cf/torreylabtoolsPy3')

# Torreylab Tools
import simread.readsubfHDF5 as rsubf

## Establish path - assumes user is running this script through Rivanna

auxtag  = 'MW_zooms'
savetag = 'DREAMS_WDM_zoom'
basedir = '/standard/DREAMS/'

## Tree Walking Function

def walk_tree(box_to_process, *args):
    """
    This function walks the merger tree within a box. Starts at snapshot 90/redshift 0 with milky way subhalo and
    works top to bottom following all branches.
    Takes in...
    box_to_process: box of interest within SB4 directory.
    Returns...
    Unique data containers for each data tag.
    """
    # Continue established path to box of interest
    datadir = basedir + 'Sims/WDM/'+auxtag+'/SB4/'
    grpdir  = basedir + 'FOF_Subfind/WDM/'+auxtag+'/SB4/'
    datadir += 'box_%s/' %box_to_process
    grpdir  += 'box_%s/' %box_to_process

    # Constants used
    snapnrs = 90
    little_h = 0.6774

    # Establish MW index
    keysel = ['SubhaloMass','GroupMassType','GroupFirstSub']
    cat = rsubf.subfind_catalog( grpdir, snapnrs, keysel=keysel )
    central_mass = cat.SubhaloMass[0] * 1.00E+10/little_h
    masses = cat.GroupMassType * 1.00E+10/little_h
    tot_masses = np.sum(masses,axis=1)
    mcut = (tot_masses > 7e11) & (tot_masses < 2.5e12)
    contamination = masses[:,2] / tot_masses
    idx = np.argmin(contamination[mcut])
    mw_idx = np.arange(len(masses))[mcut][idx]
    first_sub = cat.GroupFirstSub[mw_idx]

    # Load merger tree
    with h5py.File( grpdir + 'tree_extended.hdf5', 'r' ) as tree_file:

        # Locate MW index within box
        all_snaps = np.array(tree_file["SnapNum"])
        all_mass  = np.array(tree_file["SubhaloMass"])
        at_z0 = all_snaps == 90
        mass_at_z0 = np.where(at_z0, all_mass, -1) * 1.00E+10/little_h
        subfind_id = np.array(tree_file["SubfindID"])
        file_index = np.arange(0,len(subfind_id))[subfind_id==first_sub][0]

        # Load data for first subhalo in tree (MW)
        # Documentation can be found on https://www.tng-project.org/data/docs/specifications/#sec4a under SubLink
        # To change the data that's loaded, follow the documentation and change the tags. Do not change these first two tags.
        rootID  = tree_file["SubhaloID"][file_index] # ID in the tree
        fpID  = tree_file["FirstProgenitorID"][file_index] # ID of the first progenitor (next most massive subhalo)
        current_npID = tree_file["NextProgenitorID"][file_index] # ID of the next progenitor (most massive subhalo after FP)

        containers = [[] for i in args]
        for i in range(len(args)):
            containers[i].append(tree_file[args[i]][file_index])

        # Okay to change.
        current_snap = tree_file["SnapNum"][file_index] # Snapshot number of this subhalo
        current_SFR = tree_file["SubhaloSFR"][file_index] # SFR of this subhalo
        current_mass = tree_file["SubhaloMass"][file_index] # mass of this subhalo

        # Create containers for data. They begin with the first subhalo's data, in this case the MW, within them. This is because the
        # get progenitors function only gets progenitor data, and not current data. To have all the data we need to start with the current subhalo.
        snapList = [current_snap]
        sfrList = [current_SFR]
        massList = [current_mass]

        # Create a container with initial subhalo indices to walk from. This currently includes MW index and will include as all
        # MW next progenitors as well.
        initial_subhalo_indices = [file_index]

        # While loop over all next progenitors of MW. Not necessary to do the same for FP since get_progenitors loop begins with the initial subhalo's
        # first progenitor, and there is a max of one FP.
        while current_npID != -1:

            current_npIndex = file_index + (current_npID - rootID)

            for i in range(len(args)):
                containers[i].append(tree_file[args[i]][current_npIndex])

            initial_npSnap = tree_file['SnapNum'][current_npIndex]
            initial_npSFR = tree_file['SubhaloSFR'][current_npIndex]
            initial_npMass = tree_file["SubhaloMass"][current_npIndex]

            snapList.append( initial_npSnap )
            sfrList.append( initial_npSFR )
            massList.append(initial_npMass)

            initial_subhalo_indices.append(current_npIndex)


            current_npID = tree_file['NextProgenitorID'][current_npIndex]

        # Defining a recursive function call within tree walking function. This will allow us to follow each first and next progenitor from each
        # previous first and next progenitor, walking the tree top down. To do this we go to the first progenitor, then all of the FPs next progenitors,
        # then call the function again on each of those NPs.
        def get_progenitors(index,file_index,containers,snapList,sfrList,massList):
            """
            Takes in...
            Index: the subhalo index to begin walking the tree with. Function will save all data from all subhalos following this one.
            File_index: MW index, used to find progenitor indices.
            Lists: data containers to be defined in walking function.
            Returns...
            Lists: same data containers.
            """

            fpID = tree_file["FirstProgenitorID"][index] # fpID of the first subhalo.

            # While loop over fpID. Documentation says if fpID = -1, there is no more FP. Each subhalo should have either 1 or 0 FPs.
            while fpID != -1:

                fpIndex = file_index + (fpID - rootID) # "Calculating" the index of the FP given its subhalo ID. This works for any subhalo index.

                for i in range(len(args)):
                    containers[i].append(tree_file[args[i]][fpIndex])

                fpSnap = tree_file['SnapNum'][fpIndex]
                fpSFR = tree_file['SubhaloSFR'][fpIndex]
                fpMass = tree_file["SubhaloMass"][fpIndex]

                snapList.append( fpSnap )
                sfrList.append( fpSFR )
                massList.append( fpMass )

                npID = tree_file['NextProgenitorID'][fpIndex]

                while npID != -1:

                    npIndex = file_index + (npID - rootID)

                    for i in range(len(args)):
                        containers[i].append(tree_file[args[i]][npIndex])

                    npSnap = tree_file['SnapNum'][npIndex]
                    npSFR = tree_file['SubhaloSFR'][npIndex]
                    npMass = tree_file["SubhaloMass"][npIndex]

                    snapList.append( npSnap )
                    sfrList.append( npSFR )
                    massList.append( npMass )

                    containers,snapList,sfrList,massList = get_progenitors(npIndex,file_index,containers,snapList,sfrList,massList)

                    npID = tree_file['NextProgenitorID'][npIndex]


                fpID = tree_file['FirstProgenitorID'][fpIndex]

            return containers,snapList,sfrList,massList

        # Calling recursive function for each index in the initial indices container. Important to do it in this way to keep the structure of the
        # data the same. This way the data is always organized by first progenitor to next progenitor.
        for index in initial_subhalo_indices:

            get_progenitors(index,file_index,containers,snapList,sfrList,massList)

    return containers,snapList,sfrList,massList

print(walk_tree(1,"SubhaloSFR")[0])
print(walk_tree(1,"SubhaloSFR")[2])
