# gbcalculation.py
""" Module to create files for calculations of grain boundary properties and aggregate the results
Author:  Chris Race
Date:    3rd January 2017
Contact: christopher.race@manchester.ac.uk
"""
import numpy as np
from . import gbsupercell
from . import csl
from . import crystaltools as ct
import os
from shutil import copyfile

GBCALCULATION_TOL = 1e-3

def gamma_surface_build(supercell, shiftrange, shiftres, expansionrange, expansionres, path, filestocopy, tol=GBCALCULATION_TOL, fileformat='lammps'):
    """Generate files and folder structure for a gamma surface calculation and write a file list"""
    
    if np.shape(shiftrange) != (2,2):
        raise RuntimeError("Shift range must have 4 components, shape (2,2)")
    if len(shiftres) != 2:
        raise RuntimeError("Shift resolution must have 2 components")
    if len(expansionrange) != 2:
        raise RuntimeError("Expansion range must have 2 components")
    if len(expansionres) != 1:
        raise RuntimeError("Expansion resolution must have 1 component")

    fl = open(path + '/joblist.txt', 'w')
    numjobs = (shiftres[0] + 1)*(shiftres[1] + 1)*(expansionres[0] + 1)
    fl.write(str(numjobs) + '\n')
    
    # Save supercell repeats in order to reset supercell size for each iteration
    supercell_repeats = supercell.repeats_black.tolist()
    supercell_repeats.append(supercell.repeats_white[2])
    
    shift = np.zeros(2)
    for i in range(shiftres[0] + 1):
        if shiftres[0] == 0:
            shift[0] = shiftrange[0,0]
        else:
            shift[0] = shiftrange[0,0] + (shiftrange[0,1] - shiftrange[0,0])/shiftres[0]*i
        for j in range(shiftres[1] + 1):
            if shiftres[1] == 0:
                shift[1] = shiftrange[1,0]
            else:
                shift[1] = shiftrange[1,0] + (shiftrange[1,1] - shiftrange[1,0])/shiftres[1]*j
            supercell.set_bicrystal_shift(inplane_shift=shift[:])
            for k in range(expansionres[0] + 1):
                supercell.set_supercell_size(repeats=supercell_repeats)
                if expansionres[0] == 0:
                    expansion = expansionrange[0]
                else:
                    expansion = expansionrange[0] + (expansionrange[1] - expansionrange[0])/expansionres[0]*k
                supercell.set_expansion(expansion_discrete=expansion)
                
                folder = path + '/Sh1_' + str(np.around(shift[0],5)) + '/Sh2_' + str(np.around(shift[1],5)) + '/Exp_' + str(np.around(expansion,5)) + '/'
                fl.write(folder + ' ' + str(np.around(shift[0],5)) + ' ' + str(np.around(shift[1],5)) + ' ' + str(np.around(expansion,5)) + '\n')
                d = os.path.dirname(folder)
                if not os.path.exists(d):
                    os.makedirs(d)
                supercell.calculate_atom_arrays(gamma_surf=True)
                if fileformat == 'lammps':
                    atomfile = folder + 'lammps.txt'
                    supercell.write_lammps(filename=atomfile)
                elif fileformat == 'vasp':
                    atomfile = folder + 'POSCAR'
                    if supercell.blocks_fixed:
                        fix_xy = True
                    else:
                        fix_xy = False
                    supercell.write_vasp(filename=atomfile, fix_xy=fix_xy)
                else:
                    raise RuntimeError("Unrecognised output file format")
                #copyfile(atomfile, folder+atomfile)
                for filename in filestocopy:
                    copyfile(filename, folder+filename)
    fl.close()
    return
    
def gamma_surface_analyse(shiftrange, shiftres, expansionrange, expansionres, path, energycol=4, final=True, tol=GBCALCULATION_TOL):
    """  """
    if np.shape(shiftrange) != (2,2):
        raise RuntimeError("Shift range must have 4 components, shape (2,2)")
    if len(shiftres) != 2:
        raise RuntimeError("Shift resolution must have 2 components")
    if len(expansionrange) != 2:
        raise RuntimeError("Expansion range must have 2 components")
    if len(expansionres) != 1:
        raise RuntimeError("Expansion resolution must have 1 component")
    
    currdir = os.getcwd() # Save the current directory
    d = os.path.dirname(path)
    
    # First get details of supercell for one reference simulation
    folder = path + '/Sh1_' + str(np.around(shiftrange[0,0],5)) + '/Sh2_' + str(np.around(shiftrange[1,0],5)) + '/Exp_' + str(np.around(expansionrange[0],5)) + '/'
    os.chdir(folder)
    # First get number of atoms and supercell shape
    f = open("dump.0")
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>3 and words[3] == 'ATOMS' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open("dump.0")
    for s in range(lineIndex+1):
        f.readline()
    num_atoms = f.readline().split()[0]
    f.close()
    f = open("dump.0")
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>1 and words[1] == 'BOX' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    supercell_shape = np.zeros((3,3), dtype=float)
    f = open("dump.0")
    for s in range(lineIndex+1):
        f.readline()
    for s in range(3):
        words = f.readline().split()
        supercell_shape[s,:] = [words[0], words[1], words[2]]
    f.close()
    os.chdir(currdir)
    # Convert dump file format for shape to something sensible
    supercell_shape = ct.lammps_dump_to_ppd(supercell_shape)
    
    # Now iterate over all files in gamma surface calculation
    numshifts = np.array([shiftres[0] + 1, shiftres[1] + 1])
    numexpansions = expansionres[0] + 1
    numjobs = (shiftres[0] + 1)*(shiftres[1] + 1)*(expansionres[0] + 1)
    shift = np.zeros(2)
    results = np.zeros((numshifts[0], numshifts[1], numexpansions , 6), dtype=float)
    shifts1 = np.zeros(numshifts[0], dtype=float)
    shifts2 = np.zeros(numshifts[1], dtype=float)
    expansions = np.zeros(numexpansions, dtype=float)
    for i in range(numshifts[0]):
        if shiftres[0] == 0:
            shift[0] = shiftrange[0,0]
        else:
            shift[0] = shiftrange[0,0] + (shiftrange[0,1] - shiftrange[0,0])/shiftres[0]*i
        shifts1[i] = shift[0]
        for j in range(numshifts[1]):
            if shiftres[1] == 0:
                shift[1] = shiftrange[1,0]
            else:
                shift[1] = shiftrange[1,0] + (shiftrange[1,1] - shiftrange[1,0])/shiftres[1]*j
            shifts2[j] = shift[1]
            for k in range(numexpansions):
                if expansionres[0] == 0:
                    expansion = expansionrange[0]
                else:
                    expansion = expansionrange[0] + (expansionrange[1] - expansionrange[0])/expansionres[0]*k
                expansions[k] = expansion
                folder = path + '/Sh1_' + str(np.around(shift[0],5)) + '/Sh2_' + str(np.around(shift[1],5)) + '/Exp_' + str(np.around(expansion,5)) + '/'
                os.chdir(folder)
                # Now obtain results from lammps files
                e0 = None
                e1 = None
                
                found0 = False
                found1 = False
                # Get energy of initial configuration
                f = open("log.lammps")
                for l, line in enumerate(f):
                    words = line.split()
                    if (len(words)>0 and words[0] == 'Step' and found0 == False):
                        lineIndex = l
                        found0 = True   # This flag initially set here to allow detection of genuinely first line of output in log
                f.close()
                f = open("log.lammps")
                for l, line in enumerate(f):
                    words = line.split()
                    if l == lineIndex+1:
                        e0 = float(words[energycol])
                f.close()
                # Get energy of relaxed configuration
                f = open("log.lammps")
                exitloop = False
                for l, line in enumerate(f):
                    words = line.split()
                    if (len(words)>0 and words[0] == 'Loop' and not exitloop):
                        if not final:
                            found1 = True
                            exitloop = True
                            lineIndex = l
                        else:
                            found1 = True
                            lineIndex = l
                f.close()
                f = open("log.lammps")
                for l, line in enumerate(f):
                    words = line.split()
                    if l == lineIndex-1:
                        e1 = float(words[energycol])
                if found0:
                    results[i,j,k,0] = e0
                if found1:
                    results[i,j,k,1] = e1
                    
                os.chdir(currdir)
                
    os.chdir(path)
    # Write out details of gamma surface specification:
    fo = open('gamma_spec.txt', 'w')
    fo.write('# Number of shifts and shift range in direction 1:\n')
    fo.write(str(shiftres[0]).rjust(8) + str(shiftrange[0,0]).rjust(16) + str(shiftrange[0,1]).rjust(16) + '\n')
    fo.write('# Number of shifts and shift range in direction 2:\n')
    fo.write(str(shiftres[1]).rjust(8) + str(shiftrange[1,0]).rjust(16) + str(shiftrange[1,1]).rjust(16) + '\n')
    fo.write('# Number of expansions and expansion range:\n')
    fo.write(str(expansionres[0]).rjust(8) + str(expansionrange[0]).rjust(16) + str(expansionrange[1]).rjust(16) + '\n')
    fo.write('# GB Supercell vectors:\n')
    for s in range(3):
        for t in range(3):
            fo.write(str(np.around(supercell_shape[s,t],5)).rjust(16))
        fo.write('\n')
    fo.write('# Number of atoms:\n')
    fo.write(str(num_atoms) + '\n')
    fo.close()
    
    # Write out a master results file
    fo = open('gamma_results.txt', 'w')
    fo.write('#' + 'x_shift'.rjust(11) + 'y_shift'.rjust(12) + 'expansion'.rjust(12) + 'E_unrelaxed'.rjust(20) + 'E_relaxed'.rjust(20) + '\n')
    for i in range(numshifts[0]):
        for j in range(numshifts[1]):
            for k in range(numexpansions):
                fo.write(str(np.around(shifts1[i],5)).rjust(12) + str(np.around(shifts2[j],5)).rjust(12) + str(np.around(expansions[k],5)).rjust(12) + str(np.around(results[i,j,k,0],8)).rjust(20) + str(np.around(results[i,j,k,1],8)).rjust(20) + '\n')  
    fo.close()
    
    # Write out unrelaxed gamma surface (i.e. as constructed, with no grain boundary expansion)  
    fo = open('gamma_unrelaxed.txt', 'w')
    fo.write('#' + 'x_shift'.rjust(11) + 'y_shift'.rjust(12) + 'expansion'.rjust(12) + 'E_unrelaxed'.rjust(20) + 'E_relaxed (ignore)'.rjust(20) + '\n')
    for i in range(numshifts[0]):
        for j in range(numshifts[1]):
            fo.write(str(np.around(shifts1[i],5)).rjust(12) + str(np.around(shifts2[j],5)).rjust(12) + str(0.0).rjust(12) + str(np.around(results[i,j,0,0],8)).rjust(20) + str(np.around(results[i,j,0,1],8)).rjust(20) + '\n')  
        fo.write('\n')    
    fo.close()
    
    # Now find optimal expansions and minimum energies
    outcome = np.zeros((numshifts[0], numshifts[1]))            # Is fitting successful?
    datapoints = np.zeros((2, 3))   # Data for fitting parabola
    optima = np.zeros((numshifts[0], numshifts[1], 2))          # Optimal energy and expansion
    best = np.zeros((numshifts[0], numshifts[1], 2))          # Best (i.e. non-ineterpolated) energy and expansion
    for i in range(numshifts[0]):
        for j in range(numshifts[1]):
            emin = 99999.0
            minindex = -1
            # Find minimum
            for k in range(numexpansions):
                if results[i,j,k,1] < emin:
                    emin = results[i,j,k,1]
                    minindex = k
            best[i,j,0] = expansions[minindex]
            best[i,j,1] = results[i,j,minindex,1]
            if minindex == 0:
                outcome[i,j] = -1
                optima[i,j,0] = expansions[minindex]
                optima[i,j,1] = results[i,j,minindex,1]
            elif minindex == numexpansions-1:
                outcome[i,j] = 1
                optima[i,j,0] = expansions[minindex]
                optima[i,j,1] = results[i,j,minindex,1]
            else:
                outcome[i,j] = 0
                for l in range(3):
                    datapoints[0,l] = expansions[minindex-1+l]
                    datapoints[1,l] = results[i,j,minindex-1+l,1]
                # Now fit quadratic and find optimum values
                x = np.array([[datapoints[0,0]*datapoints[0,0], datapoints[0,0], 1.0],[datapoints[0,1]*datapoints[0,1], datapoints[0,1], 1.0],[datapoints[0,2]*datapoints[0,2], datapoints[0,2], 1.0]])
                y = np.array([datapoints[1,0],datapoints[1,1],datapoints[1,2]])
                a = np.dot(np.linalg.inv(x),y)
                x0 = -1.0 * a[1] / 2.0 / a[0]
                y0 = a[0]*x0*x0 + a[1]*x0 + a[2]
                optima[i,j,0] = x0
                optima[i,j,1] = y0
    # Get config and results for best case (no interpolation)
    best_config = np.zeros(2) # Values of best shifts 
    best_vals = np.zeros(2)   # Values of energy and expansion at best shift
    best_indices = np.unravel_index(np.argmin(best[:,:,1], axis=None), best[:,:,1].shape)
    best_config[0] = shifts1[best_indices[0]]
    best_config[1] = shifts2[best_indices[1]]
    best_vals[0] = best[best_indices[0],best_indices[1],0]
    best_vals[1] = best[best_indices[0],best_indices[1],1]
    # Calculate global optimal configuration 
    minconfig = np.zeros(2) # Values of shifts at overall minimum
    minvals = np.zeros((2,2))   # Values of energy and expansion at overall minimum (axis, exp or energy)
    mindatapoints = np.zeros((2, 3, 2))   # (axis, series, exp or energy)
    emin = 99999.0
    minindices = np.array([-1,-1], dtype=int)    
    for i in range(numshifts[0]):
        for j in range(numshifts[1]):
                if optima[i,j,1] < emin:
                    emin = optima[i,j,1]
                    minindices[0] = i
                    minindices[1] = j
    # Now build fitting data
    # Direction 1
    mindatapoints[0,1,:] = optima[minindices[0],minindices[1],:]
    if minindices[0] == 0:
        mindatapoints[0,0,:] = optima[numshifts[0]-1,minindices[1],:]
    else:
        mindatapoints[0,0,:] = optima[minindices[0]-1,minindices[1],:]
    if minindices[0] == numshifts[0]-1:
        mindatapoints[0,2,:] = optima[0,minindices[1],:]
    else:
        mindatapoints[0,2,:] = optima[minindices[0]+1,minindices[1],:]
    # Direction 2
    mindatapoints[1,1,:] = optima[minindices[0],minindices[1],:]
    if minindices[1] == 0:
        mindatapoints[1,0,:] = optima[minindices[0],numshifts[1]-1,:]
    else:
        mindatapoints[1,0,:] = optima[minindices[0],minindices[1]-1,:]
    if minindices[1] == numshifts[0]-1:
        mindatapoints[1,2,:] = optima[minindices[0],0,:]
    else:
        mindatapoints[1,2,:] = optima[minindices[0],minindices[1]+1,:]
    
    # Now fit quadratics and find optimum shifts to minimise energy along each axis
    for s in range(2): # Loop over axes
        dshift = (shiftrange[s,1] - shiftrange[s,0])/shiftres[s]  # Spacing of shift values
        t = 1 # Select energy
        x = np.array([[dshift*dshift, -1.0*dshift, 1.0],[0.0, 0.0, 1.0],[dshift*dshift, dshift, 1.0]])
        y = np.array([mindatapoints[s,0,t], mindatapoints[s,1,t], mindatapoints[s,2,t]])
        #print(   'axis:', s)
        #print('dshift:',dshift)
        #print('datapoints:',mindatapoints[s,:,:])
        #print('fitting data x:', x)
        #print('fitting data y:', y)
        a = np.dot(np.linalg.inv(x),y)
        x0 = -1.0 * a[1] / 2.0 / a[0]
        y0 = a[0]*x0*x0 + a[1]*x0 + a[2]
        minconfig[s] = x0
        minvals[s,1] = y0
        #print('opt shift:', minconfig[s])
       
        # Next calculate interpolated expansion at this point (again using a quadratic)
        t = 0 # Select expansion
        y = np.array([mindatapoints[s,0,t],mindatapoints[s,1,t],mindatapoints[s,2,t]])
        a = np.dot(np.linalg.inv(x),y)
        y0 = a[0]*x0*x0 + a[1]*x0 + a[2] # Note that this is using the value of x0 optimised for energy above
        minvals[s,0] = y0
        #print('opt values:', minvals[s,:])
        #print(shifts1[minindices[s]])
        #print(shifts1[minindices[s]]+minconfig[s])

    # Write out optimised gamma surface    
    fo = open('gamma_optimised.txt', 'w')
    fo.write('#' + 'Optimal shifts are: ' + str(np.around(shifts1[minindices[0]]+minconfig[0],5)) + ' ' +  str(np.around(shifts2[minindices[1]]+minconfig[1],5)) + '\n')
    fo.write('#' + 'Optimal energy: ' + str(np.around(0.5*(minvals[0,1]+minvals[1,1]),5)) + '\n')
    fo.write('#' + 'Optimal expansion: ' + str(np.around(0.5*(minvals[0,0]+minvals[1,0]),5)) + '\n')
    fo.write('#' + 'x_shift'.rjust(11) + 'y_shift'.rjust(12) + 'completion'.rjust(16) + 'expansion'.rjust(12)+ 'E_optimised'.rjust(20) + 'equivalence'.rjust(16) + '\n')
    for i in range(numshifts[0]):
        for j in range(numshifts[1]):
            fo.write(str(np.around(shifts1[i],5)).rjust(12) + str(np.around(shifts2[j],5)).rjust(12))
            if outcome[i,j] == -1:
                fo.write('Exp_too_large'.rjust(16))
            elif outcome[i,j] == 1:
                fo.write('Exp_too_small'.rjust(16))
            else:
                fo.write('Success'.rjust(16))
                
            fo.write(str(np.around(optima[i,j,0],5)).rjust(12) + str(np.around(optima[i,j,1],8)).rjust(20))
            fo.write('Not checked'.rjust(16))
            fo.write('\n')    
        fo.write('\n')    
    fo.close() 
    
    # Write out best gamma surface (ie. optimised without any interpolation)
    fo = open('gamma_best.txt', 'w')
    fo.write('#' + 'Best shifts are: ' + str(np.around(best_config[0],5)) + ' ' +  str(np.around(best_config[1],5)) + '\n')
    fo.write('#' + 'Best energy: ' + str(np.around(best_vals[1],5)) + '\n')
    fo.write('#' + 'Best expansion: ' + str(np.around(best_vals[0],5)) + '\n')
    fo.write('#' + 'x_shift'.rjust(11) + 'y_shift'.rjust(12) + 'completion'.rjust(16) + 'expansion'.rjust(12)+ 'E_best'.rjust(20) + 'equivalence'.rjust(16) + '\n')
    for i in range(numshifts[0]):
        for j in range(numshifts[1]):
            fo.write(str(np.around(shifts1[i],5)).rjust(12) + str(np.around(shifts2[j],5)).rjust(12))
            if outcome[i,j] == -1:
                fo.write('Exp_too_large'.rjust(16))
            elif outcome[i,j] == 1:
                fo.write('Exp_too_small'.rjust(16))
            else:
                fo.write('Success'.rjust(16))
                
            fo.write(str(np.around(best[i,j,0],5)).rjust(12) + str(np.around(best[i,j,1],8)).rjust(20))
            fo.write('Not checked'.rjust(16))
            fo.write('\n')    
        fo.write('\n')    
    fo.close()
    
    os.chdir(currdir)
    
    return

# def gamma_surface_analyse_old(supercell, shiftrange, shiftres, expansionrange, expansionres, path, tol=GBCALCULATION_TOL):
#     """  """
    
#     if np.shape(shiftrange) != (2,2):
#         raise RuntimeError("Shift range must have 4 components, shape (2,2)")
#     if len(shiftres) != 2:
#         raise RuntimeError("Shift resolution must have 2 components")
#     if len(expansionrange) != 2:
#         raise RuntimeError("Expansion range must have 2 components")
#     if len(expansionres) != 1:
#         raise RuntimeError("Expansion resolution must have 1 component")
    
#     currdir = os.getcwd() # Save the current directory
#     d = os.path.dirname(path)
#     numshifts = np.array([shiftres[0] + 1, shiftres[1] + 1])
#     numexpansions = expansionres[0] + 1
#     numjobs = (shiftres[0] + 1)*(shiftres[1] + 1)*(expansionres[0] + 1)
#     shift = np.zeros(2)
#     results = np.zeros((numshifts[0], numshifts[1], numexpansions , 6), dtype=float)
#     shifts1 = np.zeros(numshifts[0], dtype=float)
#     shifts2 = np.zeros(numshifts[1], dtype=float)
#     expansions = np.zeros(numexpansions, dtype=float)
#     for i in range(numshifts[0]):
#         shift[0] = shiftrange[0,0] + (shiftrange[0,1] - shiftrange[0,0])/shiftres[0]*i
#         shifts1[i] = shift[0]
#         for j in range(numshifts[1]):
#             shift[1] = shiftrange[1,0] + (shiftrange[1,1] - shiftrange[1,0])/shiftres[1]*j
#             shifts2[j] = shift[1]
#             for k in range(numexpansions):
#                 expansion = expansionrange[0] + (expansionrange[1] - expansionrange[0])/expansionres[0]*k
#                 expansions[k] = expansion
#                 folder = path + '/Sh1_' + str(np.around(shift[0],5)) + '/Sh2_' + str(np.around(shift[1],5)) + '/Exp_' + str(np.around(expansion,5)) + '/'
#                 print(folder)
#                 os.chdir(folder)
#                 # Now obtain results from lammps log files
#                 e0 = None
#                 e1 = None
#                 found0 = False
#                 found1 = False
#                 f = open("log.lammps")
#                 for l, line in enumerate(f):
#                     words = line.split()
#                     if (len(words)>0 and words[0] == 'Step' and found0 == False):
#                         lineIndex = l
#                         found0 = True   # This flag initially set here to allow detection of genuinely first line of output in log
#                 f.close()
#                 found0 = False
#                 f = open("log.lammps")
#                 for l, line in enumerate(f):
#                     words = line.split()
#                     if l == lineIndex+1:
#                         found0 = True
#                         e0 = float(words[4])
#                         #egb0a = float(words[6])
#                         #egb0b = float(words[7])
#                 f.close()
#                 f = open("log.lammps")
#                 for l, line in enumerate(f):
#                     words = line.split()
#                     if (len(words)>0 and words[0] == 'Loop'):
#                         lineIndex = l
#                 f.close()
#                 f = open("log.lammps")
#                 for l, line in enumerate(f):
#                     words = line.split()
#                     if l == lineIndex-1:
#                         found1 = True
#                         e1 = float(words[4])
#                         #egb1a = float(words[6])
#                         #egb1b = float(words[7])
#                 if found0:
#                     results[i,j,k,0] = e0
#                     #results[i,j,k,2] = egb0a
#                     #results[i,j,k,3] = egb0b
#                     #results[i,j,k,4] = egb1a
#                     #results[i,j,k,5] = egb1b
#                 if found1:
#                     results[i,j,k,1] = e1
                    
#                 os.chdir(currdir)
                
#     # Write out details of gamma surface specification:
#     fo = open('gamma_spec.txt', 'w')
#     fo.write('# Number of shifts and shift range in direction 1:\n')
#     fo.write(str(shiftres[0]).rjust(8) + str(shiftrange[0,0]).rjust(16) + str(shiftrange[0,1]).rjust(16) + '\n')
#     fo.write('# Number of shifts and shift range in direction 2:\n')
#     fo.write(str(shiftres[1]).rjust(8) + str(shiftrange[1,0]).rjust(16) + str(shiftrange[1,1]).rjust(16) + '\n')
#     fo.write('# Number of expansions and expansion range:\n')
#     fo.write(str(expansionres[0]).rjust(8) + str(expansionrange[0]).rjust(16) + str(expansionrange[1]).rjust(16) + '\n')
#     fo.write('# GB Supercell vectors:\n')
#     for s in range(3):
#         for t in range(3):
#             fo.write(str(supercell.supercell[s,t]).rjust(16))
#         fo.write('\n')
#     fo.write('# Number of atoms:\n')
#     fo.write(str(supercell.num_atoms_black + supercell.num_atoms_white) + '\n')
#     fo.close()
    
#     # Write out a master results file
#     fo = open('gamma_results.txt', 'w')
#     fo.write('#' + 'x_shift'.rjust(11) + 'y_shift'.rjust(12) + 'expansion'.rjust(12) + 'E_unrelaxed'.rjust(20) + 'E_relaxed'.rjust(20) + '\n')
#     for i in range(numshifts[0]):
#         for j in range(numshifts[1]):
#             for k in range(numexpansions):
#                 fo.write(str(np.around(shifts1[i],5)).rjust(12) + str(np.around(shifts2[j],5)).rjust(12) + str(np.around(expansions[k],5)).rjust(12) + str(np.around(results[i,j,k,0],8)).rjust(20) + str(np.around(results[i,j,k,1],8)).rjust(20) + '\n')  
#     fo.close()
    
#     # Write out unrelaxed gamma surface (i.e. as constructed, with no grain boundary expansion)  
#     fo = open('gamma_unrelaxed.txt', 'w')
#     fo.write('#' + 'x_shift'.rjust(11) + 'y_shift'.rjust(12) + 'expansion'.rjust(12) + 'E_unrelaxed'.rjust(20) + 'E_relaxed (ignore)'.rjust(20) + '\n')

#     for i in range(numshifts[0]):
#         for j in range(numshifts[1]):
#             fo.write(str(np.around(shifts1[i],5)).rjust(12) + str(np.around(shifts2[j],5)).rjust(12) + str(0.0).rjust(12) + str(np.around(results[i,j,0,0],8)).rjust(20) + str(np.around(results[i,j,0,1],8)).rjust(20) + '\n')  
#         fo.write('\n')    
#     fo.close()
    
#     # Now find optimal expansions and minimum energies
#     outcome = np.zeros((numshifts[0], numshifts[1]))            # Is fitting successful?
#     #symcheck = np.zeros((numshifts[0], numshifts[1]), dtype=int)            # Are boundaries equivalent?
#     equivtol = 1e-3
#     datapoints = np.zeros((2, 3))   # Data for fitting parabola
#     optima = np.zeros((numshifts[0], numshifts[1], 2))          # Optimal energy and expansion
#     for i in range(numshifts[0]):
#         for j in range(numshifts[1]):
#             emin = 99999.0
#             minindex = -1
#             # Find minimum
#             for k in range(numexpansions):
#                 if results[i,j,k,1] < emin:
#                     emin = results[i,j,k,1]
#                     minindex = k
#             # Check for boundary equivalence
#             # if abs(2.0*(results[i,j,minindex,2] - results[i,j,minindex,3])/(results[i,j,minindex,2] + results[i,j,minindex,3])) < equivtol:
#             #     symcheck[i,j] = 0
#             # else:
#             #     symcheck[i,j] = 1
#             # if abs(2.0*(results[i,j,minindex,4] - results[i,j,minindex,5])/(results[i,j,minindex,4] + results[i,j,minindex,5])) < equivtol:
#             #     symcheck[i,j] = symcheck[i,j]
#             # else:
#             #     symcheck[i,j] = symcheck[i,j] + 2
#             # Now check for values either side
#             if minindex == 0:
#                 outcome[i,j] = -1
#                 optima[i,j,0] = expansions[minindex]
#                 optima[i,j,1] = results[i,j,minindex,1]
#             elif minindex == numexpansions-1:
#                 outcome[i,j] = 1
#                 optima[i,j,0] = expansions[minindex]
#                 optima[i,j,1] = results[i,j,minindex,1]
#             else:
#                 outcome[i,j] = 0
#                 for l in range(3):
#                     datapoints[0,l] = expansions[minindex-1+l]
#                     datapoints[1,l] = results[i,j,minindex-1+l,1]
#                 # Now fit quadratic and find optimum values
#                 x = np.array([[datapoints[0,0]*datapoints[0,0], datapoints[0,0], 1.0],[datapoints[0,1]*datapoints[0,1], datapoints[0,1], 1.0],[datapoints[0,2]*datapoints[0,2], datapoints[0,2], 1.0]])
#                 y = np.array([datapoints[1,0],datapoints[1,1],datapoints[1,2]])
#                 a = np.dot(np.linalg.inv(x),y)
#                 x0 = -1.0 * a[1] / 2.0 / a[0]
#                 y0 = a[0]*x0*x0 + a[1]*x0 + a[2]
#                 optima[i,j,0] = x0
#                 optima[i,j,1] = y0
                
#     # Calculate global optimal configuration 
#     minconfig = np.zeros(2) # Values of shifts at overall minimum
#     minvals = np.zeros((2,2))   # Values of energy and expansion at overall minimum (axis, exp or energy)
#     mindatapoints = np.zeros((2, 3, 2))   # (axis, series, exp or energy)
#     emin = 99999.0
#     minindices = np.array([-1,-1], dtype=int)    
#     for i in range(numshifts[0]):
#         for j in range(numshifts[1]):
#                 if optima[i,j,1] < emin:
#                     emin = optima[i,j,1]
#                     minindices[0] = i
#                     minindices[1] = j
#     # Now build fitting data
#     # Direction 1
#     mindatapoints[0,1,:] = optima[minindices[0],minindices[1],:]
#     if minindices[0] == 0:
#         mindatapoints[0,0,:] = optima[numshifts[0]-1,minindices[1],:]
#     else:
#         mindatapoints[0,0,:] = optima[minindices[0]-1,minindices[1],:]
#     if minindices[0] == numshifts[0]-1:
#         mindatapoints[0,2,:] = optima[0,minindices[1],:]
#     else:
#         mindatapoints[0,2,:] = optima[minindices[0]+1,minindices[1],:]
#     # Direction 2
#     mindatapoints[1,1,:] = optima[minindices[0],minindices[1],:]
#     if minindices[1] == 0:
#         mindatapoints[1,0,:] = optima[minindices[0],numshifts[1]-1,:]
#     else:
#         mindatapoints[1,0,:] = optima[minindices[0],minindices[1]-1,:]
#     if minindices[1] == numshifts[0]-1:
#         mindatapoints[1,2,:] = optima[minindices[0],0,:]
#     else:
#         mindatapoints[1,2,:] = optima[minindices[0],minindices[1]+1,:]
    
#     # Now fit quadratics and find optimum shifts to minimise energy along each axis
#     for s in range(2): # Loop over axes
#         dshift = (shiftrange[s,1] - shiftrange[s,0])/shiftres[s]  # Spacing of shift values
#         t = 1 # Select energy
#         x = np.array([[dshift*dshift, -1.0*dshift, 1.0],[0.0, 0.0, 1.0],[dshift*dshift, dshift, 1.0]])
#         y = np.array([mindatapoints[s,0,t], mindatapoints[s,1,t], mindatapoints[s,2,t]])
#         #print(   'axis:', s)
#         #print('dshift:',dshift)
#         #print('datapoints:',mindatapoints[s,:,:])
#         #print('fitting data x:', x)
#         #print('fitting data y:', y)
#         a = np.dot(np.linalg.inv(x),y)
#         x0 = -1.0 * a[1] / 2.0 / a[0]
#         y0 = a[0]*x0*x0 + a[1]*x0 + a[2]
#         minconfig[s] = x0
#         minvals[s,1] = y0
#         #print('opt shift:', minconfig[s])
       
#         # Next calculate interpolated expansion at this point (again using a quadratic)
#         t = 0 # Select expansion
#         y = np.array([mindatapoints[s,0,t],mindatapoints[s,1,t],mindatapoints[s,2,t]])
#         a = np.dot(np.linalg.inv(x),y)
#         y0 = a[0]*x0*x0 + a[1]*x0 + a[2] # Note that this is using the value of x0 optimised for energy above
#         minvals[s,0] = y0
#         #print('opt values:', minvals[s,:])
#         #print(shifts1[minindices[s]])
#         #print(shifts1[minindices[s]]+minconfig[s])

#     # Write out optimised gamma surface    
#     fo = open('gamma_optimised.txt', 'w')
#     fo.write('#' + 'Optimal shifts are: ' + str(np.around(shifts1[minindices[0]]+minconfig[0],5)) + ' ' +  str(np.around(shifts2[minindices[1]]+minconfig[1],5)) + '\n')
#     fo.write('#' + 'Optimal energy: ' + str(np.around(0.5*(minvals[0,1]+minvals[1,1]),5)) + '\n')
#     fo.write('#' + 'Optimal expansion: ' + str(np.around(0.5*(minvals[0,0]+minvals[1,0]),5)) + '\n')
#     fo.write('#' + 'x_shift'.rjust(11) + 'y_shift'.rjust(12) + 'completion'.rjust(16) + 'expansion'.rjust(12)+ 'E_optimised'.rjust(20) + 'equivalence'.rjust(16) + '\n')
#     for i in range(numshifts[0]):
#         for j in range(numshifts[1]):
#             fo.write(str(np.around(shifts1[i],5)).rjust(12) + str(np.around(shifts2[j],5)).rjust(12))
#             if outcome[i,j] == -1:
#                 fo.write('Exp_too_large'.rjust(16))
#             elif outcome[i,j] == 1:
#                 fo.write('Exp_too_small'.rjust(16))
#             else:
#                 fo.write('Success'.rjust(16))
                
#             fo.write(str(np.around(optima[i,j,0],5)).rjust(12) + str(np.around(optima[i,j,1],8)).rjust(20))
            
#             fo.write('Not checked'.rjust(16))  # replaces lines below
#             # if symcheck[i,j] == 0:
#             #     fo.write('Equivalence_YY'.rjust(16))
#             # elif symcheck[i,j] == 1:
#             #     fo.write('Equivalence_NY'.rjust(16))
#             # elif symcheck[i,j] == 2:
#             #     fo.write('Equivalence_YN'.rjust(16))
#             # elif symcheck[i,j] == 3:
#             #     fo.write('Equivalence_YY'.rjust(16))

#             fo.write('\n')    
#         fo.write('\n')    
#     fo.close() 

#     return
        
if __name__ == "__main__": 
    # Test code
    
    #-------------------------- test some fcc boundaries
    
    testcsl = csl.Csl('fcc')
    #testcsl.set_debug()
    
    # Will give a symmetric tilt for orthogonal or non-orthogonal csl basis
    h = 1; k = 1; l = 1
    theta = 38.213211
    H = 0; K = 0; L = 1
    theta = theta*np.pi/180.0
    testcsl.set_axis([h,k,l])
    testcsl.set_angle(theta)
    testcsl.find_axis_rotation_matrix()
    testcsl.find_angle_rotation_matrix()
    testcsl.enable_search(10)
    testcsl.find_csl_basis()
    testcsl.set_boundary_plane([H,K,L])
    testcsl.find_gb_basis()
    testcsl.classify_gb()
    testcsl.calculate_gb_cell()

    testsc = gbsupercell.Supercell(testcsl)
    testsc.set_lattice_parameter(3.0)
    testsc.set_repeats([1,1,3])
    testsc.set_fixblock(1.5)
    
    shiftrange = np.array([[0.0,1.0],[0.0,1.0]])
    shiftres = np.array([2,2], dtype=int)
    expansionrange = np.array([0.0,2.0])
    expansionres = np.array([2], dtype=int)
    path = './GammaTest/'
    filestocopy = ['relax.in']
    gamma_surface_build(testsc, shiftrange, shiftres, expansionrange, expansionres, path, filestocopy)
    
    
    #testsc.set_shift([0.2,0.2])
    #testsc.set_expansion(0.5)
    #testsc.set_vacuum(3.0)
    #testsc.set_fixblock(1.0)
    #testsc.write_lammps()