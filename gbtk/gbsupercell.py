# gbsupercell.py
""" Module to handle definitions of supercells including
generating atom coordinates and supercell specifications and writing out files
in various formats

| Author:  Chris Race
| Date:    11th January 2017
| Contact: christopher.race@manchester.ac.uk
"""
import numpy as np
import pandas as pd
from . import crystaltools as ct
from . import grainboundary as gb

# For visualisation only
try:
    import plotly
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)
except ModuleNotFoundError:
    pass

GBSUPERCELL_TOL = 1e-3

class Supercell(object):
    """A Supercell holds details of the simulation cell. Details will depend on the type of supercell to be built.
    
    :ivar object grainboundary: Grain boundary object
    :ivar ndarray((3,3),dtype=float) lattice_basis_black: Array holding lattice basis vectors for black cell in supercell orientation
    :ivar ndarray((3,3),dtype=float) lattice_basis_white: Array holding lattice basis vectors for white cell in supercell orientation
    :ivar ndarray((3,3),dtype=float) supercell_rotation: Rotation matrix into supercell orientation such that grain boiundary normal lies along z in cartesian space 
    :ivar ndarray((3,3),dtype=float) supercell_unit_cell: Supercell unit cell vectors (cartesian basis)
    :ivar ndarray((3,3),dtype=float) lattice_basis_black: Lattice basis vectors in supercell orientation (cartesian basis)
    :ivar ndarray((3,3),dtype=float) lattice_basis_white: Lattice basis vectors in supercell orientation (cartesian basis)
    :ivar ndarray(3,dtype=float) shift: Relative shift of cells at boundary
    :ivar float boundary_plane_shift: Out of plane shift of each boundary
    :ivar float expansion: Amount to expand cell by at each boundary (free volume per unit area)
    :ivar float vacuum: Amount of vacuum to add at ends of cell - setting a non-zero value results in a supercell containing a single boundary
    :ivar float fix_blocks: Length of regions in which to fix positions of atoms (region will be chosen most distant from the boundary or boundaries)
    :ivar float a: Lattice parameter, defualts to 1.0
    :ivar bool bicrystal_shifted: Flag indicating whether two halves of bicrystal are shifted relative to one antother at the boundary
    :ivar bool boundary_plane_shifted: Flag indicating whether the nominal boundary plane between the two crystals has been shifter perpendicular to the boudnary plane
    :ivar bool vacuum_set: Flag indicate whether vacuum has been added at th ends of the supercell (resulting in a single boundary and two free surfaces)
    :ivar bool expansion_set: Flag indicates whether a volume expansion has been added at the boundary
    :ivar int num_atoms: Number of atoms in the supercell
    :ivar bool supercell_size_set: Flag indicates whether the size of the supercell has been set
    :ivar bool normal_constraint: Flag indicates whether atoms are constrined to move only in direction normal to boundary plane (via format of certain output files)
    :ivar bool blocks_fixed: Flag indicates whether blocks of atoms have been fixed (via format of certain output files)
    :ivar bool debug: Flag to indicate whether debug information is requested
    :ivar bool atom_arrays_calculated: Flag to indicate whether the atomic coordinates in the supercell have been calculated
    :ivar ndarray(2,dtype=float) boundary_plane_z: Positions of the two grain boundaries (z-coordinates of boundary planes)
    :ivar ndarray((N,3),dtype=float) r_black: Coordinates of atoms in black half of bicrystal
    :ivar ndarray((N,3),dtype=float) r_white: Coordinates of atoms in white half of bicrystal
    :ivar ndarray(N,dtype=float) atom_types_black: Types of atoms in black half of bicrystal
    :ivar ndarray(N,,dtype=float) atom_types_white: Types of atoms in white half of bicrystal
    :ivar int num_atoms_black: Number of atoms in black half of bicrystal
    :ivar int num_atoms_white: Number of atoms in white half of bicrystal
    :ivar ndarray((N,3),dtype=float) r_black_orig: Copy of above variable used to make gamma surface calculations more efficient
    :ivar ndarray((N,3),dtype=float) r_white_orig: Copy of above variable used to make gamma surface calculations more efficient
    :ivar ndarray(N,dtype=float) atom_types_black_orig: Copy of above variable used to make gamma surface calculations more efficient
    :ivar ndarray(N,,dtype=float) atom_types_white_orig: Copy of above variable used to make gamma surface calculations more efficient
    :ivar int num_atoms_black_orig: Copy of above variable used to make gamma surface calculations more efficient
    :ivar int num_atoms_white_orig: Copy of above variable used to make gamma surface calculations more efficient
    """    

    def __init__(self,grainboundary):
        """Initialise an empty supercell

        :param grainboundary: Grain boundary object for which a supercell is to be created
        :type grainboundary: object
        """        
    
        
        #@classmethod
        #def from_csl_grainboundary(self,grainboundary):
        self.grainboundary = grainboundary
        self.a = 1.0
        self.bicrystal_shifted = False
        self.boundary_plane_shifted = False
        self.vacuum_set = False
        self.expansion_set = False
        self.num_atoms = 0
        self.supercell_size_set = False
        self.normal_constraint = False
        self.blocks_fixed = False
        self.debug = False
        self.atom_arrays_calculated = False
        
    def calculate_rotation(self):
        """Calculate the rotation requiree to bring the grain boundary normal parallel withe the cartesian z axis.
        Also poultate details of the supercell cell vectors and the rotated form of the lattice basis vectors in each half of the bicrystal
        Reorders the description of the supercell in a format suitable for use in lammps input

        :raises RuntimeError: "Error generating unit cell for supercell" if reordering of the basis vectors fails
        """        
        
        if self.debug: 
            print("----------------------------------------------------")
            print("gbsupercell.calculate_rotation() debug:")
            print("***************************************")
        
        # Calculate rotation into supercell frame
        gbcell_vectors_rotated_black = np.dot(np.linalg.inv(self.grainboundary.csl.misorientation_rotation),self.grainboundary.boundary_cell_cartesian_black.T).T
        # if self.debug:
        #     print("TEMP First rotation (cartesian basis)")
        #     for s in range(3):
        #         print('[ %12.6f %12.6f %12.6f ]' % (gbcell_vectors_rotated_black[s,0],gbcell_vectors_rotated_black[s,1],gbcell_vectors_rotated_black[s,2]))
        #     print("")
        # First rotate first gb cell vector into x axis
        dirn_1 = gbcell_vectors_rotated_black[0,:]
        target_1 = np.array([1.0,0.0,0.0])
        rotation_1 = ct.rotation_matrix_into_direction(dirn_1,target_1)
        # Now rotate normal to plane defined by first and second GB cell vectors into z axis
        # These two rotations should bring GB normal parallel to z-axis
        gbcell_vectors_rotated_black = np.dot(rotation_1,gbcell_vectors_rotated_black.T).T
        dirn_2 = np.cross(gbcell_vectors_rotated_black[0,:],gbcell_vectors_rotated_black[1,:])
        target_2 = np.array([0.0,0.0,-1.0])
        rotation_2 = ct.rotation_matrix_into_direction(dirn_2,target_2)
        self.supercell_rotation = np.dot(rotation_2,rotation_1)
        
        # Now calculate cell vectors for a unit of supercell
        # We want these in the positive octant with a conformation suitable for Lammps
        supercell_unit_cell = np.dot(self.supercell_rotation,np.dot(np.linalg.inv(self.grainboundary.csl.misorientation_rotation),self.grainboundary.boundary_cell_cartesian_black.T)).T
        # if self.debug:
        #     print("TEMP Supercell unit cell vectors, before reordering (cartesian basis)")
        #     for s in range(3):
        #         print('[ %12.6f %12.6f %12.6f ]' % (supercell_unit_cell[s,0],supercell_unit_cell[s,1],supercell_unit_cell[s,2]))
        #     print("")
        reorder_indices = np.zeros(3, dtype=int)
        for s in range(3):
            if abs(supercell_unit_cell[s,2]) < GBSUPERCELL_TOL:
                if abs(supercell_unit_cell[s,1]) < GBSUPERCELL_TOL:
                    reorder_indices[s] = 0
                else:
                    reorder_indices[s] = 1
            else:
                reorder_indices[s] = 2
        if np.sum(reorder_indices) != 3:
            print(reorder_indices)
            print(supercell_unit_cell)
            raise RuntimeError("Error generating unit cell for supercell")
        self.supercell_unit_cell = np.zeros((3,3), dtype=float)
        for s in range(3):
            self.supercell_unit_cell[reorder_indices[s]] = supercell_unit_cell[s]
            
        for s in range(3):
            if self.supercell_unit_cell[s,s] < 0.0:
                self.supercell_unit_cell[s,:] = -self.supercell_unit_cell[s,:]

        if self.debug:
            print("Supercell unit cell vectors (cartesian basis)")
            for s in range(3):
                print('[ %12.6f %12.6f %12.6f ]' % (self.supercell_unit_cell[s,0],self.supercell_unit_cell[s,1],self.supercell_unit_cell[s,2]))
            print("")
        
        # Calculate lattice basis vectors in supercell orientation
        self.lattice_basis_black = np.dot(self.supercell_rotation,np.dot(np.linalg.inv(self.grainboundary.csl.misorientation_rotation),self.grainboundary.csl.lattice.cell_vectors.T)).T
        self.lattice_basis_white = np.dot(self.supercell_rotation,np.dot(self.grainboundary.csl.misorientation_rotation,self.grainboundary.csl.lattice.cell_vectors.T)).T
        if self.debug:
            print("Basis vectors (cartesian basis)")
            print('Black                                       White')
            for s in range(3):
                print('[ %12.6f %12.6f %12.6f ]  [ %12.6f %12.6f %12.6f ]' % (
                    self.lattice_basis_black[s,0],self.lattice_basis_black[s,1],self.lattice_basis_black[s,2],
                    self.lattice_basis_white[s,0],self.lattice_basis_white[s,1],self.lattice_basis_white[s,2]
                    ))
            print("")
    
    def set_lattice_parameter(self,a):
        """Set the value of the primary lattice parameter

        :param a: Primary lattice parameter
        :type a: float
        """        
        self.a = a
            
    def set_supercell_size(self,repeats=None, radius=None):
        """Set the size of the simulation supercell as either a number of repeats of the minimal grain boundary cell 
        or as a sphere radius (sphere method nod uccrently implemented)

        :param repeats: vector or list with four components, two specifiying number of repeats in GB plane and two more specifying repeats normal to plane in black and white halves, defaults to None
        :type repeats: ndarray(4,dtype=int) or [int,int,int,int], optional
        :param radius: Radius of spherical cluster, defaults to None
        :type radius: float, optional
        :raises RuntimeError: "Either number of repeats in crystal blocks or sphere radius must be specified", if no parameter passed
        :raises RuntimeError: "Vector of repeats must have 4 components", if specification of size has wrong number of components
        :raises RuntimeError: "Total supercell length must be an even number of repeats"
        :raises RuntimeError: "Spherical cluster case not currently supported", if a radius for a spherical cluster has been specified
        """        
    
        if self.debug: 
            print("----------------------------------------------------")
            print("gbsupercell.set_supercell_size() debug:")
            print("***************************************")
            
        if repeats is None and radius is None:
            raise RuntimeError("Either number of repeats in crystal blocks or sphere radius must be specified")
            # Handle case of blocks of bicrystal
        if repeats is not None:
            if len(repeats) != 4:
                raise RuntimeError("Vector of repeats must have 4 components") 
            if (repeats[2]+repeats[3])%2 != 0 and repeats[2] != 0 and repeats[3] !=0 :
                raise RuntimeError("Total supercell length must be an even number of repeats") 
            self.repeats_black = np.array(repeats[0:3])
            self.repeats_white = np.array(repeats[0:3])
            self.repeats_white[2] = repeats[3]
            self.supercell = np.zeros((3,3), dtype=float)
            for s in range(2):
                self.supercell[s,:] = self.a*(self.repeats_black[s])*self.supercell_unit_cell[s,:]
            self.supercell[2,:] = self.a*(self.repeats_black[2]+self.repeats_white[2])*self.supercell_unit_cell[2,:]
            
            self.boundary_plane_z = np.array([
                self.a*0.5*self.repeats_white[2]*self.supercell_unit_cell[2,2],
                self.a*(0.5*self.repeats_white[2] + self.repeats_black[2])*self.supercell_unit_cell[2,2]
            ])
            
            self.supercell_centre = self.a*0.5*(self.repeats_white[2] + self.repeats_black[2])*self.supercell_unit_cell[2,2]
            
            self.original_supercell = np.copy(self.supercell)
            self.original_boundary_plane_z = np.copy(self.boundary_plane_z)
            
            if self.debug:
                print("Supercell vectors (cartesian basis)")
                for s in range(3):
                    print('[ %12.6f %12.6f %12.6f ]' % (self.supercell[s,0],self.supercell[s,1],self.supercell[s,2]))
                print("Boundary planes at (z-coordinate)")
                print('%12.6f and %12.6f' % (self.boundary_plane_z[0],self.boundary_plane_z[1]))
                print("Supercell centre at (z-coordinate)")
                print('%12.6f' % (self.supercell_centre))
                print("")
                
        if radius is not None:
            # Handle case of spherical bicrystal
                raise RuntimeError("Spherical cluster case not currently supported")
        
        self.supercell_size_set = True
    
    def set_bicrystal_shift(self,dsc_shift=None, inplane_shift=None):
        """Set the relative inter-granular translation at the boundary

        :param dsc_shift: Relative shift of grains at boundary as multiples of DSC lattice vectors, defaults to None
        :type dsc_shift: ndarray(3,dtype=float) or [float,float,float], optional
        :param inplane_shift: Relative shift of grains at boundary as multiples of the two supercell cell vectors in the GB plane, defaults to None
        :type inplane_shift: ndarray(2,dtype=float) or [float,float], optional
        :raises RuntimeError: "Relative shift must be specified either as a multiple of dsc vectors or of in-plane csl vectors", if neither paramter set
        :raises RuntimeError: "Relative crystal shift must have 2 components"
        :raises RuntimeError: "Shifts must be less than 1.0"
        :raises RuntimeError: "Relative shift vector should have zero z-component, something has gone wrong!". This error should never be invoked if the code is working as intended
        :raises RuntimeError: "Relative crystal shift must have 3 components"
        :raises RuntimeError: "DSC basis must be calculated before setting shift in this way"
        :raises RuntimeError: "Shifts must be less than 1.0"
        """        
        if dsc_shift is None and inplane_shift is None:
            raise RuntimeError("Relative shift must be specified either as a multiple of dsc vectors or of in-plane csl vectors")
        
        if inplane_shift is not None:
            if len(inplane_shift) != 2:
                raise RuntimeError("Relative crystal shift must have 2 components")
            if np.max(inplane_shift) > 1.0:
                raise RuntimeError("Shifts must be less than 1.0")
            self.shift = np.zeros(3, dtype=float)
            self.shift = self.a * (inplane_shift[0]*self.supercell_unit_cell[0,:] + inplane_shift[1]*self.supercell_unit_cell[1,:])
            if abs(self.shift[2]) > GBSUPERCELL_TOL:
                raise RuntimeError("Relative shift vector should have zero z-component, something has gone wrong!") 
            self.bicrystal_shifted = True
            
        elif dsc_shift is not None:
            if len(dsc_shift) != 3:
                raise RuntimeError("Relative crystal shift must have 3 components")
            if not self.grainboundary.csl.dsc_basis_set:
                raise RuntimeError("DSC basis must be calculated before setting shift in this way")
            if np.max(dsc_shift) >= 1.0:
                raise RuntimeError("Shifts must be less than 1.0")
            csl_vectors_rotated_black = np.dot(np.linalg.inv(self.grainboundary.csl.misorientation_rotation),self.grainboundary.csl.csl_vectors_cartesian_black.T).T
            self.shift = np.zeros(3, dtype=float)
            for s in range(3):
                self.shift = self.shift + self.a * dsc_shift[s] * self.grainboundary.csl.dsc_vector_fractions[s] * csl_vectors_rotated_black[s,:]
            self.bicrystal_shifted = True
    
    def set_boundary_plane_shift(self,plane_shift):
        """Set the relative shift of the boundary plane in direction normal to plane
        Use with caution - there has been only limited testing of this functionality

        :param plane_shift: Shift the boundary plane perpendicular to plane, expressed as a fraction of repeat period based on DSC lattice.
        :type plane_shift: float
        """        
        csl_vectors_rotated_black = np.dot(np.linalg.inv(self.grainboundary.csl.misorientation_rotation),self.grainboundary.csl.csl_vectors_cartesian_black.T).T
        self.boundary_plane_shift = 0.0
        for s in range(3):
            self.boundary_plane_shift = self.boundary_plane_shift + self.a * plane_shift * self.grainboundary.csl.dsc_vector_fractions[s] * csl_vectors_rotated_black[s,2]
        self.boundary_plane_shifted = True
    
    def set_expansion(self,expansion_discrete=None, expansion_sigmoid=None):
        """Set the expansion (excess volume per unit area) at each boundary in distance units

        :param expansion_discrete: Expansion to apply at each boundary in distance units, defaults to None
        :type expansion_discrete: float, optional
        :param expansion_sigmoid: Parmeters of sigmoid function to determine a smooth distribution of excess volume at boundary, defaults to None
        :type expansion_sigmoid: [float,float], optional
        :raises RuntimeError: "Grain boundary expansion must be specified either as a single value for a discrete block expansion or as a pair of values for a sigmoid expansion"
        :raises RuntimeError: "Sigmoidal expansion method not currently implemented"
        :raises RuntimeError: "Specification of expansion sigmoid requires two numbers, the second larger than the first", this error is not currently reachable
        """        
        if expansion_discrete is None and expansion_sigmoid is None:
            raise RuntimeError("Grain boundary expansion must be specified either as a single value for a discrete block expansion or as a pair of values for a sigmoid expansion")
        if expansion_discrete is not None:
            self.expansion = expansion_discrete
            self.expansion_set = True
        if expansion_sigmoid is not None:
            raise RuntimeError("Sigmoidal expansion method not currently implemented")
            if len(expansion_sigmoid) != 2 or expansion_sigmoid[1]<expansion_sigmoid[0]:
                raise RuntimeError("Specification of expansion sigmoid requires two numbers, the second larger than the first")
            self.expansion = np.array(expansion_sigmoid)
            self.expansion_set = True
            
    def set_vacuum(self,vacuum):
        """Set the vacuum width at the end of the cell in distance units

        :param vacuum: Amount of vacuum to add at ends of supercell
        :type vacuum: float
        """        
        self.vacuum = vacuum
        self.vacuum_set = True
    
    def set_debug(self):
        """Turn on debug info"""
        self.debug = True
    
    def unset_debug(self):
        """Turn off debug info"""
        self.debug = False
        
    def set_normal_constraint(self):
        """Constrain all atoms to move only in z-direction. Note that this will only affect certain file formats"""
        self.normal_constraint = True
    
    def set_fix_block(self,fixblock):
        """Constrain atoms in block of width given in distance units to remain fixed

        :param fixblock: length of blocks of atoms to fix
        :type fixblock: float
        """        
        self.fix_block = fixblock
        self.blocks_fixed = True
    
    def calculate_atom_arrays(self, tol=GBSUPERCELL_TOL, vis_type=None, gamma_surf=False, overfill=False):
        """Method for calculating the final atom arrays and supercell specification ready for use, and prior to writing out.
        
        Note that this method does the majority of the work of this class. This makes the method large and complicated, but at least keeps all the 
        logic for handling microscopic degrees of freedom, constraints, etc in one place, i.e. features like the vacuum layer, whether certain atoms 
        are constrained, relative grain shifts and excess volume at the boundaries  are all implemented in the process of calculating the atomic coordinates. 

        If debug information is requested, by calling set_debug(), then this method also checks for equivalence between the two boundaries and reports the result of the test.

        A final complexity is that the method works slightly differently depending on whether it is called to build a single instance of a boundary or as part of a gamma surface calculation.
        If the latter, then the initial set of atomic coordinates, prior to the application of various types of microscopic adjustment, is retained in order to increase efficiency. 
        See specification of the parameter gamma_surf, below.

        :param tol: Tolerance for atom position tests, defaults to GBSUPERCELL_TOL
        :type tol: float, optional
        :param vis_type: Request visualisation for display in a Jupyter notebook. Can be '2d' or '3d', defaults to None
        :type vis_type: str, optional
        :param gamma_surf: Set to True if this function is called from gbcalculation.gamma_surface_build(), defaults to False. This greatly increases the efficiency of gamma surface calculations by avoiding repeated recalculstation of the atom coordinates
        :type gamma_surf: bool, optional
        :param overfill: set to True to overfill the box. WARNING! useful for visualisations, but will probably break periodicity with periodic boundaries! Defaults to False
        :type overfill: bool, optional
        :raises RuntimeError: "Number of repeats in crystals must be set before filling with atoms", if set_supercell_size() has not been called
        :raises RuntimeError: "Incorrect value for vis_type in gbsupercell.calculate_atom_arrays()"
        """        
        if not self.supercell_size_set:
            raise RuntimeError("Number of repeats in crystals must be set before filling with atoms")
        
        if self.debug: 
            print("----------------------------------------------------")
            print("gbsupercell.calculate_atom_arrays() debug:")
            print("******************************************")
        
        if not gamma_surf or not self.atom_arrays_calculated:
            # Generate initial position arrays
            #fill_repeats_black = ct.get_repeats(self.supercell,self.a*self.lattice_basis_black,self.a)
            fill_repeats_black = ct.get_repeats(self.supercell/self.a,self.lattice_basis_black)
            num_atoms_black, r_black, atom_types_black = ct.fill_box(self.supercell,self.a*self.lattice_basis_black, self.grainboundary.csl.lattice.basis_coords, fill_repeats_black, self.grainboundary.csl.lattice.atom_types[:], overfill=overfill)
            r_black = np.array(r_black)
            atom_types_black = np.array(atom_types_black)
            
            #fill_repeats_white = ct.get_repeats(self.supercell,self.a*self.lattice_basis_white,self.a)
            fill_repeats_white = ct.get_repeats(self.supercell/self.a,self.lattice_basis_white)
            num_atoms_white, r_white, atom_types_white = ct.fill_box(self.supercell,self.a*self.lattice_basis_white, self.grainboundary.csl.lattice.basis_coords, fill_repeats_white, self.grainboundary.csl.lattice.atom_types[:], overfill=overfill)
            r_white = np.array(r_white)
            atom_types_white = np.array(atom_types_white)
            
            if self.debug:
                print("Filling repeats: Black [(%d -> %d),(%d -> %d),(%d -> %d)]" % (fill_repeats_black[0,0],fill_repeats_black[0,1],fill_repeats_black[1,0],fill_repeats_black[1,1],fill_repeats_black[2,0],fill_repeats_black[2,1]))
                print("                 White [(%d -> %d),(%d -> %d),(%d -> %d)]" % (fill_repeats_white[0,0],fill_repeats_white[0,1],fill_repeats_white[1,0],fill_repeats_white[1,1],fill_repeats_white[2,0],fill_repeats_white[2,1]))
            
            if gamma_surf:
                self.r_black_orig = np.copy(r_black)
                self.atom_types_black_orig = np.copy(atom_types_black)
                self.num_atoms_black_orig = num_atoms_black
                self.r_white_orig = np.copy(r_white)
                self.atom_types_white_orig = np.copy(atom_types_white)
                self.num_atoms_white_orig = num_atoms_white
                if self.debug:
                    print("First supercell in gamma surface construction, copying atom positions etc.")
        else:
            r_black = np.copy(self.r_black_orig)
            atom_types_black = np.copy(self.atom_types_black_orig)
            num_atoms_black = self.num_atoms_black_orig
            r_white = np.copy(self.r_white_orig)
            atom_types_white = np.copy(self.atom_types_white_orig)
            num_atoms_white = self.num_atoms_white_orig
            
            if self.debug:
                print("After first supercell in gamma surface construction, so reusing old atom positions etc.")
        
        atom_shift = np.zeros(3, dtype=float) # This vector will accumulate the displacement needed to return corner atom to origin of supercell
        
        if self.debug:
            print("Original numbers of atoms: Black %6.1f, White %6.1f" % (num_atoms_black, num_atoms_white))
        if self.bicrystal_shifted:
            if self.debug:
                print("Shifting white crystal by [ %6.3f, %6.3f, %6.3f ]" % (self.shift[0], self.shift[1], self.shift[2]))
            for i in range(num_atoms_white):
                if r_white[i,2] < self.supercell_centre:
                    r_white[i,:] = r_white[i,:] - 1.0*self.shift
                else:
                    r_white[i,:] = r_white[i,:] + 1.0*self.shift
            #Adjust supercell size
            supercell_adjustment = ct.components_in_basis(2.0*self.shift,self.supercell)
            # if self.debug:
            #     print("    Requires adjustment of supercell by [ %6.3f, %6.3f, %6.3f ]" % (supercell_adjustment[0], supercell_adjustment[1], supercell_adjustment[2]))
            for s in range(3):
                self.supercell[2,:] = self.supercell[2,:] + supercell_adjustment[s]*self.supercell[s,:]
            atom_shift = atom_shift + 1.0*self.shift
            
        if self.boundary_plane_shifted:
            if self.debug:
                print("Shifting boundary planes by %6.3f" % (self.boundary_plane_shift))
            self.boundary_plane_z[0] = self.boundary_plane_z[0] - self.boundary_plane_shift
            self.boundary_plane_z[1] = self.boundary_plane_z[1] + self.boundary_plane_shift
            
        
        # Pare atoms according to grain boundary position
        if not overfill:
            self.r_black = r_black[(r_black[:,2]>self.boundary_plane_z[0]+tol) & (r_black[:,2]<self.boundary_plane_z[1]-tol),:]
        else:
            self.r_black = r_black[(r_black[:,2]>=self.boundary_plane_z[0]-tol) & (r_black[:,2]<=self.boundary_plane_z[1]+tol),:]
        self.r_white = r_white[(r_white[:,2]<=self.boundary_plane_z[0]+tol) | (r_white[:,2]>=self.boundary_plane_z[1]-tol),:]
        if not overfill:
            self.atom_types_black = atom_types_black[(r_black[:,2]>self.boundary_plane_z[0]+tol) & (r_black[:,2]<self.boundary_plane_z[1]-tol)]
        else:
            self.atom_types_black = atom_types_black[(r_black[:,2]>=self.boundary_plane_z[0]-tol) & (r_black[:,2]<=self.boundary_plane_z[1]+tol)]
        self.atom_types_white = atom_types_white[(r_white[:,2]<=self.boundary_plane_z[0]+tol) | (r_white[:,2]>=self.boundary_plane_z[1]-tol)]

        self.num_atoms_black = np.size(self.r_black,0)
        self.num_atoms_white = np.size(self.r_white,0)
            
        self.atom_arrays_calculated = True
        
        if self.debug:
            print("Numbers of atoms after pareing: Black %6.1f, White %6.1f" % (self.num_atoms_black, self.num_atoms_white))
            
        if self.expansion_set:
            if np.size(self.expansion)==1:
                if self.debug:
                    print("Expanding bicrystal at each boundary by %6.3f" % (self.expansion))
                for i in range(self.num_atoms_white):
                    if self.r_white[i,2] < self.boundary_plane_z[0] - tol:
                        self.r_white[i,2] = self.r_white[i,2] - 1.0*self.expansion
                    elif self.r_white[i,2] < self.boundary_plane_z[0] + tol:
                        self.r_white[i,2] = self.r_white[i,2] - 0.5*self.expansion
                    elif self.r_white[i,2] > self.boundary_plane_z[1] + tol:
                        self.r_white[i,2] = self.r_white[i,2] + 1.0*self.expansion
                    elif self.r_white[i,2] > self.boundary_plane_z[1] - tol:
                        self.r_white[i,2] = self.r_white[i,2] + 0.5*self.expansion
                #Adjust supercell size
                self.supercell[2,2] = self.supercell[2,2] + 2.0*self.expansion
                atom_shift[2] = atom_shift[2] + 1.0*self.expansion
            self.boundary_plane_z[0] = self.boundary_plane_z[0] + 0.5*self.expansion
            self.boundary_plane_z[1] = self.boundary_plane_z[1] + 1.5*self.expansion
            
        # Implement atom position adjustments
        self.r_white = self.r_white + atom_shift
        self.r_black = self.r_black + atom_shift
                
        if self.vacuum_set:
            if self.debug:
                print("Adjusting to single-boundary configuration and adding vacuum at each end of %6.3f" % (self.vacuum))
            shift = (1.0 - self.boundary_plane_z[1] / self.supercell[2,2]) * self.supercell[2,:]
            # First shift atoms through cell and rewrap to give a single boundary.
            for i in range(self.num_atoms_black):
                self.r_black[i,:] = ct.wrap_to_cell(self.supercell,self.r_black[i,:] + shift, overfill=overfill)
            if not overfill:
                for i in range(self.num_atoms_white):
                    self.r_white[i,:] = ct.wrap_to_cell(self.supercell,self.r_white [i,:] + shift, overfill=overfill)
            else:
                extra_shift = np.array([0.0,0.0,1.0]) # If overfilling the cell, then apply an extra shift to ensure correct wrapping
                for i in range(self.num_atoms_white):
                    self.r_white[i,:] = ct.wrap_to_cell(self.supercell,self.r_white [i,:] + shift + extra_shift, overfill=overfill) - extra_shift
            self.boundary_plane_z = self.boundary_plane_z + shift[2]
            # Now add vacuum
            vacuum_shift = (self.vacuum/self.supercell[2,2]) * self.supercell[2,:]
            self.r_black = self.r_black + vacuum_shift
            self.r_white = self.r_white + vacuum_shift
            self.boundary_plane_z = self.boundary_plane_z + vacuum_shift[2]
            self.supercell[2,:] = (1.0+ 2.0*self.vacuum/self.supercell[2,2]) * self.supercell[2,:]
        
        if self.blocks_fixed:
            if self.debug:
                print("Flagging atoms for fixing in place in blocks of depth %6.3f" % (self.fix_block))
            self.fix_flag_black = np.zeros(self.num_atoms_black, dtype=bool)
            self.fix_flag_white = np.zeros(self.num_atoms_white, dtype=bool)
            if not self.vacuum_set:
                self.fix_flag_white[(self.r_white[:,2]<self.fix_block/2.0) | (self.r_white[:,2]>(self.supercell[2,2] - self.fix_block/2.0))] = True
                self.fix_flag_black[(self.r_black[:,2]>(self.supercell[2,2]/2.0 - self.fix_block/2.0)) & (self.r_black[:,2]<(self.supercell[2,2]/2.0 + self.fix_block/2.0))] = True
            else:
                self.fix_flag_white[(self.r_white[:,2]>(self.boundary_plane_z[0] + self.vacuum - self.fix_block)/2.0) & (self.r_white[:,2]<(self.boundary_plane_z[0] + self.vacuum + self.fix_block)/2.0)] = True
                self.fix_flag_black[(self.r_black[:,2]>(self.boundary_plane_z[0] + self.boundary_plane_z[1] - self.fix_block)/2.0) & (self.r_black[:,2]<(self.boundary_plane_z[0] + self.boundary_plane_z[1] + self.fix_block)/2.0)] = True
        
        self.num_particle_types = self.grainboundary.lattice.num_atom_types
        if self.blocks_fixed:
            self.atom_types_black[self.fix_flag_black] = self.atom_types_black[self.fix_flag_black] + self.num_particle_types
            self.atom_types_white[self.fix_flag_white] = self.atom_types_white[self.fix_flag_white] + self.num_particle_types
            self.num_particle_types = 2*self.num_particle_types
            
        # Check boundary equivalence
        if self.debug:
            print("")
            print("Checking boundary equivalence")
        equivalence_tol = 1e-2
        check_range = self.a
        check_a = np.concatenate((
            self.r_black[(self.r_black[:,2]>(self.boundary_plane_z[0] - check_range)) & (self.r_black[:,2]<(self.boundary_plane_z[0] + check_range)),:],
            self.r_white[(self.r_white[:,2]>(self.boundary_plane_z[0] - check_range)) & (self.r_white[:,2]<(self.boundary_plane_z[0] + check_range)),:]
        ))
        check_b = np.concatenate((
            self.r_black[(self.r_black[:,2]>(self.boundary_plane_z[1] - check_range)) & (self.r_black[:,2]<(self.boundary_plane_z[1] + check_range)),:],
            self.r_white[(self.r_white[:,2]>(self.boundary_plane_z[1] - check_range)) & (self.r_white[:,2]<(self.boundary_plane_z[1] + check_range)),: ]
        ))
        c = 0.5*np.sum(self.supercell, axis=0)
        check_a = c - (check_a - c)
        self.boundaries_equivalent = True
        max_sep = 0.0
        for i in range(len(check_a)):
            check_a[i,:] = ct.wrap_to_cell(self.supercell, check_a[i,:])
            if(abs(check_a[i,2] - self.boundary_plane_z[1])<0.9*check_range):
                min_sep = 999.0
                for j in range(len(check_b)):
                    sep = np.linalg.norm(ct.wrap_vector_to_cell(self.supercell,check_b[j,:]-check_a[i,:]))
                    if sep < min_sep:
                        min_sep = sep
                        min_vec = ct.wrap_vector_to_cell(self.supercell,check_b[j,:]-check_a[i,:])
                if (min_sep > equivalence_tol):
                    self.boundaries_equivalent = False
                if min_sep > max_sep:
                    max_sep = min_sep  # Required only for debug output
                    max_sep_a = ct.wrap_to_cell(self.supercell, check_a[i,:])
                    max_sep_b = ct.wrap_to_cell(self.supercell, check_b[j,:])
                    max_sep_vec = ct.wrap_vector_to_cell(self.supercell, max_sep_b - max_sep_a)
                    
                    
        if self.debug and not self.vacuum_set:
            if self.boundaries_equivalent:
                print("Grain boundary eqivalence test: PASS")
            else:
                print("Grain boundary eqivalence test: FAIL, max separation =  %6.3f" % (max_sep))
                print("    Problem atom in boundary 1:", max_sep_a)
                print("    Problem atom in boundary 2:", max_sep_b)
                print("    Vector between problem atoms:", max_sep_vec)
                
            # Code below provides some visual feedback on boundary equivalence suitable for use in a notebook
            if vis_type is not None:
                data = []
                x,y,z = ct.vis_data_box_ppp(self.supercell, np.array([[0,1],[0,1],[0,1]]))
                if vis_type == '3d':
                    data.append(go.Scatter3d(x = x, y = y, z = z,  mode = 'lines', name = 'Supercell', line = dict(width = 2, color = 'rgb(0, 0, 0)')))
                    data.append(go.Scatter3d(x = check_a[:,0], y = check_a[:,1], z = check_a[:,2], mode = 'markers', name = 'Atoms black', marker = dict(size = 7, color = 'rgb(255, 0, 0)')))
                    data.append(go.Scatter3d(x = check_b[:,0], y = check_b[:,1], z = check_b[:,2], mode = 'markers', name = 'Atoms white', marker = dict(size = 5, color = 'rgb(0, 0, 255)')))
                elif vis_type == '2d':
                    data.append(go.Scatter(x = y, y = z,  mode = 'lines', name = 'Supercell', line = dict(width = 2, color = 'rgb(0, 0, 0)')))
                    data.append(go.Scatter(x = check_a[:,1], y = check_a[:,2], mode = 'markers', name = 'Atoms black', marker = dict(size = 7, color = 'rgb(255, 0, 0)')))
                    data.append(go.Scatter(x = check_b[:,1], y = check_b[:,2], mode = 'markers', name = 'Atoms white', marker = dict(size = 5, color = 'rgb(0, 0, 255)')))
                else:
                    raise RuntimeError("Incorrect value for vis_type in gbsupercell.calculate_atom_arrays()")
                layout = go.Layout(width = 800, height = 500,title = "Grain boundary equivalence check", xaxis = dict( nticks = 10, domain = [0, 0.9]),yaxis = dict(scaleanchor = "x"))
                plotly.offline.iplot({ "data": data,"layout": layout})
                
    def get_supercell_definition(self):
        """Return details of a supercell. The atom_data array contains details of both black and white atoms. First three columns are the coordinates. The fourth column is the type.

        :return: num_atoms, self.supercell, atom_data
        :rtype: int, ndarray((3,3),dtype=float), ndarray((N,4),dtype=float)
        """        
        n_black = np.shape(self.r_black)[0]
        n_white = np.shape(self.r_white)[0]
        num_atoms = n_black + n_white
        atom_data = np.zeros((num_atoms,4))
        for i in range(n_black):
            atom_data[i,0:3] = self.r_black[i,:]
            atom_data[i,3] = self.atom_types_black[i]
        for i in range(n_white):
            atom_data[n_black+i,0:3] = self.r_white[i,:]
            atom_data[n_black+i,3] = self.atom_types_white[i]
        return num_atoms, self.supercell, atom_data
            
    def write_lammps(self, filename='lammps.txt', skew=True):
        """Write out the supercell in Lammps format

        :param filename: Name of file to write to, defaults to 'lammps.txt'
        :type filename: str, optional
        :param skew: Flag to indicate if a highly skewed supercell is permitted (True) or if skew should be minimised (False), defaults to True
        :type skew: bool, optional
        """        
        fo = open(filename,'w')
        header = '#Lammps coordinate file for CSL - Axis: ['
        header = header + str(int(self.grainboundary.csl.axis[0])) + ' ' + str(int(self.grainboundary.csl.axis[1])) + ' ' + str(int(self.grainboundary.csl.axis[2])) + ']'
        header = header + '  Angle: (' + str(int(self.grainboundary.csl.angle_indices[0])) + ', ' + str(int(self.grainboundary.csl.angle_indices[1])) + ') -> ' + str(self.grainboundary.csl.angle*180/np.pi)
        header = header + '  Plane indices: [' + str(self.grainboundary.boundary_indices[0]) + ' ' + str(self.grainboundary.boundary_indices[1]) + ' ' + str(self.grainboundary.boundary_indices[2]) + ']'
        header = header + '  z-position: [' + str(self.boundary_plane_z[0]) + ' ' + str(self.boundary_plane_z[1]) + ']'
        #header = header + ' (' + self.grainboundary.csl.boundary_type + ')'
        header = header + '  Repeats: ' + str(int(self.repeats_black[0])) + ' ' + str(int(self.repeats_white[1])) + ' (' + str(int(self.repeats_black[2])) + ', ' + str(int(self.repeats_white[2])) + ') (black, white)'
        if self.blocks_fixed:
            header = header + '  Fixed: ' + str(np.around(self.fix_block,3))
        if self.vacuum_set:
            header = header + '  Vacuum: ' + str(np.around(self.vacuum,3))
        if self.normal_constraint:
            header = header + '  Normal const applied'
        header = header + '  Boundary equiv: ' + str(self.boundaries_equivalent)
        fo.write(header)
        fo.write('\n')
        fo.write(str(self.num_atoms_black + self.num_atoms_white) + ' atoms\n')
        fo.write('\n')
        fo.write(str(self.num_particle_types) + ' atom types\n')
        fo.write('\n')
        
        fo.write('0.0 ' + str(self.supercell[0,0]) + ' xlo xhi\n')
        fo.write('0.0 ' + str(self.supercell[1,1]) + ' ylo yhi\n')
        fo.write('0.0 ' + str(self.supercell[2,2]) + ' zlo zhi\n')
        if skew:
            fo.write(str(self.supercell[1,0]) + ' ' + str(self.supercell[2,0]) + ' ' + str(self.supercell[2,1]) + ' xy xz yz\n')
        else:
            # Adjust skew parameters to avoid overly large skew in Lammps cell
            skew_indices = [(1,0), (2,0), (2,1)]
            skew_lengths = [(0,0), (0,0), (1,1)]
            new_supercell = np.copy(self.supercell)
            for s in range(3):
                while new_supercell[skew_indices[s]] > 0.5*new_supercell[skew_lengths[s]]:
                    new_supercell[skew_indices[s]] = new_supercell[skew_indices[s]] - new_supercell[skew_lengths[s]]
                while new_supercell[skew_indices[s]] < -0.5*new_supercell[skew_lengths[s]]:
                    new_supercell[skew_indices[s]] = new_supercell[skew_indices[s]] + new_supercell[skew_lengths[s]]   
            fo.write(str(new_supercell[1,0]) + ' ' + str(new_supercell[2,0]) + ' ' + str(new_supercell[2,1]) + ' xy xz yz\n')
        fo.write('\n')
        fo.write('Atoms\n')
        fo.write('\n')
        count = 1
        for i in range(self.num_atoms_black):
            fo.write(str(count) + ' ' + str(self.atom_types_black[i]) + ' ' + str(self.r_black[i,0]) + ' ' + str(self.r_black[i,1]) + ' ' + str(self.r_black[i,2]) + '\n')
            count = count + 1
        for i in range(self.num_atoms_white):
            fo.write(str(count) + ' ' + str(self.atom_types_white[i]) + ' ' + str(self.r_white[i,0]) + ' ' + str(self.r_white[i,1]) + ' ' + str(self.r_white[i,2]) + '\n')
            count = count + 1
        
        fo.flush()
        fo.close()
        
    def write_vasp(self, filename='vasp.txt', fix_xy=False):
        """Write out the supercell in Vasp POSCAR format

        :param filename: Name of file to write to, defaults to 'vasp.txt'
        :type filename: str, optional
        :param fix_xy: Flag to request selective dynamics and constrain particles to move only in z-direction (perpendicualr to boundary plane), defaults to False
        :type fix_xy: bool, optional
        """
        fo = open(filename,'w')
        header = '#VASP coordinate file for CSL - Axis: ['
        header = header + str(int(self.grainboundary.csl.axis[0])) + ' ' + str(int(self.grainboundary.csl.axis[1])) + ' ' + str(int(self.grainboundary.csl.axis[2])) + ']'
        header = header + '  Angle: (' + str(int(self.grainboundary.csl.angle_indices[0])) + ', ' + str(int(self.grainboundary.csl.angle_indices[1])) + ') -> ' + str(self.grainboundary.csl.angle*180/np.pi)
        header = header + '  Plane indices: [' + str(self.grainboundary.boundary_indices[0]) + ' ' + str(self.grainboundary.boundary_indices[1]) + ' ' + str(self.grainboundary.boundary_indices[2]) + '] '
        header = header + '  z-position: [' + str(self.boundary_plane_z[0]) + ' ' + + str(self.boundary_plane_z[1]) + ']'
        #header = header + ' (' + self.grainboundary.csl.boundary_type + ')'
        header = header + '  Repeats: ' + str(int(self.repeats_black[0])) + ' ' + str(int(self.repeats_white[1])) + ' (' + str(int(self.repeats_black[2])) + ', ' + str(int(self.repeats_white[2])) + ') (black, white)'
        if self.blocks_fixed:
            header = header + '  Fixed: ' + str(np.around(self.fix_block,3))
        if self.vacuum_set:
            header = header + '  Vacuum: ' + str(np.around(self.vacuum,3))
        if self.normal_constraint:
            header = header + '  Normal const applied'
        header = header + '  Boundary equiv: ' + str(self.boundaries_equivalent)
        fo.write(header)
        fo.write('\n')
        # Lattice parameter already factored into supercell and coordinates
        fo .write('1.0\n')
        # Supercell shape
        order = [0,1,2]
        if np.dot(self.supercell[0,:],np.cross(self.supercell[1,:],self.supercell[2,:])) < 0:
            order = [1,0,2]
        for s in range(3):
            for t in range(3):
                fo.write(str(self.supercell[order[s],t]) + ' ')
            fo.write('\n')
        # Atom positions
        fo .write(str(self.num_atoms_black + self.num_atoms_white) + '\n')
        if fix_xy or self.blocks_fixed:
            fo .write('Selective dynamics\n')
        fo.write('Cartesian\n')
        for i in range(self.num_atoms_black):
            fo.write(str(self.r_black[i,0]) + ' ' + str(self.r_black[i,1]) + ' ' + str(self.r_black[i,2]))
            if self.blocks_fixed and self.atom_types_black[i]>self.num_particle_types/2:
                fo.write(' F F F')
            elif fix_xy:
                fo.write(' F F T')
            else:
                fo.write(' T T T')
            fo.write('\n')
        for i in range(self.num_atoms_white):
            fo.write(str(self.r_white[i,0]) + ' ' + str(self.r_white[i,1]) + ' ' + str(self.r_white[i,2]))
            if self.blocks_fixed and self.atom_types_white[i]>self.num_particle_types/2:
                fo.write(' F F F')
            elif fix_xy:
                fo.write(' F F T')
            else:
                fo.write(' T T T')
            fo.write('\n')
            
        fo.flush()
        fo.close()
        
    try:
        def visualise_3d(self):
            """Generate a 3D plot showing the Supercell geometry"""
            data = []
            colors = ['rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)', 'rgb(0, 0, 0)', 'rgb(155, 155, 155)']
    
            x,y,z = ct.vis_data_box_ppp(self.a*self.supercell_unit_cell, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Supercell unit cell',
                line = dict(width = 2, color = colors[4])
            )
            data.append(trace)
    
            x,y,z = ct.vis_data_box_ppp(self.supercell, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Supercell',
                line = dict(width = 2, color = colors[3])
            )
            if self.atom_arrays_calculated:
                data.append(trace)
        
                name = 'Atoms black'
                trace = go.Scatter3d(
                    x = self.r_black[:,0], y = self.r_black[:,1], z = self.r_black[:,2], mode = 'markers', name = name,
                    marker = dict(size = 5, color = colors[0])
                )
                data.append(trace)
        
                name = 'Atoms white'
                trace = go.Scatter3d(
                    x = self.r_white[:,0], y = self.r_white[:,1], z = self.r_white[:,2], mode = 'markers', name = name,
                    marker = dict(size = 5, color = colors[2])
                )
                data.append(trace)
    
            layout = go.Layout(
                width = 800, height = 500,
                title = "Grain boundary supercell",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )
        
            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })
    
        def visualise_2d(self):
            """Generate a 2D plot showing the Supercell geometry"""
            data = []
            colors = ['rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)', 'rgb(0, 0, 0)', 'rgb(155, 155, 155)', 'rgb(255, 255, 255)']
    

        
            x,y,z = ct.vis_data_box_ppp(self.a*self.supercell_unit_cell, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter(
                x = y, y = z,  mode = 'lines', name = 'Supercell unit cell',
                line = dict(width = 2, color = colors[4])
            )
            data.append(trace)
        
            points = []
            points.append(self.original_boundary_plane_z[0]/self.original_supercell[2,2]*self.original_supercell[2,:])
            points.append(self.original_boundary_plane_z[0]/self.original_supercell[2,2]*self.original_supercell[2,:] + self.original_supercell[1,:])
            points.append([None,None,None])
            points.append(self.original_boundary_plane_z[1]/self.original_supercell[2,2]*self.original_supercell[2,:])
            points.append(self.original_boundary_plane_z[1]/self.original_supercell[2,2]*self.original_supercell[2,:] + self.original_supercell[1,:])
            points.append([None,None,None])
            r = np.array(points)
            trace = go.Scatter(
                x = r[:,1], y = r[:,2],  mode = 'lines', name = 'Original Boundary Planes',
                line = dict(width = 2, dash='dash', color = colors[3])
            )
            data.append(trace)
        
            x,y,z = ct.vis_data_box_ppp(self.original_supercell, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter(
                x = y, y = z,  mode = 'lines', name = 'Original Supercell',
                line = dict(width = 2, dash='dash', color = colors[3])
            )
            data.append(trace)
        
            points = []
            points.append(self.boundary_plane_z[0]/self.supercell[2,2]*self.supercell[2,:])
            points.append(self.boundary_plane_z[0]/self.supercell[2,2]*self.supercell[2,:] + self.supercell[1,:])
            points.append([None,None,None])
            points.append(self.boundary_plane_z[1]/self.supercell[2,2]*self.supercell[2,:])
            points.append(self.boundary_plane_z[1]/self.supercell[2,2]*self.supercell[2,:] + self.supercell[1,:])
            points.append([None,None,None])
            r = np.array(points)
            trace = go.Scatter(
                x = r[:,1], y = r[:,2],  mode = 'lines', name = 'Boundary Planes',
                line = dict(width = 2, dash='solid', color = colors[3])
            )
            data.append(trace)
        
            x,y,z = ct.vis_data_box_ppp(self.supercell, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter(
                x = y, y = z,  mode = 'lines', name = 'Supercell',
                line = dict(width = 2, color = colors[3])
            )
            data.append(trace)
        
            if self.atom_arrays_calculated:
                name = 'Atoms black'
                trace = go.Scatter(
                    x = self.r_black[:,1], y = self.r_black[:,2], mode = 'markers', name = name,
                    marker = dict(size = 7, color = colors[0])
                )
                data.append(trace)
        
                name = 'Atoms white'
                trace = go.Scatter(
                    x = self.r_white[:,1], y = self.r_white[:,2], mode = 'markers', name = name,
                    marker = dict(size = 5, color = colors[2])
                )
                data.append(trace)
        
                if self.blocks_fixed:
                    name = 'Atoms fixed white'
                    trace = go.Scatter(
                        x = self.r_white[self.fix_flag_white,1], y = self.r_white[self.fix_flag_white,2], mode = 'markers', name = name,
                        marker = dict(size = 2, color = colors[5])
                    )
                    data.append(trace)
                    name = 'Atoms fixed black'
                    trace = go.Scatter(
                        x = self.r_black[self.fix_flag_black,1], y = self.r_black[self.fix_flag_black,2], mode = 'markers', name = name,
                        marker = dict(size = 2, color = colors[5])
                    )
                    data.append(trace)
    
            layout = go.Layout(
                width = 800, height = 500,
                title = "Grain boundary supercell",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )
        
            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })
    except NameError:
        pass



