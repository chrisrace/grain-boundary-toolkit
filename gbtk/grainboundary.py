# grainboundary.py
""" Module to handle definitions of grain boundaries. A boundary is described by 5 macroscopic degrees of freedom. 
It may or may not have a CSL associated with the axis-angle combination
Author:  Chris Race
Date:    4th January 2017
Contact: christopher.race@manchester.ac.uk
"""

import numpy as np
import math
import pandas as pd

from . import symmetry
from . import lattice
from . import spatialsearch
from . import crystaltools as ct
from . import csl as csl
#from . import symmetry as symmetry

# For visualisation only
try:
    import plotly
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)
except ModuleNotFoundError:
    pass

GRAINBOUNDARY_TOL = 1e-3
TIGHT_TOL = 1e-6

class GrainBoundary(object):
    """A grain holds details of the misorientation axis-angle combination and the boundary plane for a grain boundary.
    Where the axis-angle combination corresponds to a CSL it will also contain the csl object

    :ivar object lattice: lattice object defining lattice in which grain boundary exists
    :ivar [int,int,int] axis: Misorientation axis for grain boundary
    :ivar float angle: Misorientation angle (in radians)
    :ivar ndarray(2,dtype=int) angle_indices: specification of misorientation angle via a pair of integers m.n
    :ivar object csl: csl object defining the coincident site lattice for the grain boundary
    :ivar ndarray(3,dtype=float) boundary_normal_black: Normal to boundary in black half of bicrystal, expressed in multiples of lattice basis vectors
    :ivar ndarray(3,dtype=float) boundary_normal_white: Normal to boundary in white half of bicrystal, expressed in multiples of lattice basis vectors
    :ivar ndarray(3,dtype=float) boundary_normal_cartesian_black: Normal to boundary in black half of bicrystal, in cartesian coordinates
    :ivar ndarray(3,dtype=float) boundary_normal_cartesian_white: Normal to boundary in white half of bicrystal, in cartesian coordinates
    :ivar ndarray(3,dtype=int) boundary_indices: Boundary plane specification as multiples of the three CSL cell vectors
    :ivar ndarray((3,3),dtype=float) boundary_cell_csl: Smallest repeat unit for bicrystal system, vectors of edges in rows, expressed in multiples of CSL cell vectors
    :ivar ndarray((3,3),dtype=float) boundary_cell_black: Smallest repeat unit for bicrystal system, in black system, vectors of edges in rows, expressed in multiples of lattice vectors
    :ivar ndarray((3,3),dtype=float) boundary_cell_white: Smallest repeat unit for bicrystal system, in white system, vectors of edges in rows, expressed in multiples of lattice vectors
    :ivar ndarray((3,3),dtype=float) boundary_cell_cartesian_black: Smallest repeat unit for bicrystal system, in black system, vectors of edges in rows, expressed in cartesians
    :ivar ndarray((3,3),dtype=float) boundary_cell_cartesian_white: Smallest repeat unit for bicrystal system, in white system, vectors of edges in rows, expressed in cartesians
    :ivar float cell_volume: Volume of smallest repeat unit for bicrystal system
    :ivar float cell_volume_csl: Number of CSL cells in smallest repeat unit for bicrystal system 
    :ivar float cell_volume_lattice: Number of lattice cells in smallest repeat unit for bicrystal system 
    :ivar float cell_volume_atoms: Number of atoms in smallest repeat unit for bicrystal system 
    :ivar string boundary_type: Broad boundary class - twist, tilt, symmetric (tilt) or mixed
    :ivar boolean search: 
    :ivar boolean debug: Flag set to True if debug information requested
    :ivar boolean angle_set: Flag to indicate if misorientation angle is set
    :ivar boolean axis_set: Flag to indicate if misorientation axis is set
    :ivar boolean csl_basis_set: Flag to indicate if csl_basis has been calculated (and csl_vectors_black, etc. populated)
    :ivar boolean boundary_plane_set: Flag to indicate if boundary plane has been set and normal vectors calculated.
    """
    
    def __init__(self, latticetype, lengths=None, angles=None):
        """Initialise an empty grain boundary object with the specified lattice type

        :param latticetype: Lattice type, passed to Lattice module
        :type latticetype: string
        :param lengths: Lengths of unit cell vectors, passed to Lattice module, defaults to None
        :type lengths: list of floats, optional
        :param angles: Angles between unit cell vectors, passed to Lattice module, defaults to None
        :type angles: list of floats, optional
        """        
        
        self.lattice = lattice.Lattice(latticetype, lengths, angles)
        self.axis = np.zeros((3), dtype=float)
        self.angle = 0.0
        
        # Control flags
        self.debug = False
        self.angle_set = False
        self.axis_set = False
        self.boundary_plane_set = False

    def set_debug(self):    
        """Turn on debug info
        """
        self.debug = True
        
    def unset_debug(self):
        """Turn off debug info
        """
        self.debug = False
        
    def set_axis(self, axis):
        """_summary_

        :param axis: Components of axis specified in multiples of the lattice cell vectors
        :type axis: [int,int,int]
        :raises RuntimeError: "Misorientation axis vector must have 3 components" if wrong number of components specified
        """   
        if len(axis) != 3:
            raise RuntimeError("Misorientation axis vector must have 3 components")    
        self.axis = np.array(axis)
        self.axis_cartesian = ct.vector_in_basis(self.axis,self.lattice.cell_vectors)
        self.axis_set = True
    
    def set_angle(self, angle):
        """Set the misorientation angle directly with a value in radians

        :param angle: Angle specified in radians
        :type angle: float
        """   
        self.angle = angle
        self.angle_set = True
    
    def set_angle_mn(self, m, n):
        """Set the misorientation angle via a pair of integers.
        Only works for cubic lattice types

        :param m: first parameter
        :type m: integer
        :param n: second parameter
        :type n: integer
        :raises RuntimeError: "Axis must be specified before setting angle using this method" if axis not already specified
        :raises RuntimeError: "Lattice type " + self.lattice.lattice_type + " not supported." if an unsupported lattice type is specified
        """  
        if not self.axis_set:
            raise RuntimeError("Axis must be specified before setting angle using this method")    
        self.angle = csl.calculate_theta(self.axis[0],self.axis[1],self.axis[2],m,n)
        self.angle_indices = np.array([m,n])
        self.angle_set = True
        
    def set_csl(self,csl_vectors_black=None,csl_vectors_white=None, search_limit=None):
        """_summary_

        :param csl_vectors_black: Basis vectors of CSL cell in black crystal, defaults to None - PROBABLY NOT CURRENTLY IMPLEMENTED
        :type csl_vectors_black: ndarray((3,3), dtype=float), optional
        :param csl_vectors_white: Basis vectors of CSL cell in white crystal, defaults to None - PROBABLY NOT CURRENTLY IMPLEMENTED
        :type csl_vectors_white: ndarray((3,3), dtype=float), optional
        :param search_limit: Limit for spatial search used to find CSL cell, defaults to None
        :type search_limit: integer, optional
        :raises RuntimeError: "Axis and angle must be set before setting csl basis", if either the misorientation axis or angle are not currently set
        """        
        if (not self.axis_set) or (not self.angle_set) :
            raise RuntimeError("Axis and angle must be set before setting csl basis")
        self.csl = csl.CSL('null')
        if self.debug:
            self.csl.set_debug()
        self.csl.lattice.copy_lattice(self.lattice)
        self.csl.set_axis(self.axis)
        self.csl.set_angle(self.angle)
        if hasattr(self, 'angle_indices'):
            self.csl.angle_indices = self.angle_indices
        if csl_vectors_black is None or csl_vectors_white is None: # This condition can probably be removed. 
            # Corresponding else block would, if implemented, allow direct specification of CSL cell, removing need for search
            self.csl.find_misorientation_rotation_matrix()
            if search_limit is not None:
                self.csl.enable_search(maxradius=search_limit)
            else:
                self.csl.enable_search()
                
            self.csl.find_csl_basis()
            
    def set_boundary_plane_csl(self,boundary_indices, orthogonal=False, tol=GRAINBOUNDARY_TOL, search_radius=10, min_angle=30.0/180.0*np.pi, suppress_error=False):
        """Set the boundary plane for a boundary with a csl.
        Plane is specified by components of plane normal in the CSL basis

        :param boundary_indices: List (or array) of three indices specifying the bounday plane normal in multiples of the CSL basis vectors
        :type boundary_indices: [int,int,int]
        :param orthogonal: Flag specifiying whether an orthogonal grain boundary cell is required, defaults to False
        :type orthogonal: bool, optional
        :param tol: Tolerance to use for determing orthogonality or non-equivalence of vectors, defaults to GRAINBOUNDARY_TOL
        :type tol: float, optional
        :param search_radius: Radius to use in search for basis vectors, defaults to 10
        :type search_radius: int, optional
        :param min_angle: Minimum angle to allow between grain boundary cell vectors (sets limit on extent of skewing), defaults to 30 degrees
        :type min_angle: float, optional
        :param suppress_error: Allows exception on failing to find a set of cell vectors to be ignored. Useful for catalogueing, defaults to False
        :type suppress_error: bool, optional
        :raises RuntimeError: "csl basis must be set before setting boundary plane via this method", if attempting to specify plane, without first setting up CSL. Run set_csl() first.
        :raises RuntimeError: "Could not find first in-plane vector - consider increasing search_radius", if no vector in GB plane can be found within search bounds
        :raises RuntimeError: "Could not find second in-plane vector - consider increasing search_radius", if second vector in GB plane can be found within search bounds (which may also be required to be orthogonal to first)
        :raises RuntimeError: "No vector out of boundary plane could be found - consider increasing search_radius", if the final vector cannot be found (which may also be required to be orthogonal to other two)
        :return: Flag indicating success of attempt to define GB cell
        :rtype: bool
        """        
        if (not self.csl.csl_basis_set):
            raise RuntimeError("csl basis must be set before setting boundary plane via this method")
        
        if self.debug: 
            print("----------------------------------------------------")
            print("grainboundary.set_boundary_plane_csl() debug:")
            print("*********************************************")
        
        self.boundary_indices = np.array(boundary_indices)
        self.boundary_cell_csl = np.zeros((3,3), dtype=int)
        self.boundary_cell_black = np.zeros((3,3), dtype=int)
        self.boundary_cell_white = np.zeros((3,3), dtype=int)
        self.boundary_cell_cartesian_black = np.zeros((3,3), dtype=int)
        self.boundary_cell_cartesian_white = np.zeros((3,3), dtype=int)
        
        cos_min_angle = np.cos(np.pi/2.0-min_angle)
        search_failed = False
        
        gb_normal = ct.vector_in_basis(boundary_indices, self.csl.csl_vectors_black)
        gb_normal_cartesian = ct.vector_in_basis(gb_normal, self.csl.lattice.cell_vectors)
        target_normal = gb_normal_cartesian/np.linalg.norm(gb_normal_cartesian)
        
        if self.debug:
            print("Grain boundary normal (lattice basis)")
            print('[ %6.0d %6.0d %6.0d ]' % (
                gb_normal[0], gb_normal[1], gb_normal[2] 
            ))
        
        search = spatialsearch.SpatialSearch(search_radius, basis_vectors=self.csl.csl_vectors_cartesian_black)
        found1 = False
        found2 = False
        index = 0
        # Look for first in-plane vector
        while (not found1) and (index < search.nspace):
            test_vector = ct.vector_in_basis(search.space[index,0:3], self.csl.csl_vectors_black)
            test_vector_cartesian = ct.vector_in_basis(test_vector, self.csl.lattice.cell_vectors)
            #print('***', search.space[index,0:3], test_vector, test_vector_cartesian, target_normal)
            if abs(np.dot(test_vector_cartesian,target_normal)/np.linalg.norm(test_vector_cartesian)) < tol:
                found1 = True
                self.boundary_cell_csl[0,:] = search.space[index,0:3]
                # print(target_normal)
                # print(test_vector_cartesian)
                # print(np.dot(test_vector_cartesian,target_normal)/np.linalg.norm(test_vector_cartesian))
            index = index + 1
        if not found1:
            if suppress_error:
                search_failed = True
            else:
                raise RuntimeError("Could not find first in-plane vector - consider increasing search_radius")
        if self.debug: 
            print("First vector (lattice basis)")
            print('[ %6.0d %6.0d %6.0d ]' % (
                ct.vector_in_basis(self.boundary_cell_csl[0,:], self.csl.csl_vectors_black)[0],
                ct.vector_in_basis(self.boundary_cell_csl[0,:], self.csl.csl_vectors_black)[1],
                ct.vector_in_basis(self.boundary_cell_csl[0,:], self.csl.csl_vectors_black)[2]
            ))
        
        target_vector_1 = ct.vector_in_basis(ct.vector_in_basis(self.boundary_cell_csl[0,:], self.csl.csl_vectors_black), self.csl.lattice.cell_vectors)
        target_vector_1 = target_vector_1/np.linalg.norm(target_vector_1)
        #look for second in-plane vector, which may also need to be orthogonal to first
        if orthogonal:
            if self.debug: 
                print("Looking for second vector perpendicular to first")
        while (not found2) and (index < search.nspace):
            test_vector = ct.vector_in_basis(search.space[index,0:3], self.csl.csl_vectors_black)
            test_vector_cartesian = ct.vector_in_basis(test_vector, self.csl.lattice.cell_vectors)
            if abs(np.dot(test_vector_cartesian,target_normal)/np.linalg.norm(test_vector_cartesian)) < tol:
                if orthogonal:
                    if abs(np.dot(test_vector_cartesian,target_vector_1)/np.linalg.norm(test_vector_cartesian)) < tol:
                        found2 = True
                        self.boundary_cell_csl[1,:] = search.space[index,0:3]
                else:
                    if abs(1.0 - abs(np.dot(test_vector_cartesian,target_vector_1)/np.linalg.norm(test_vector_cartesian))) > tol:
                        found2 = True
                        self.boundary_cell_csl[1,:] = search.space[index,0:3]
            index = index + 1
        if not found2:
            if suppress_error:
                search_failed = True
            else:
                raise RuntimeError("Could not find second in-plane vector - consider increasing search_radius")
        if self.debug: 
            print("Second vector (lattice basis)")
            print('[ %6.0d %6.0d %6.0d ]' % (
                ct.vector_in_basis(self.boundary_cell_csl[1,:], self.csl.csl_vectors_black)[0],
                ct.vector_in_basis(self.boundary_cell_csl[1,:], self.csl.csl_vectors_black)[1],
                ct.vector_in_basis(self.boundary_cell_csl[1,:], self.csl.csl_vectors_black)[2]
            ))
        
        # Look for an out-of-plane vector, which may need to be parallel to boundary normal
        index = 0
        found0 = False
        if self.debug: 
            if orthogonal:
                print("Looking for vector parallel to boundary normal")
        while (not found0) and (index < search.nspace):
            test_vector = ct.vector_in_basis(search.space[index,0:3], self.csl.csl_vectors_black)
            test_vector_cartesian = ct.vector_in_basis(test_vector, self.csl.lattice.cell_vectors)
            if orthogonal:
                if abs(1.0 - abs(np.dot(test_vector_cartesian,target_normal)/np.linalg.norm(test_vector_cartesian))) < tol:
                    found0 = True
                    self.boundary_cell_csl[2,:] = search.space[index,0:3]
            else:
                 if abs(np.dot(test_vector_cartesian,target_normal)/np.linalg.norm(test_vector_cartesian)) > cos_min_angle:
                     found0 = True
                     self.boundary_cell_csl[2,:] = search.space[index,0:3]
            index = index + 1
        if not found0:
            if suppress_error:
                search_failed = True
            else:
                raise RuntimeError("No vector out of boundary plane could be found - consider increasing search_radius")

        # Set vectors describing grain boundary cell
        self.boundary_cell_black = np.einsum('ij,jk',self.boundary_cell_csl,self.csl.csl_vectors_black)
        self.boundary_cell_white = np.einsum('ij,jk',self.boundary_cell_csl,self.csl.csl_vectors_white)
        self.boundary_cell_cartesian_black = np.einsum('ij,jk',self.boundary_cell_black,self.csl.lattice.cell_vectors)
        self.boundary_cell_cartesian_white = np.einsum('ij,jk',self.boundary_cell_white,self.csl.lattice.cell_vectors)
        # Calculate grain boundary cell volumes
        self.cell_volume = abs(np.dot(self.boundary_cell_cartesian_black[0],np.cross(self.boundary_cell_cartesian_black[1],self.boundary_cell_cartesian_black[2])))
        self.cell_volume_csl = self.cell_volume / self.csl.cell_volume
        self.cell_volume_lattice = self.cell_volume_csl * self.csl.cell_volume_lattice
        self.cell_volume_atoms = self.cell_volume_lattice * self.csl.lattice.num_basis
        
        if self.debug:
            print("Grain boundary cell vectors (csl basis)")
            for s in range(3):
                print('[ %6.0d %6.0d %6.0d ]' % (
                    self.boundary_cell_csl[s,0],self.boundary_cell_csl[s,1],self.boundary_cell_csl[s,2])
                    )
            print("Grain boundary cell vectors (lattice basis)")
            print('Black                      White')
            for s in range(3):
                print('[ %6.0d %6.0d %6.0d ]   [ %6.0d %6.0d %6.0d ]' % (
                    self.boundary_cell_black[s,0],self.boundary_cell_black[s,1],self.boundary_cell_black[s,2],
                    self.boundary_cell_white[s,0],self.boundary_cell_white[s,1],self.boundary_cell_white[s,2])
                    )
            print("Grain boundary cell vectors (cartesian basis)")
            print('Black                                       White')
            for s in range(3):
                print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
                    self.boundary_cell_cartesian_black[s,0],self.boundary_cell_cartesian_black[s,1],self.boundary_cell_cartesian_black[s,2],
                    self.boundary_cell_cartesian_white[s,0],self.boundary_cell_cartesian_white[s,1],self.boundary_cell_cartesian_white[s,2])
                    )
            print("Grain boundary cell Volume:")
            print("Cartesian: %12.6f, CSL cells: %6.0d, Lattice unit cells: %6.0d, Atoms: %6.0d" % (
                self.cell_volume, self.cell_volume_csl, self.cell_volume_lattice, self.cell_volume_atoms
            ))
            print("")
        
        # Now assign GB normal vectors
        self.boundary_normal_black = ct.vector_in_basis(boundary_indices, self.csl.csl_vectors_black)
        self.boundary_normal_cartesian_black = ct.vector_in_basis(self.boundary_normal_black, self.csl.lattice.cell_vectors)
        self.boundary_normal_cartesian_black = self.boundary_normal_cartesian_black / np.linalg.norm(self.boundary_normal_cartesian_black)
        
        self.boundary_normal_white = ct.vector_in_basis(boundary_indices, self.csl.csl_vectors_white)
        self.boundary_normal_cartesian_white = ct.vector_in_basis(self.boundary_normal_white, self.csl.lattice.cell_vectors)
        self.boundary_normal_cartesian_white = self.boundary_normal_cartesian_white / np.linalg.norm(self.boundary_normal_cartesian_white)
        
        if abs(1.0-abs(np.dot(self.boundary_normal_cartesian_black,self.axis_cartesian)/np.linalg.norm(self.axis_cartesian))) < tol:
            self.boundary_type = 'twist'
        elif abs(np.dot(self.boundary_normal_cartesian_black,self.axis_cartesian)/np.linalg.norm(self.axis_cartesian)) < tol:
            self.boundary_type = 'tilt'
            # Check if tilt boundary is symmetric (revise from old function)
            sym = symmetry.Symmetry(self.lattice.lattice_type)
            if sym.vector_equiv(self.boundary_normal_black,self.boundary_normal_white):
                self.boundary_type = 'symmetric'
        else:
            self.boundary_type = 'mixed'
        
        if self.debug:

            print("Boundary normals (lattice basis)")
            print('Black                                       White')
            #print('[ %6.0d %6.0d %6.0d ]   [ %6.0d %6.0d %6.0d ]' % (
            print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
                self.boundary_normal_black[0],self.boundary_normal_black[1],self.boundary_normal_black[2],
                self.boundary_normal_white[0],self.boundary_normal_white[1],self.boundary_normal_white[2])
                )
            print("Boundary normals (cartesian basis)")
            print('Black                                       White')
            print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
                self.boundary_normal_cartesian_black[0],self.boundary_normal_cartesian_black[1],self.boundary_normal_cartesian_black[2],
                self.boundary_normal_cartesian_white[0],self.boundary_normal_cartesian_white[1],self.boundary_normal_cartesian_white[2])
                )
            print("")
            print("Grain boundary type: " + self.boundary_type)
            print("")
        
        self.boundary_plane_set = True
        if search_failed:
            return False
        else:
            return True
    
    # def set_boundary_plane_csl_old(self,boundary_indices, orthogonal=False, tol=GRAINBOUNDARY_TOL, search_radius=10):
    #     """Set the boundary plane for a boundary with a csl.
    #     Plane is specified by intercepts of axes in the CSL basis"""
    #     if (not self.csl.csl_basis_set):
    #         raise RuntimeError("csl basis must be set before setting boundary plane via this method")
        
    #     if self.debug: 
    #         print("----------------------------------------------------")
    #         print("grainboundary.set_boundary_plane_csl() debug:")
    #         print("*********************************************")
        
    #     self.boundary_indices = np.array(boundary_indices)
    #     self.boundary_cell_csl = np.zeros((3,3), dtype=int)
    #     self.boundary_cell_black = np.zeros((3,3), dtype=int)
    #     self.boundary_cell_white = np.zeros((3,3), dtype=int)
    #     self.boundary_cell_cartesian_black = np.zeros((3,3), dtype=int)
    #     self.boundary_cell_cartesian_white = np.zeros((3,3), dtype=int)
    #     # Calculate vectors for boundary cell. First two vectors lie in GB plane, third vector should be most out of plane of the csl vectors
    #     zero_indices = np.where(np.array(boundary_indices) == 0)[0]
    #     non_zero_indices = np.where(np.array(boundary_indices) != 0)[0]
    #     num_zeros = np.size(zero_indices)
    #     if num_zeros == 2:
    #         self.boundary_cell_csl[2,non_zero_indices[0]] = boundary_indices[non_zero_indices[0]]
    #         self.boundary_cell_csl[0,zero_indices[0]] = 1
    #         self.boundary_cell_csl[1,zero_indices[1]] = 1
    #     elif num_zeros == 1:
    #         if abs(boundary_indices[non_zero_indices[0]]) < abs(boundary_indices[non_zero_indices[1]]):
    #             ordered_indices = [non_zero_indices[0], non_zero_indices[1]]
    #         else:
    #             ordered_indices = [non_zero_indices[1], non_zero_indices[0]]
    #         #self.boundary_cell_csl[2,ordered_indices[0]] = boundary_indices[ordered_indices[0]]
    #         self.boundary_cell_csl[2,ordered_indices[1]] = 1
    #         nonzero_product = np.abs(boundary_indices[ordered_indices[0]]*boundary_indices[ordered_indices[1]])
    #         self.boundary_cell_csl[0,ordered_indices[0]] = -int(nonzero_product/boundary_indices[ordered_indices[0]])
    #         self.boundary_cell_csl[0,ordered_indices[1]] = int(nonzero_product/boundary_indices[ordered_indices[1]])
    #         self.boundary_cell_csl[1,zero_indices[0]] = 1
    #     else:
    #         if (abs(boundary_indices[0]) >= abs(boundary_indices[1])) and (abs(boundary_indices[0]) >= abs(boundary_indices[2])):
    #             ordered_indices = [1,2,0]
    #         elif (abs(boundary_indices[1]) >= abs(boundary_indices[0])) and (abs(boundary_indices[1]) >= abs(boundary_indices[2])):
    #             ordered_indices = [2,0,1]
    #         else:
    #             ordered_indices = [0,1,2]
    #         self.boundary_cell_csl[2,ordered_indices[2]] = 1
    #         nonzero_product = np.abs(boundary_indices[ordered_indices[0]]*boundary_indices[ordered_indices[2]])
    #         self.boundary_cell_csl[0,ordered_indices[0]] = int(nonzero_product/boundary_indices[ordered_indices[0]])
    #         self.boundary_cell_csl[0,ordered_indices[2]] = -int(nonzero_product/boundary_indices[ordered_indices[2]])
    #         nonzero_product = np.abs(boundary_indices[ordered_indices[1]]*boundary_indices[ordered_indices[2]])
    #         self.boundary_cell_csl[1,ordered_indices[1]] = int(nonzero_product/boundary_indices[ordered_indices[1]])
    #         self.boundary_cell_csl[1,ordered_indices[2]] = -int(nonzero_product/boundary_indices[ordered_indices[2]])
        
    #     if orthogonal:
    #         if self.debug: 
    #             print("Looking for an orthogonal cell")
    #         # Find a combination of boundary cell vectors that gives an orthogonal cell
    #         boundary_cell = np.einsum('ij,jk',self.boundary_cell_csl,self.csl.csl_vectors_black)
    #         boundary_cell_cartesian = np.einsum('ij,jk',boundary_cell,self.csl.lattice.cell_vectors)
    #         target_normal = np.cross(boundary_cell_cartesian[0,:],boundary_cell_cartesian[1,:])
    #         target_normal = target_normal / np.linalg.norm(target_normal)
    #         maxradius = 10
    #         search = spatialsearch.SpatialSearch(search_radius,boundary_cell_cartesian)
    #         new_boundary_cell = np.zeros((3,3), dtype=float)
    #         # First find a vector parallel to boundary normal
    #         found = False
    #         index = 0
    #         while (not found) and (index < search.nspace):
    #             trial_vector = np.einsum('i,ij', search.space[index,0:3], boundary_cell_cartesian)
    #             if (1.0 - abs(np.dot(target_normal,trial_vector)/np.linalg.norm(trial_vector))) < tol:
    #                 new_boundary_cell[0,:] = search.space[index,0:3]
    #                 found = True
    #             index = index + 1
    #         if not found:
    #             raise RuntimeError("no vector parallel to boundary plane normal could be found - consider increasing search_radius")
    #         # Now find first perpendicular vector
    #         test_vector_1 = np.einsum('i,ij', new_boundary_cell[0,:], boundary_cell_cartesian)
    #         test_vector_1 = test_vector_1/np.linalg.norm(test_vector_1)
    #         found = False
    #         index = 0
    #         while (not found) and (index < search.nspace):
    #             trial_vector = np.einsum('i,ij', search.space[index,0:3], boundary_cell_cartesian)
    #             if abs(np.dot(test_vector_1,trial_vector)/np.linalg.norm(trial_vector)) < tol:
    #                 new_boundary_cell[1,:] = search.space[index,0:3]
    #                 found = True
    #             index = index + 1
    #         if not found:
    #             raise RuntimeError("no matching vector perpendicular to boundary plane normal could be found - consider increasing search_radius")
    #         # Now find first perpendicular vector
    #         test_vector_2 = np.einsum('i,ij', new_boundary_cell[1,:], boundary_cell_cartesian)
    #         test_vector_2 = test_vector_2/np.linalg.norm(test_vector_2)
    #         found = False
    #         index = 0
    #         while (not found) and (index < search.nspace):
    #             trial_vector = np.einsum('i,ij', search.space[index,0:3], boundary_cell_cartesian)
    #             if abs(np.dot(test_vector_1,trial_vector)/np.linalg.norm(trial_vector)) < tol:
    #                 if abs(np.dot(test_vector_2,trial_vector)/np.linalg.norm(trial_vector)) < tol:
    #                     new_boundary_cell[2,:] = search.space[index,0:3]
    #                     found = True
    #             index = index + 1
    #         if not found:
    #             raise RuntimeError("second matching vector perpendicular to boundary plane normal could be found - consider increasing search_radius")
    #         if self.debug: 
    #             print("Old boundary cell")
    #             print(self.boundary_cell_csl)
    #         self.boundary_cell_csl = np.einsum('ik,kj->ij',new_boundary_cell,self.boundary_cell_csl)
    #         if self.debug: 
    #             print("New boundary cell")
    #             print(self.boundary_cell_csl)
            
    #     # Set vectors describing grain boundary cell
    #     self.boundary_cell_black = np.einsum('ij,jk',self.boundary_cell_csl,self.csl.csl_vectors_black)
    #     self.boundary_cell_white = np.einsum('ij,jk',self.boundary_cell_csl,self.csl.csl_vectors_white)
    #     self.boundary_cell_cartesian_black = np.einsum('ij,jk',self.boundary_cell_black,self.csl.lattice.cell_vectors)
    #     self.boundary_cell_cartesian_white = np.einsum('ij,jk',self.boundary_cell_white,self.csl.lattice.cell_vectors)
    #     # Calculate grain boundary cell volumes
    #     self.cell_volume = abs(np.dot(self.boundary_cell_cartesian_black[0],np.cross(self.boundary_cell_cartesian_black[1],self.boundary_cell_cartesian_black[2])))
    #     self.cell_volume_csl = self.cell_volume / self.csl.cell_volume
    #     self.cell_volume_lattice = self.cell_volume_csl * self.csl.cell_volume_lattice
    #     self.cell_volume_atoms = self.cell_volume_lattice * self.csl.lattice.num_basis
        
    #     if self.debug:
    #         print("Grain boundary cell vectors (csl basis)")
    #         for s in range(3):
    #             print('[ %6.0d %6.0d %6.0d ]' % (
    #                 self.boundary_cell_csl[s,0],self.boundary_cell_csl[s,1],self.boundary_cell_csl[s,2])
    #                 )
    #         print("Grain boundary cell vectors (lattice basis)")
    #         print('Black                      White')
    #         for s in range(3):
    #             print('[ %6.0d %6.0d %6.0d ]   [ %6.0d %6.0d %6.0d ]' % (
    #                 self.boundary_cell_black[s,0],self.boundary_cell_black[s,1],self.boundary_cell_black[s,2],
    #                 self.boundary_cell_white[s,0],self.boundary_cell_white[s,1],self.boundary_cell_white[s,2])
    #                 )
    #         print("Grain boundary cell vectors (cartesian basis)")
    #         print('Black                                       White')
    #         for s in range(3):
    #             print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
    #                 self.boundary_cell_cartesian_black[s,0],self.boundary_cell_cartesian_black[s,1],self.boundary_cell_cartesian_black[s,2],
    #                 self.boundary_cell_cartesian_white[s,0],self.boundary_cell_cartesian_white[s,1],self.boundary_cell_cartesian_white[s,2])
    #                 )
    #         print("Grain boundary cell Volume:")
    #         print("Cartesian: %12.6f, CSL cells: %6.0d, Lattice unit cells: %6.0d, Atoms: %6.0d" % (
    #             self.cell_volume, self.cell_volume_csl, self.cell_volume_lattice, self.cell_volume_atoms
    #         ))
    #         print("")
        
    #     self.boundary_plane_set = True
        
    # def calculate_boundary_normals_old(self, tol=GRAINBOUNDARY_TOL):
    #     """Calculate boundary plane normals"""
    #     if self.debug: 
    #         print("----------------------------------------------------")
    #         print("grainboundary.calculate_boundary_normals() debug:")
    #         print("*************************************************")
    #     self.boundary_normal_cartesian_black = np.cross(self.boundary_cell_cartesian_black[0,:],self.boundary_cell_cartesian_black[1,:])
    #     self.boundary_normal_cartesian_black = self.boundary_normal_cartesian_black / np.linalg.norm(self.boundary_normal_cartesian_black)
    #     self.boundary_normal_cartesian_white = np.cross(self.boundary_cell_cartesian_white[0,:],self.boundary_cell_cartesian_white[1,:])
    #     self.boundary_normal_cartesian_white = self.boundary_normal_cartesian_white / np.linalg.norm(self.boundary_normal_cartesian_white)
    #     self.boundary_normal_black = ct.indices_in_basis(self.boundary_normal_cartesian_black,self.csl.lattice.cell_vectors)
    #     self.boundary_normal_white = ct.indices_in_basis(self.boundary_normal_cartesian_white,self.csl.lattice.cell_vectors)
    #     self.boundary_normals_set = True
        
    #     if abs(1.0-abs(np.dot(self.boundary_normal_cartesian_black,self.axis_cartesian)/np.linalg.norm(self.axis_cartesian))) < tol:
    #         self.boundary_type = 'twist'
    #     elif abs(np.dot(self.boundary_normal_cartesian_black,self.axis_cartesian)/np.linalg.norm(self.axis_cartesian)) < tol:
    #         self.boundary_type = 'tilt'
    #         # Check if tilt boundary is symmetric
    #         mean_boundary_normal = 0.5*(self.boundary_normal_cartesian_black + self.boundary_normal_cartesian_white)
    #         mean_boundary_normal = mean_boundary_normal/np.linalg.norm(mean_boundary_normal)
    #         sym_tilt = False
    #         for j in range(np.shape(self.lattice.symmetries.mirror_planes)[0]):
    #             if abs(1.0 - abs(np.dot(mean_boundary_normal,self.lattice.symmetries.mirror_planes[j,:])/np.linalg.norm(self.lattice.symmetries.mirror_planes[j,:]))) < tol:
    #                 sym_tilt = True
    #         if sym_tilt:
    #             self.boundary_type = 'symmetric'
    #     else:
    #         self.boundary_type = 'mixed'
        
        
    #     if self.debug:

    #         print("Boundary normals (lattice basis)")
    #         print('Black                                       White')
    #         #print('[ %6.0d %6.0d %6.0d ]   [ %6.0d %6.0d %6.0d ]' % (
    #         print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
    #             self.boundary_normal_black[0],self.boundary_normal_black[1],self.boundary_normal_black[2],
    #             self.boundary_normal_white[0],self.boundary_normal_white[1],self.boundary_normal_white[2])
    #             )
    #         print("Boundary normals (cartesian basis)")
    #         print('Black                                       White')
    #         print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
    #             self.boundary_normal_cartesian_black[0],self.boundary_normal_cartesian_black[1],self.boundary_normal_cartesian_black[2],
    #             self.boundary_normal_cartesian_white[0],self.boundary_normal_cartesian_white[1],self.boundary_normal_cartesian_white[2])
    #             )
    #         print("")
    #         print("Grain boundary type: " + self.boundary_type)
    #         print("")
        
    # Visualisation functionality
    try:
        def visualise_3d(self):
            """Generate a plot showing the Grain boundary geometry"""
            data = []
            colors = ['rgb(155, 0, 0)', 'rgb(0, 155, 0)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']
        
            x,y,z = ct.vis_data_box_ppp(self.csl.csl_vectors_cartesian_black, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell black',
                line = dict(width = 2, color = colors[0])
            )
            data.append(trace)
        
            x,y,z = ct.vis_data_box_ppp(self.boundary_cell_cartesian_black, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Boundary cell black',
                line = dict(width = 2, color = colors[2])
            )
            data.append(trace)
        
            x,y,z = ct.vis_data_box_ppp(self.csl.csl_vectors_cartesian_white, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell white',
                line = dict(width = 2, color = colors[1])
            )
            data.append(trace)
        
            x,y,z = ct.vis_data_box_ppp(self.boundary_cell_cartesian_white, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Boundary cell white',
                line = dict(width = 2, color = colors[3])
            )
            data.append(trace)
        
            scale = 1.0*np.trace(self.boundary_cell_cartesian_black)/np.linalg.norm(self.boundary_normal_cartesian_black)
            x,y,z = ct.vis_data_vectors(np.array([[0.0,0.0,0.0],[0.0,0.0,0.0]]), scale*np.array([self.boundary_normal_cartesian_black.tolist(),self.boundary_normal_cartesian_white.tolist()]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Boundary normals',
                line = dict(width = 2, color = 'blue')
            )
            data.append(trace)
        
            layout = go.Layout(
                width = 800, height = 500,
                title = "Grain boundary cells",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )
        
            print("----------------------------------------------------")
            print("Grain boundary visualisation:")
            print("*****************************")
            print()

            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })
        
        def visualise_3d_rotated(self):
            """Generate a plot showing the Grain boundary geometry"""
            data = []
            colors = ['rgb(155, 0, 0)', 'rgb(0, 155, 0)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']
        
            csl_vectors_rotated_black = np.dot(np.linalg.inv(self.csl.misorientation_rotation),self.csl.csl_vectors_cartesian_black.T).T
            x,y,z = ct.vis_data_box_ppp(csl_vectors_rotated_black, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell black',
                line = dict(width = 2, color = colors[0])
            )
            data.append(trace)
        
            boundary_cell_rotated_black = np.dot(np.linalg.inv(self.csl.misorientation_rotation),self.boundary_cell_cartesian_black.T).T
            x,y,z = ct.vis_data_box_ppp(boundary_cell_rotated_black, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Boundary cell black',
                line = dict(width = 2, color = colors[2])
            )
            data.append(trace)
        
            csl_vectors_rotated_white = np.dot(self.csl.misorientation_rotation,self.csl.csl_vectors_cartesian_white.T).T
            x,y,z = ct.vis_data_box_ppp(csl_vectors_rotated_white, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell white',
                line = dict(width = 4, color = colors[1], dash='dash')
            )
            data.append(trace)
        
            boundary_cell_rotated_white = np.dot(self.csl.misorientation_rotation,self.boundary_cell_cartesian_white.T).T
            x,y,z = ct.vis_data_box_ppp(boundary_cell_rotated_white, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Boundary cell black',
                line = dict(width = 4, color = colors[3], dash='dash')
            )
            data.append(trace)
        
            scale = 1.0*np.trace(boundary_cell_rotated_black)/np.linalg.norm(self.boundary_normal_cartesian_black)
            x,y,z = ct.vis_data_vectors(np.array([[0.0,0.0,0.0],[0.0,0.0,0.0]]), scale*np.array([
                np.dot(np.linalg.inv(self.csl.misorientation_rotation),self.boundary_normal_cartesian_black.T).T.tolist(),
                np.dot(self.csl.misorientation_rotation,self.boundary_normal_cartesian_white.T).T.tolist()
            ]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Boundary normals',
                line = dict(width = 2, color = 'black')
            )
            data.append(trace)
        
            layout = go.Layout(
                width = 800, height = 500,
                title = "Grain boundary cells",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )
        
            print("----------------------------------------------------")
            print("Grain boundary visualisation:")
            print("*****************************")
            print()

            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })
    except NameError:
        pass
            

# Cataloguing functionality

def write_gb_catalogue(folder, lattice_type, axis, m, n, limit, ext='df', atom_limit=None, csl_limit=5, orthogonal=False, search_radius=10):
    """Write out a catalogue of possible grain boundary planes for a given axis-angle combination 

    :param folder: Location of folder to write catalogue to
    :type folder: string
    :param lattice_type: Lattice type. Can be 'sc', 'bcc', 'fcc'
    :type lattice_type: string
    :param axis: Misorientation axis to consider
    :type axis: [int,int,int] or ndarray(3, dtype=int)
    :param m: First parameter in pair (m,n) used to specify a misorientation angle. Matches values output in a CSL catalogue
    :type m: int
    :param n: Second parameter in pair (m,n) used to specify a misorientation angle. Matches values output in a CSL catalogue
    :type n: int
    :param limit: Limit on values of indices specifying grain boundary plane
    :type limit: int
    :param ext: Extension for file output (controls file type), defaults to 'df'. Can be 'df', 'csv', 'xlsx'
    :type ext: str, optional
    :param atom_limit: Maximum allowable nummber of atoms in grain boundary cell, defaults to None
    :type atom_limit: int, optional
    :param csl_limit: Search limit applied in finding CSL cell, defaults to 5. Increasing this will slow things down but increase success rate of search
    :type csl_limit: int, optional
    :param orthogonal: Flag determines if vectors defining GB cell forced to be mutually orthogonal, defaults to False
    :type orthogonal: bool, optional
    :param search_radius: Search limit used in setting grain boundary plane, defaults to 10
    :type search_radius: int, optional
    :raises RuntimeError: "Cataloguing functionality not available for " + lattice_type + " lattice type", if unsupported lattice type specified
    :raises RuntimeError: "Unsupported file type: " + ext + ". Extention must be one of: df, csv, xlsx"
    :return: Grain boundary catalogue
    :rtype: Pandas dataframe
    """    
    
    if lattice_type not in ('sc', 'bcc', 'fcc'):
        raise RuntimeError("Cataloguing functionality not available for " + lattice_type + " lattice type")
        
    if ext not in ('df', 'csv', 'xlsx'):
            raise RuntimeError("Unsupported file type: " + ext + ". Extention must be one of: df, csv, xlsx")
    
    h = axis[0]
    k = axis[1]
    l = axis[2]
    
    filename = folder + 'gb_cat_' + lattice_type + '_' + str(h) + '_' + str(k) + '_' + str(l) + '__' + str(m)  + '_' + str(n)  + '_limit_' + str(limit) + '.' + ext
    
    theta = csl.calculate_theta(h,k,l,m,n)
    sigma = csl.calculate_sigma(h,k,l,m,n)
    test_gb = GrainBoundary(lattice_type)
    test_gb.set_axis([h,k,l])
    #test_gb.set_angle(theta)
    test_gb.set_angle_mn(m,n)
    #test_gb.set_debug()
    test_gb.set_csl(search_limit=csl_limit)
    #test_gb.unset_debug()
    
    # if include_basis:
    #     filename = folder + 'csl_cat_full_' + lattice_type + '_' + str(h) + '_' + str(k) + '_' + str(l) + '_limit_' + str(limit) + '.txt'
    # else:
    #     filename = folder + 'csl_cat_' + lattice_type + '_' + str(h) + '_' + str(k) + '_' + str(l) + '_limit_' + str(limit) + '.txt'
    
    gbcattemp = np.zeros(((2*limit+1)**3,4))
    gbnormalstemp = np.zeros(((2*limit+1)**3,2,3))
    gbtypestemp = []
    
    # Explore possible boundaries
    s = 0
    for H in range(limit,-(limit+1), -1):
        for K in range(limit,-(limit+1), -1):
            for L in range(limit,-(limit+1), -1):
                if (H != 0 or K != 0 or L != 0):
                    use = True
                    if ct.check_common_factors([H,K,L]):
                        use = False
                    if use:
                        # if H==0 and K==1 and L==0:
                        #     test_gb.set_debug()
                        # else:
                        #     test_gb.unset_debug()
                        gb_plane_set = test_gb.set_boundary_plane_csl([H,K,L], orthogonal=orthogonal, search_radius=search_radius, suppress_error=True)
                        if gb_plane_set:
                            #test_gb.calculate_boundary_normals() Normals now set in set_boundary_plane_csl()
                            gbcattemp[s,0:3]=[H,K,L]
                            gbcattemp[s,3] = test_gb.cell_volume_atoms
                            gbnormalstemp[s,0,:] = test_gb.boundary_normal_black[:]
                            gbnormalstemp[s,1,:] = test_gb.boundary_normal_white[:]
                            gbtypestemp.append(test_gb.boundary_type)
                            #print('***', gbcattemp[s,0:3], gbnormalstemp[s,0,:], gbnormalstemp[s,1,:], test_gb.boundary_type)
                            s = s + 1
                        else:
                            print('[{0},{1},{2}] '.format(H,K,L), end="", flush=True)
    num_boundaries = s
    gbtypestemp = np.array(gbtypestemp)
    idx = np.argsort(gbcattemp[:num_boundaries,3],0)  # Sort list by size of GB cell
    
    # write catalogue
    catalogue_df = pd.DataFrame({ 
        'h' : np.full(num_boundaries,h),
        'k' : np.full(num_boundaries,k),
        'l' : np.full(num_boundaries,l),
        'm' : np.full(num_boundaries,m),
        'n' : np.full(num_boundaries,n),
        'theta' : np.full(num_boundaries,theta*180/np.pi),
        'sigma' : np.full(num_boundaries,sigma),
        'H' : gbcattemp[idx[:],0],
        'K' : gbcattemp[idx[:],1],
        'L' : gbcattemp[idx[:],2],
        'num_atoms' : gbcattemp[idx[:],3]
    })
    catalogue_df.loc[:,'type'] = pd.Series(gbtypestemp[idx[:]], index=catalogue_df.index)
    for t in range(3):
        column_name = 'N_b' + '[' + str(t) + ']'
        catalogue_df.loc[:,column_name] = pd.Series(gbnormalstemp[idx[:],0,t], index=catalogue_df.index)
    for t in range(3):
        column_name = 'N_w' + '[' + str(t) + ']'
        catalogue_df.loc[:,column_name] = pd.Series(gbnormalstemp[idx[:],1,t], index=catalogue_df.index)
    
    if (atom_limit is not None):
        new_df = pd.concat([pd.DataFrame(),catalogue_df[catalogue_df['num_atoms'] <= atom_limit]], ignore_index=True)
        if ext == 'df':
            new_df.to_hdf(filename,'df')
        elif ext == 'csv':
            new_df.to_csv(filename)
        elif ext == 'xlsx':
            new_df.to_excel('filename', sheet_name='Sheet1')
        return new_df
    else:
        if ext == 'df':
            catalogue_df.to_hdf(filename,'df')
        elif ext == 'csv':
            catalogue_df.to_csv(filename)
        elif ext == 'xlsx':
            catalogue_df.to_excel('filename', sheet_name='Sheet1')
        return catalogue_df 
        
def dedupe_catalogue(gb_df, lattice_type):
    """Remove duplicate grain boundaries from a catalogue dataframe, based on symmetric equivalency i underlying crystal lattice

    :param gb_df: Grain boundry catalogue dataframe to dedupe
    :type gb_df: Pandas dataframe
    :param lattice_type: Lattice type. Can be 'sc', 'bcc', 'fcc'
    :type lattice_type: string
    :raises RuntimeError: "Cataloguing functionality not available for " + lattice_type + " lattice type", if unsupported lattice type specified
    :return: Dataframe without duplicates
    :rtype: Pandas dataframe
    """    
    gb_df_sorted = gb_df.copy()
    #pd.Categorical(gb_df_sorted['type'], ['symmetric', 'twist', 'tilt', 'mixed'])
    gb_df_sorted['type'] = pd.Categorical(gb_df_sorted['type'], ['symmetric', 'twist', 'tilt', 'mixed'])
    gb_df_sorted = gb_df_sorted.sort_values(['sigma', 'num_atoms', 'type', 'theta'], ascending=[True, True, True, True])
    sLength = len(gb_df_sorted['sigma'])
    gb_df_sorted = gb_df_sorted.assign(block=pd.Series(np.zeros(sLength, dtype=int)).values)
    num_rows = gb_df_sorted.shape[0]
    sym = symmetry.Symmetry(lattice_type)
    if not sym.symops_set:
        raise RuntimeError("Cataloguing functionality not available for " + lattice_type + " lattice type")
    rows_to_drop = []
    counter = 0
    equiv_index = 1
    while counter < num_rows: 
        block_counter = 0
        #num_atoms = gb_df_sorted.iloc[counter]['num_atoms']
        sigma = gb_df_sorted.iloc[counter]['sigma']
        #gb_type = gb_df_sorted.iloc[counter]['type']
        #sim_df = gb_df_sorted.loc[gb_df_sorted['type']==gb_type].loc[gb_df_sorted['num_atoms']==num_atoms]
        #sim_df = gb_df_sorted.loc[gb_df_sorted['type']==gb_type].loc[gb_df_sorted['sigma']==sigma]
        block_df = gb_df_sorted.loc[gb_df_sorted['sigma']==sigma]
        num_in_block = block_df.shape[0]
        #print('**', counter, num_atoms, gb_type, num_sim)
        while block_counter < num_in_block:
            ref_row = counter + block_counter
            if gb_df_sorted.iloc[ref_row, gb_df_sorted.columns.get_loc('block')] == 0:
                gb_df_sorted.iloc[ref_row, gb_df_sorted.columns.get_loc('block')] = equiv_index
                N_b_1 = np.array(gb_df_sorted.iloc[ref_row][['N_b[0]','N_b[1]','N_b[2]']])
                N_w_1 = np.array(gb_df_sorted.iloc[ref_row][['N_w[0]','N_w[1]','N_w[2]']])
                #print(ref_row,N_b_1, N_w_1)
                for i in range(1,num_in_block - block_counter):
                    this_row = ref_row + i
                    if gb_df_sorted.iloc[this_row, gb_df_sorted.columns.get_loc('block')] == 0:
                        N_b_2 = np.array(gb_df_sorted.iloc[this_row][['N_b[0]','N_b[1]','N_b[2]']])
                        N_w_2 = np.array(gb_df_sorted.iloc[this_row][['N_w[0]','N_w[1]','N_w[2]']])
                        if sym.vector_equiv(N_b_1,N_b_2) and sym.vector_equiv(N_w_1,N_w_2):
                            dupe = True
                            rows_to_drop.append(this_row)
                            gb_df_sorted.iloc[this_row, gb_df_sorted.columns.get_loc('block')] = equiv_index
                        elif sym.vector_equiv(N_b_1,N_w_2) and sym.vector_equiv(N_w_1,N_b_2):
                            rows_to_drop.append(this_row)
                            gb_df_sorted.iloc[this_row, gb_df_sorted.columns.get_loc('block')] = equiv_index
                            dupe = True
                        else:
                            dupe = False
                equiv_index = equiv_index + 1
            #print('  ',this_row, dupe, N_b_2, N_w_2)
            #print()
            block_counter = block_counter + 1
            #print(rows_to_drop)
        counter = counter + num_in_block
    
    return gb_df_sorted.drop(gb_df_sorted.index[rows_to_drop]), gb_df_sorted
