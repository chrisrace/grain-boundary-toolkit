# csl.py
""" Module to handle definitions of coincidence site lattices
Author:  Chris Race
Date:    3rd January 2017
Contact: christopher.race@manchester.ac.uk
"""
import numpy as np
import math
import pandas as pd
import warnings
from fractions import Fraction
from . import lattice
from . import spatialsearch
from . import crystaltools as ct

# For visualisation only
try:
    import plotly
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)
except ModuleNotFoundError:
    pass
    

CSL_TOL = 1e-3
TIGHT_TOL = 1e-6

class CSL(object):
    """A csl holds details of the misorientation axis-angle combination for a grain boundary.
    It also contains the csl basis vectors.

    :ivar object lattice: Lattice object containg description of underlying lattice
    :ivar ndarray(3,dtype=float) axis: Misorientation axis in multiples of lattice cell vectors
    :ivar ndarray(3,dtype=float) axis_cartesian: Misorientation axis in cartesian space
    :ivar float angle: misorientation angle in radians
    :ivar ndarray(2,dtype=integer) angle_indices: Values of m,n specifying misorentation angle
    :ivar ndarray((3,3),dtype=float) misorientation_rotation: Rotation matrix of 0.5*angle about misorientation axis
    :ivar object search: spatialsearch object to faciliate search for CSL basis vectors
    :ivar ndarray((3,3),dtype=float) csl_vectors_black: Vectors of CSL basis (in rows) in black half of bicrystal in multiples of lattice vectors
    :ivar ndarray((3,3),dtype=float) csl_vectors_white: Vectors of CSL basis (in rows) in white half of bicrystal in multiples of lattice vectors
    :ivar ndarray((3,3),dtype=float) csl_vectors_cartesian_black: Vectors of CSL basis (in rows) in black half of bicrystal in cartesian space
    :ivar ndarray((3,3),dtype=float) csl_vectors_cartesian_white: Vectors of CSL basis (in rows) in black half of bicrystal in cartesian space
    :ivar ndarray((3,3),dtype=float) dsc_vectors_cartesian_black:
    :ivar ndarray((3,3),dtype=float) dsc_vectors_cartesian_white:
    :ivar ndarray(3,dtype=float) dsc_vector_fractions:
    :ivar float cell_volume: Volume of CSL cell in cartesian space units
    :ivar float cell_volume_lattice: Volume of CSL cell in units of lattice unit cells
    :ivar integer cell_volume_atoms: Number of atoms in CSL cell
    :ivar boolean debug: Flag set to True if debug information requested
    :ivar boolean search_enabled: Flag to indicate if spatialsearch object is initialised
    :ivar boolean angle_set: Flag to indicate if misorientation angle is set
    :ivar boolean axis_set: Flag to indicate if misorientation axis is set
    :ivar boolean csl_basis_set: Flag to indicate if csl_basis has been calculated (and csl_vectors_black, etc. populated)
    :ivar boolean misorientation_rotation_set: Flag to indicate if misorientation_rotation matrix is populated
    """    

    
    def __init__(self, latticetype, lengths=None, angles=None):
        """Initialise an empty csl with the specified lattice type

        :param latticetype: Lattice type, passed to Lattice module
        :type latticetype: string
        :param lengths: Lengths of unit cell vectors, passed to Lattice module, defaults to None
        :type lengths: list of floats, optional
        :param angles: Angles between unit cell vectors, passed to Lattice module, defaults to None
        :type angles: list of floats, optional
        """        
        
        self.lattice = lattice.Lattice(latticetype, lengths, angles)
        self.axis = np.zeros((3), dtype=float)
        self.axis_cartesian = np.zeros((3), dtype=float)
        self.angle = 0.0
        self.csl_vectors_black = np.zeros((3,3), dtype=int)
        self.csl_vectors_white = np.zeros((3,3), dtype=int)
        self.csl_vectors_cartesian_black = np.zeros((3,3), dtype=float)
        self.csl_vectors_cartesian_white = np.zeros((3,3), dtype=float)
        self.dsc_vectors_fractions = np.zeros(3, dtype=float)
        self.dsc_vectors_cartesian_black = np.zeros((3,3), dtype=float)
        self.dsc_vectors_cartesian_white = np.zeros((3,3), dtype=float)
        
        # Control flags
        self.debug = False
        self.search_enabled = False
        self.angle_set = False
        self.axis_set = False
        self.csl_basis_set = False
        self.dsc_basis_set = False
        self.misorientation_rotation_set = False

    def set_debug(self):
        """Turn on debug info
        """        
        self.debug = True
        
    def unset_debug(self):
        """Turn off debug info
        """
        self.debug = False
    
    def enable_search(self, maxradius=5):
        """Populate an array of search vectors using spatialsearch module

        :param maxradius: Maximum searhc radius, defaults to 5
        :type maxradius: int, optional
        """        
        if self.lattice.lattice_type in ('fcc','sc','bcc'):
            self.search = spatialsearch.SpatialSearch(maxradius)  # Slight shortcut for cubic bases
        else:
            self.search = spatialsearch.SpatialSearch(maxradius,self.lattice.cell_vectors)
        self.search_enabled = True
        
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
        self.angle_indices = np.array([0.0,0.0])
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
        if lattice_type not in ['fcc', 'bcc', 'sc']:
            raise RuntimeError("Lattice type " + self.lattice.lattice_type + " not supported.")    
        self.angle = calculate_theta(self.axis[0],self.axis[1],self.axis[2],m,n)
        self.angle_indices = np.array([m,n])
        self.angle_set = True
        
    def find_misorientation_rotation_matrix(self):
        """Find rotation matrix to rotate each crystal about misorientation axis (by half misorientation angle)

        :raises RuntimeError: "Misorientation axis must be set before calculating misorientation rotation" if misorientation axis not already set
        :raises RuntimeError: "Misorientation angle must be set before calculating misorientation rotation" if misorientation angle not already set
        """        
        if (not self.axis_set):
            raise RuntimeError("Misorientation axis must be set before calculating misorientation rotation")
        elif (not self.angle_set):
            raise RuntimeError("Misorientation angle must be set before calculating misorientation rotation")
        self.misorientation_rotation = ct.rotation_matrix(self.axis_cartesian,0.5*self.angle)
        self.misorientation_rotation_set = True
        return
        
    def find_csl_basis(self, tol=CSL_TOL):
        """Find a set of basis vectors suitable for defining the csl by finding 3 non-colinear points in CSL lattice
        i.e. lattice points commont to black and white lattices 
        These vectors are recorded both in the original crystal lattice and in the cartesian basis in the original orientation

        :param tol: Tolerance used to define when two vectors are equivalent (max deviation of normalised dot product from unity), defaults to CSL_TOL
        :type tol: float, optional
        :raises RuntimeError: "Misorientation axis and angle must be set before finding csl basis" if either misorientation axis or angle not yet set
        :raises RuntimeError: "Misorientation rotation matrix must be calculated before finding csl basis - callfind_misorientation_rotation_matrix()" if rotation matrix for misorientation not yet calculated
        :raises RuntimeError: "Search not enabled, please call enable_search() to find csl basis" is a spatialsearch object has not been instantiated
        :return: Flag for success in finding basis
        :rtype: boolean
        """  

        if (not self.axis_set) or (not self.angle_set):
            raise RuntimeError("Misorientation axis and angle must be set before finding csl basis")
        if (not self.misorientation_rotation_set):
            raise RuntimeError("Misorientation rotation matrix must be calculated before finding csl basis - callfind_misorientation_rotation_matrix()")
        if (not self.search_enabled):
            raise RuntimeError("Search not enabled, please call enable_search() to find csl basis")

        if self.debug: 
            print("----------------------------------------------------")
            print("csl.find_csl_basis() debug:")
            print("***************************")
        
        # Calculate rotation matrix for misorientation in which black crystal remains in original orientation
        # This is used to check for pairs of vectors that rotate one into the other under the GB misorientation     
        full_misorientation_rotation = np.dot(self.misorientation_rotation,self.misorientation_rotation)

        # Find pairs of lattice points, equidistant from origin and related by the misorientation angle           
        pairs_black = np.zeros((3,3),dtype=int)
        pairs_white = np.zeros((3,3),dtype=int)
        pairs_cartesian_black = np.zeros((3,3),dtype=float)
        pairs_cartesian_white = np.zeros((3,3),dtype=float)
        
        basis_found = False
        pairs_needed = 3
        pairs_found = 0
        black_index = 0
        # Keep looking for non-colinear CSL lattice points until either 3 are found or searchspace is exhausted
        while (pairs_found < pairs_needed) and (black_index < self.search.nspace):
            # Consider a candidate point in the black lattice
            r_black = np.einsum('i,ij', self.search.space[black_index,0:3], self.lattice.cell_vectors) # Cartesian coordinate of candidate point in black lattice
            l_black = self.search.space[black_index,3] # Distance from origin of candidate point in lattice units
            noncolinear = True
            noncoplanar = True
            # Check if newly found candidate point in black lattice is colinear with previously recorded CSL points
            for i in range(pairs_found):
                if (1.0-abs(np.dot(pairs_cartesian_black[i,:],r_black[:])/np.linalg.norm(pairs_cartesian_black[i,:])/np.linalg.norm(r_black[:])) < tol):
                    noncolinear = False
            # Check if newly found candidate point in black lattice is coplanar with pair of previously recorded CSL points
            if pairs_found == 2:
                if abs(np.dot(np.cross(pairs_cartesian_black[0,:],pairs_cartesian_black[1,:]),r_black[:])/np.linalg.norm(pairs_cartesian_black[0,:])/np.linalg.norm(pairs_cartesian_black[1,:])/np.linalg.norm(r_black[:])) < tol:
                    noncoplanar = False
            # if newly found candidate point passes tests for nonc-colinearity and non-coplanarity proceed to find a corresponding point in white lattice
            if noncolinear and noncoplanar:
                this_point_paired = False
                # Find range to search white lattice vectors
                stop_search = False
                white_index = black_index
                # Two while loops find bounds on the index for searchspace that constrais search for white lattice point to points same distance from origin as candidate black point
                while not stop_search:
                    if self.search.space[white_index,3]-l_black < -tol  or white_index == 0:
                        stop_search = True
                        min_white_index = white_index
                    white_index = white_index - 1
                stop_search = False
                white_index = black_index
                while not stop_search:
                    if self.search.space[white_index,3]-l_black > tol or white_index >= self.search.nspace-1:
                        stop_search = True
                        max_white_index = white_index
                    white_index = white_index + 1
                #print(black_index,min_white_index,max_white_index)
                # Now search white lattice points between ranges found above (i.e. points at correct idstance from origin)
                white_index = min_white_index    
                while ((not this_point_paired) and (white_index <= max_white_index)):
                    # Rotate candidate white point by misorientation angle and then check for coincidence with black point
                    r_white = np.dot(full_misorientation_rotation, np.einsum('i,ij', self.search.space[white_index,0:3], self.lattice.cell_vectors))
                    if np.linalg.norm(r_black - r_white) < tol:
                        pairs_black[pairs_found,:] = self.search.space[black_index,0:3]
                        pairs_cartesian_black[pairs_found,:] = r_black
                        pairs_white[pairs_found,:] = self.search.space[white_index,0:3]
                        pairs_cartesian_white[pairs_found,:] = np.einsum('i,ij', self.search.space[white_index,0:3], self.lattice.cell_vectors)
                        this_point_paired = True
                        pairs_found = pairs_found + 1
                    white_index = white_index + 1
            black_index = black_index + 1
        
        if (pairs_found != pairs_needed):
            self.csl_basis_set = False
            warnings.warn("Failed to find three vectors for basis within pair list - consider a wider search")
            #raise RuntimeError("Failed to find three vectors for basis within pair list - consider a wider search")
        self.csl_basis_set = True
        
        self.csl_vectors_black = np.copy(pairs_black)
        self.csl_vectors_white = np.copy(pairs_white)
        self.csl_vectors_cartesian_black = np.copy(pairs_cartesian_black)
        self.csl_vectors_cartesian_white = np.copy(pairs_cartesian_white)
        
        self.cell_volume = abs(np.dot(self.csl_vectors_cartesian_black[0],np.cross(self.csl_vectors_cartesian_black[1],self.csl_vectors_cartesian_black[2])))
        self.cell_volume_lattice = self.cell_volume / abs(np.dot(self.lattice.cell_vectors[0],np.cross(self.lattice.cell_vectors[1],self.lattice.cell_vectors[2])))
        self.cell_volume_atoms = self.cell_volume_lattice * self.lattice.num_basis
        # Write out details of CSL lattice vectors if debug flag set
        if self.debug:
            print("CSL Cell vectors (lattice basis)")
            print('Black                      White')
            for s in range(3):
                print('[ %6.0d %6.0d %6.0d ]   [ %6.0d %6.0d %6.0d ]' % (
                    self.csl_vectors_black[s,0],self.csl_vectors_black[s,1],self.csl_vectors_black[s,2],
                    self.csl_vectors_white[s,0],self.csl_vectors_white[s,1],self.csl_vectors_white[s,2])
                    )
            print("CSL Cell vectors (cartesian basis)")
            print('Black                                       White')
            for s in range(3):
                print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
                    self.csl_vectors_cartesian_black[s,0],self.csl_vectors_cartesian_black[s,1],self.csl_vectors_cartesian_black[s,2],
                    self.csl_vectors_cartesian_white[s,0],self.csl_vectors_cartesian_white[s,1],self.csl_vectors_cartesian_white[s,2])
                    )
            print("")
        
        return pairs_found == pairs_needed
        
    def find_dsc_lattice(self, tol=CSL_TOL):
        """Finds the DSC (displacement shift complete) lattice vectors for the CSL and populates instance variables to store the details

        :param tol: Tolerance used to define when two vectors are equivalent (max deviation of normalised dot product from unity), defaults to CSL_TOL
        :type tol: float, optional
        :raises RuntimeError: "CSL basis must be found before calculating DSC lattice" if attemtping to find DSC lattice before CSL basis calculated
        """        
        if not self.csl_basis_set:
            raise RuntimeError("CSL basis must be found before calculating DSC lattice")
            
        if self.debug: 
            print("----------------------------------------------------")
            print("csl.find_dsc_lattice() debug:")
            print("*****************************")
            
        csl_vectors_rotated_black = np.dot(np.linalg.inv(self.misorientation_rotation),self.csl_vectors_cartesian_black.T).T
        repeats = ct.get_repeats(self.csl_vectors_cartesian_black, self.lattice.cell_vectors)
        n_black,r = ct.fill_box(self.csl_vectors_cartesian_black, self.lattice.cell_vectors, self.lattice.basis_coords[self.lattice.atom_types[:]==1,:], repeats)
        r_b = np.dot(np.linalg.inv(self.misorientation_rotation),np.array(r).T).T
        
        csl_vectors_rotated_white = np.dot(self.misorientation_rotation,self.csl_vectors_cartesian_white.T).T
        repeats = ct.get_repeats(self.csl_vectors_cartesian_white, self.lattice.cell_vectors)
        n_white,r = ct.fill_box(self.csl_vectors_cartesian_white, self.lattice.cell_vectors, self.lattice.basis_coords[self.lattice.atom_types[:]==1,:], repeats)
        r_w = np.dot(self.misorientation_rotation,np.array(r).T).T
        frac_dr = np.zeros(3,dtype=float)
        min_frac_dr = np.full(3, 1.0)
        for i in range(n_black):
            for j in range(n_white):
                dr = r_b[i,:] - r_w[j,:]
                frac_dr = abs(ct.components_in_basis(dr,csl_vectors_rotated_black))
                for s in range(3):
                    if frac_dr[s] < min_frac_dr[s] and frac_dr[s] > tol:
                        min_frac_dr[s] = frac_dr[s]
        self.dsc_vector_fractions = min_frac_dr
        self.dsc_vectors_cartesian_black = np.einsum('i,ij->ij',min_frac_dr,self.csl_vectors_cartesian_black)
        self.dsc_vectors_cartesian_white = np.einsum('i,ij->ij',min_frac_dr,self.csl_vectors_cartesian_white)
        
        if self.debug:
            print("DSC Cell fractions (and reciprocals)")
            print('[ %12.6f (%12.6f) %12.6f (%12.6f) %12.6f (%12.6f) ]' % (
                self.dsc_vector_fractions[0],1.0/self.dsc_vector_fractions[0],
                self.dsc_vector_fractions[1],1.0/self.dsc_vector_fractions[1],
                self.dsc_vector_fractions[2],1.0/self.dsc_vector_fractions[2]
                ))
            print("DSC Cell vectors (cartesian basis)")
            print('Black                                       White')
            for s in range(3):
                print('[ %12.6f %12.6f %12.6f]   [ %12.6f %12.6f %12.6f ]' % (
                    self.dsc_vectors_cartesian_black[s,0],self.dsc_vectors_cartesian_black[s,1],self.dsc_vectors_cartesian_black[s,2],
                    self.dsc_vectors_cartesian_white[s,0],self.dsc_vectors_cartesian_white[s,1],self.dsc_vectors_cartesian_white[s,2])
                    )
            print("")
        
        self.dsc_basis_set = True


    # Visualisation functionality
    
    try:
        def visualise_3d(self):
            """Generate a plot showing the CSL"""
            data = []
            colors = ['rgb(155, 0, 0)', 'rgb(0, 155, 0)', 'rgb(0, 0, 155)']
    
            x,y,z = ct.vis_data_box_ppp(self.csl_vectors_cartesian_black, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell black',
                line = dict(width = 2, color = colors[0])
            )
            data.append(trace)
    
            x,y,z = ct.vis_data_box_ppp(self.csl_vectors_cartesian_white, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell white',
                line = dict(width = 2, color = colors[1])
            )
            data.append(trace)
    
            repeats = ct.get_repeats(self.csl_vectors_cartesian_black, self.lattice.cell_vectors)
            n_black,r = ct.fill_box(self.csl_vectors_cartesian_black, self.lattice.cell_vectors, self.lattice.basis_coords, repeats)
            r = np.array(r)
            name = 'Atoms black'
            trace = go.Scatter3d(
                x = r[:,0], y = r[:,1], z = r[:,2], mode = 'markers', name = name,
                marker = dict(size = 5, color = colors[0])
            )
            data.append(trace)
    
            repeats = ct.get_repeats(self.csl_vectors_cartesian_white, self.lattice.cell_vectors)
            n_white,r = ct.fill_box(self.csl_vectors_cartesian_white, self.lattice.cell_vectors, self.lattice.basis_coords, repeats)
            r = np.array(r)
            name = 'Atoms white'
            trace = go.Scatter3d(
                x = r[:,0], y = r[:,1], z = r[:,2], mode = 'markers', name = name,
                marker = dict(size = 3, color = colors[1])
            )
            data.append(trace)
    
            layout = go.Layout(
                width = 800, height = 500,
                title = "CSL cells",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )
    
            print("----------------------------------------------------")
            print("CSL visualisation:")
            print("******************")
            print("CSL cells contain %6.d black atoms, %6.d white atoms" %(n_black, n_white))
            print()

            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })

        def visualise_3d_rotated(self):
            """Generate a plot showing the CSL"""
            data = []
            colors = ['rgb(155, 0, 0)', 'rgb(0, 155, 0)', 'rgb(0, 0, 155)']
    
            csl_vectors_rotated_black = np.dot(np.linalg.inv(self.misorientation_rotation),self.csl_vectors_cartesian_black.T).T
            x,y,z = ct.vis_data_box_ppp(csl_vectors_rotated_black, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell black',
                line = dict(width = 2, color = colors[0])
            )
            data.append(trace)
    
            csl_vectors_rotated_white = np.dot(self.misorientation_rotation,self.csl_vectors_cartesian_white.T).T
            x,y,z = ct.vis_data_box_ppp(csl_vectors_rotated_white, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'CSL cell white',
                line = dict(width = 4, color = colors[1], dash='dash')
            )
            data.append(trace)
    
            repeats = ct.get_repeats(self.csl_vectors_cartesian_black, self.lattice.cell_vectors)
            n_black,r = ct.fill_box(self.csl_vectors_cartesian_black, self.lattice.cell_vectors, self.lattice.basis_coords, repeats)
            r = np.dot(np.linalg.inv(self.misorientation_rotation),np.array(r).T).T
            name = 'Atoms black'
            trace = go.Scatter3d(
                x = r[:,0], y = r[:,1], z = r[:,2], mode = 'markers', name = name,
                marker = dict(size = 5, color = colors[0])
            )
            data.append(trace)
    
            repeats = ct.get_repeats(self.csl_vectors_cartesian_white, self.lattice.cell_vectors)
            n_white,r = ct.fill_box(self.csl_vectors_cartesian_white, self.lattice.cell_vectors, self.lattice.basis_coords, repeats)
            r = np.dot(self.misorientation_rotation,np.array(r).T).T
            name = 'Atoms white'
            trace = go.Scatter3d(
                x = r[:,0], y = r[:,1], z = r[:,2], mode = 'markers', name = name,
                marker = dict(size = 3, color = colors[1])
            )
            data.append(trace)
    
            layout = go.Layout(
                width = 800, height = 500,
                title = "CSL cells",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )
    
            print("----------------------------------------------------")
            print("CSL visualisation:")
            print("******************")
            print("CSL cells contain %6.d black atoms, %6.d white atoms" %(n_black, n_white))
            print()

            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })
    
        def visualise_2d_rotated(self, axis=-1):
            """Generate a plot showing a 2D projection of the CSL

            :param axis: Determines the vector down which the structure is projected (viewed). -1 selects misoreientation axis, [0,1,2] select one of the CSL vectors, defaults to -1
            :type axis: int, optional
            :raises RuntimeError: "Invalid choice for projection axis", if an invalid value for the axis is specified
            """
            data = []
            colors = ['rgb(155, 0, 0)', 'rgb(0, 155, 0)', 'rgb(0, 0, 155)', 'rgb(0, 0, 0)', 'rgb(155, 155, 155)']
    
            if axis == -1:
                projection_axis = self.axis
            elif axis in [0,1,2]:
                projection_axis = np.dot(np.linalg.inv(self.misorientation_rotation),self.csl_vectors_cartesian_black.T).T[axis,:]
            else:
                raise RuntimeError("Invalid choice for projection axis")
        
            axis_rotation = ct.rotation_matrix_into_direction(projection_axis,np.array([0.0,0.0,1.0]))
    
            csl_vectors_rotated_black = np.dot(np.dot(axis_rotation,np.linalg.inv(self.misorientation_rotation)),self.csl_vectors_cartesian_black.T).T
            x,y,z = ct.vis_data_box_ppp(csl_vectors_rotated_black, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter(
                x = x, y = y,  mode = 'lines', name = 'CSL cell black',
                line = dict(width = 2, color = colors[0])
            )
            data.append(trace)
    
            csl_vectors_rotated_white = np.dot(np.dot(axis_rotation,self.misorientation_rotation),self.csl_vectors_cartesian_white.T).T
            x,y,z = ct.vis_data_box_ppp(csl_vectors_rotated_white, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter(
                x = x, y = y,  mode = 'lines', name = 'CSL cell white',
                line = dict(width = 4, color = colors[1], dash='dash')
            )
            data.append(trace)
            if self.dsc_basis_set and np.min(self.dsc_vector_fractions) > 0.01:
                csl_vectors_rotated_black = np.dot(np.dot(axis_rotation,np.linalg.inv(self.misorientation_rotation)),self.csl_vectors_cartesian_black.T).T
                points = []
                repeats = np.round(1.0/self.dsc_vector_fractions)
                for s in range(3):
                    for i in range(int(repeats[s])):
                        for j in range(int(repeats[(s+1)%3])):
                            points.append((i*self.dsc_vector_fractions[s]*csl_vectors_rotated_black[s,:] + j*self.dsc_vector_fractions[(s+1)%3]*csl_vectors_rotated_black[(s+1)%3,:]).tolist())
                            points.append((i*self.dsc_vector_fractions[s]*csl_vectors_rotated_black[s,:] + j*self.dsc_vector_fractions[(s+1)%3]*csl_vectors_rotated_black[(s+1)%3,:] + csl_vectors_rotated_black[(s+2)%3,:]).tolist())
                            points.append([None,None,None]) 
                r = np.array(points)
                name = 'DSC Lattice'
                trace = go.Scatter(
                    x = r[:,0], y = r[:,1],  mode = 'lines', name = name,
                    line = dict(width = 0.5, color = colors[4])
                )
                data.append(trace)

            repeats = ct.get_repeats(self.csl_vectors_cartesian_black, self.lattice.cell_vectors)
            n_black,r = ct.fill_box(self.csl_vectors_cartesian_black, self.lattice.cell_vectors, self.lattice.basis_coords, repeats)
            r = np.dot(np.dot(axis_rotation,np.linalg.inv(self.misorientation_rotation)),np.array(r).T).T
            name = 'Atoms black'
            trace = go.Scatter(
                x = r[:,0], y = r[:,1], mode = 'markers', name = name,
                marker = dict(size = 8, color = colors[0])
            )
            data.append(trace)

            repeats = ct.get_repeats(self.csl_vectors_cartesian_white, self.lattice.cell_vectors)
            n_white,r = ct.fill_box(self.csl_vectors_cartesian_white, self.lattice.cell_vectors, self.lattice.basis_coords, repeats)
            r = np.dot(np.dot(axis_rotation,self.misorientation_rotation),np.array(r).T).T
            name = 'Atoms white'
            trace = go.Scatter(
                x = r[:,0], y = r[:,1], mode = 'markers', name = name,
                marker = dict(size = 5, color = colors[1])
            )
            data.append(trace)

    
            layout = go.Layout(
                width = 800, height = 500,
                title = "CSL cells",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )
    
            print("----------------------------------------------------")
            print("CSL visualisation:")
            print("******************")
            print("CSL cells contain %6.d black atoms, %6.d white atoms" %(n_black, n_white))
            print()

            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })
    except NameError:
        pass

# Cataloguing functionality

def write_csl_catalogue(folder, lattice_type, axis, limit, ext='df', include_basis=False, atom_limit=None):
    """Write out a catalogue of possible angles and sigma values for a given grain boundary axis. Currently only works for cubic lattices.

    :param folder: Location of folder to write catalogue to
    :type folder: string
    :param lattice_type: Lattice type
    :type lattice_type: string
    :param axis: Misorientation axis to consider
    :type axis: [int,int,int] or ndarray(3, dtype=int)
    :param limit: Limit to use foir search in the form of maximum valed for m and n
    :type limit: int
    :param ext: File extension, defaults to 'df'
    :type ext: str, optional
    :param include_basis: Flag to control whether CSL basis included in output (significantly increases runtime), defaults to False
    :type include_basis: bool, optional
    :param atom_limit: Write out only those CSLs with fewer atoms than this, defaults to None
    :type atom_limit: int, optional
    :raises RuntimeError: "Cataloguing functionality not available for " + lattice_type + " lattice type", if cataloguing for an unsupported lattice is requested
    :raises RuntimeError: "Unsupported file type: " + ext + ". Extention must be one of: df, csv, xlsx", if an unsupported file type is specified
    :raises RuntimeError: "Atom limit can only be applied if include_basis=true", if atom_limit is set, but inclide_basis is not set to True
    :return: A catalogue of the possible CSL grain boundaries for the given axis within the limits set
    :rtype: Pandas dataframe
    """    
    
    if lattice_type not in ('sc', 'bcc', 'fcc'):
        raise RuntimeError("Cataloguing functionality not available for " + lattice_type + " lattice type")
    
    if ext not in ('df', 'csv', 'xlsx'):
            raise RuntimeError("Unsupported file type: " + ext + ". Extention must be one of: df, csv, xlsx")
    
    h = axis[0]
    k = axis[1]
    l = axis[2]
    
    if include_basis:
        filename = folder + 'csl_cat_full_' + lattice_type + '_' + str(h) + '_' + str(k) + '_' + str(l) + '_limit_' + str(limit) + '.' + ext
    else:
        filename = folder + 'csl_cat_' + lattice_type + '_' + str(h) + '_' + str(k) + '_' + str(l) + '_limit_' + str(limit) + '.' + ext
    
    cslcattemp = np.zeros(((2*limit) * (2*limit-1),6))
    # Explore possible boundaries
    s = 0
    for m in range(limit+1):
        for n in range(1,limit+1):
                use = True
                cf = 2
                while use and cf<=100:
                    if (m%cf==0 and n%cf==0):
                        use = False   
                    cf = cf + 1 
                this_sigma = calculate_sigma(h,k,l,m,n)
                if use and this_sigma == 1.0:
                    use = False
                if use:
                    if lattice_type in ('sc', 'bcc', 'fcc'):
                        rational_cosine = calculate_cosine(h,k,l,m,n)
                        if atom_limit is None:
                            cslcattemp[s,:]=[m, n, 180.0*calculate_theta(h,k,l,m,n)/np.pi, this_sigma, rational_cosine.numerator, rational_cosine.denominator]
                            s = s + 1
                        else:
                            if this_sigma <= atom_limit:
                                cslcattemp[s,:]=[m, n, 180.0*calculate_theta(h,k,l,m,n)/np.pi, this_sigma, rational_cosine.numerator, rational_cosine.denominator]
                                s = s + 1
    num_boundaries = s
    # Sort array by last column (length)
    idx = np.argsort(cslcattemp[:num_boundaries,2],0)
    #cslcattemp = cslcattemp[idx[:],:]
    sigmas_found = []
    temp_idx = []
    for i in range(num_boundaries):
        if cslcattemp[idx[i],3] not in sigmas_found:
            sigmas_found.append(cslcattemp[idx[i],3])
            temp_idx.append(idx[i])
    num_boundaries = len(temp_idx)
    idx = np.array(temp_idx)
    
    if num_boundaries > 0:
        catalogue_df = pd.DataFrame({ 
            'h' : np.full(num_boundaries,h),
            'k' : np.full(num_boundaries,k),
            'l' : np.full(num_boundaries,l),
            'm' : cslcattemp[idx[:],0],
            'n' : cslcattemp[idx[:],1],
            'theta' : cslcattemp[idx[:],2],
            'sigma' : cslcattemp[idx[:],3],
            'cos_num' : cslcattemp[idx[:],4],
            'cos_den' : cslcattemp[idx[:],5]
        })
    
        if include_basis:
            csl_bases = np.zeros((num_boundaries,6,3), dtype=float)
            csl_num_atoms = np.zeros((num_boundaries), dtype=float)
            for i in range(num_boundaries):
                test_csl = CSL(lattice_type)
                test_csl.set_axis([h,k,l])
                test_csl.enable_search(20)
                test_csl.set_angle(calculate_theta(h,k,l,int(cslcattemp[idx[i],0]),int(cslcattemp[idx[i],1])))
                test_csl.find_misorientation_rotation_matrix()
                test_csl.find_csl_basis()
                csl_bases[i,0:3,:] = test_csl.csl_vectors_black
                csl_bases[i,3:6,:] = test_csl.csl_vectors_white
                csl_num_atoms[i] = test_csl.cell_volume_atoms
            catalogue_df.loc[:,'num_atoms'] = pd.Series(csl_num_atoms[:], index=catalogue_df.index)
            for s in range(3):
                for t in range(3):
                    column_name = 'b_' + str(s) + '[' + str(t) + ']'
                    catalogue_df.loc[:,column_name] = pd.Series(csl_bases[:,s,t], index=catalogue_df.index)
            for s in range(3):
                for t in range(3):
                    column_name = 'w_' + str(s) + '[' + str(t) + ']'
                    catalogue_df.loc[:,column_name] = pd.Series(csl_bases[:,s+3,t], index=catalogue_df.index)
    
        if (atom_limit is not None):
            if not include_basis:
                raise RuntimeError("Atom limit can only be applied if include_basis=true")
            new_df = pd.concat([pd.DataFrame(),catalogue_df[catalogue_df['num_atoms'] <= atom_limit]], ignore_index=True)
            if ext == 'df':
                new_df.to_hdf(filename,'df')
            elif ext == 'csv':
                catalogue_df[catalogue_df['num_atoms'] <= atom_limit].to_csv(filename)
            elif ext == 'xlsx':
                catalogue_df[catalogue_df['num_atoms'] <= atom_limit].to_excel('filename', sheet_name='Sheet1')
            return new_df
        else:
            if ext == 'df':
                catalogue_df.to_hdf(filename,'df')
            elif ext == 'csv':
                catalogue_df.to_csv(filename)
            elif ext == 'xlsx':
                catalogue_df.to_excel('filename', sheet_name='Sheet1')
            return catalogue_df
    else:
        return
                
def calculate_sigma(h,k,l,m,n,lattice_type='fcc'):
    """Calculate the value of sigma for the grain boundary from the axis [h,k,l] and the integers m,n in a specified lattice type.
    Currently only implemented for cubic lattice types.

    :param h: First component of misorientation axis
    :type h: integer
    :param k: Second component of misorientation axis
    :type k: integer
    :param l: Third component of misorientation axis
    :type l: integer
    :param m: First parameter specifying misorientation angle
    :type m: integer
    :param n: Second parameter specifying misorientation angle
    :type n: integer
    :param lattice_type: lattice type, defaults to 'fcc'
    :type lattice_type: str, optional
    :raises RuntimeError: "Lattice type " + self.lattice.lattice_type + " not supported." if an unsupported lattice type is specified
    :return: Sigma value
    :rtype: integer
    """    
    
    if lattice_type in ['fcc', 'bcc', 'sc']:
        sigma = n*n+m*m*(h*h + k*k + l*l)
        while sigma%2==0:
            sigma = sigma/2
    else:
        raise RuntimeError("Lattice type " + self.lattice.lattice_type + " not supported.")
    return sigma
    
def calculate_theta(h,k,l,m,n,lattice_type='fcc'):
    """Calculate the misorientation angle for the grain boundary from the axis [h,k,l] and the integers m,n in a specified lattice type.
    Currently only implemented for cubic lattice types.

    :param h: First component of misorientation axis
    :type h: integer
    :param k: Second component of misorientation axis
    :type k: integer
    :param l: Third component of misorientation axis
    :type l: integer
    :param m: First parameter specifying misorientation angle
    :type m: integer
    :param n: Second parameter specifying misorientation angle
    :type n: integer
    :param lattice_type: _description_, defaults to 'fcc'
    :type lattice_type: str, optional
    :raises RuntimeError: "Lattice type " + self.lattice.lattice_type + " not supported." if an unsupported lattice type is specified
    :return: Misorientation angle in radians 
    :rtype: float
    """    
    msq = 1.0*m*m
    nsq = 1.0*n*n
    if lattice_type in ['fcc', 'bcc', 'sc']:
        tansqphi = (msq/nsq)*(h*h + k*k + l*l)
        theta = 2.0*np.arctan(np.sqrt(tansqphi))
    else:
        raise RuntimeError("Lattice type " + self.lattice.lattice_type + " not supported.")
    return theta

def calculate_cosine(h,k,l,m,n,lattice_type='fcc'):
    """Calculate the cosine of misorientation angle, expressed as the components of a fraction in a tuple 
    for the grain boundary from the axis [h,k,l] and the integers m,n in a specified lattice type
    Currently only implemented for cubic lattice types.

    :param h: First component of misorientation axis
    :type h: integer
    :param k: Second component of misorientation axis
    :type k: integer
    :param l: Third component of misorientation axis
    :type l: integer
    :param m: First parameter specifying misorientation angle
    :type m: integer
    :param n: Second parameter specifying misorientation angle
    :type n: integer
    :param lattice_type: _description_, defaults to 'fcc'
    :type lattice_type: str, optional
    :raises RuntimeError: "Lattice type " + self.lattice.lattice_type + " not supported." if an unsupported lattice type is specified
    :return: Cosine of the misorientation angle expressed as the components of a fraction in a tuple (numerator,denominator) 
    :rtype: (integer,integer)
    """    
    msq = 1.0*m*m
    nsq = 1.0*n*n
    if lattice_type in ['fcc', 'bcc', 'sc']:
        tansqphi = (msq/nsq)*(h*h + k*k + l*l)
        costheta = (1.0 - tansqphi)/(1.0 + tansqphi)
        costhetafrac = Fraction(costheta).limit_denominator()
    else:
        raise RuntimeError("Lattice type " + self.lattice.lattice_type + " not supported.")
    return costhetafrac
