# crystaltools.py
""" Module containing a number of helper functions for handling and visualising crystal structures

| Author:  Chris Race
| Date:    3rd January 2017
| Contact: christopher.race@manchester.ac.uk
"""

import numpy as np
import math
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
# from . import outcar

CRYSTALTOOLS_TOL = 1e-3

def check_common_factors(indices, limit=100):
    """Check the list of integers for a common factor up to the value of limit.

    :param indices: list of integers to check for common factors
    :type indices: list of integers
    :param limit: maximum value of common factor to try, defaults to 100
    :type limit: int, optional
    :return: True if a common factor <=limit exists
    :rtype: boolean
    """    
    common_factors = False
    cf = 2
    while (not common_factors) and (cf <= limit):
        if (np.prod(np.array(indices)%cf==0) == 1):
            common_factors = True
        cf = cf + 1
    return common_factors

def indices_in_basis(vector,basis):
    """Express the direction of the given N-dimensional vector in the given basis such that the components are integer.

    :param vector: cartesian vector to express
    :type vector: ndarray(N,dtype=float)
    :param basis: basis vectors in cartesian space
    :type basis: ndarray((N,N),dtype=float)
    :return: components of direction of vector in basis
    :rtype: ndarray(N,dtype=int)
    """    
    return integer_indices(components_in_basis(vector,basis))
    
def components_in_basis(vector,basis):
    """Calculate the components of the N-dimensional vector in the given basis.

    :param vector: cartesian vector to express
    :type vector: ndarray(N,dtype=float)
    :param basis: basis vectors in cartesian space
    :type basis: ndarray((N,N),dtype=float)
    :return: components of vector in basis
    :rtype: ndarray(N,dtype=float)
    """
    return np.einsum('i,ij',vector,np.linalg.inv(basis))

def integer_indices(vector, limit=10000, tol=1e-3):
    """Return a multiple of the given N-dimensional vector such that the components are integer.

    :param vector: cartesian vector to express
    :type vector: ndarray(N,dtype=float)
    :param limit: maximum multiple to consider, defaults to 10000
    :type limit: int, optional
    :param tol: Tolerance used to determine when a component is integer (max rounding error), defaults to 1e-3
    :type tol: float, optional
    :return: multiple of input vestor such that components are integer
    :rtype: ndarray(N,dtype=float)
    """    
    vp = vector/np.min(np.abs(vector[np.where(vector!=0.0)]))
    found = False
    i = 1
    while (not found) and (i<=limit):
        if np.sum(abs(np.round(vp*i)-vp*i)) < tol:
            found = True
        i = i + 1
    vp = vp*(i-1)
    if found:
        return np.round(vp,0)
    else:
        return np.array([0.0,0.0,0.0])

def vector_in_basis(components,basis_vectors):
    """Return an N-dimensional vector specified by N components as multiples of the N basis vectors.

    :param components: multiples of the basis vectors 
    :type components: ndarray(N,dtype=float)
    :param basis: basis vectors in cartesian space
    :type basis: ndarray(N,dtype=float)
    :return: cartesian vector
    :rtype: ndarray(N,dtype=float)
    """    
    return np.einsum('i,ij',components,basis_vectors)
        
def rotation_matrix(axis,theta):
    """Calculate and return a rotation matrix from an axis-angle combination.

    :param axis: axis
    :type axis: ndarray(3,dtype=float)
    :param theta: angle, in radians
    :type theta: float
    :return: rotation matrix in cartesian space
    :rtype: ndarray((3,3),dtype=float)
    """    
    #print(axis)
    axis = axis/math.sqrt(np.dot(axis,axis))
    c = math.cos(theta)
    s = math.sin(theta)
    x,y,z = axis
    return np.array([[c+x*x*(1.0-c), x*y*(1.0-c)-z*s, x*z*(1.0-c)+y*s],
                     [y*x*(1.0-c)+z*s, c+y*y*(1.0-c), y*z*(1.0-c)-x*s], 
                     [z*x*(1.0-c)-y*s, z*y*(1.0-c)+x*s, c+z*z*(1.0-c)]])

def rotation_matrix_into_direction(vector, direction):
    """Rotation matrix required to rotate a given vector into a given direction.

    :param vector: vector in cartesian space, vector to rotate
    :type vector: ndarray(3,dtype=float)
    :param direction: vector in cartesian space, target direction to rotate input vector into
    :type direction: ndarray(3,dtype=float)
    :return: rotation matrix in cartesian space
    :rtype: ndarray((3,3),dtype=float)
    """    
    
    v = np.cross(vector,direction)/np.linalg.norm(vector)/np.linalg.norm(direction)
    c = np.dot(vector,direction)/np.linalg.norm(vector)/np.linalg.norm(direction)
    vmat = np.array([
        [0,-v[2],v[1]],
        [v[2],0,-v[0]],
        [-v[1],v[0],0]
    ])
    if c == -1:
        rotation = -np.identity(3)
    else:
        rotation = np.identity(3) + vmat + np.dot(vmat,vmat)*(1.0/(1.0+c))
    return rotation
                     
def rotate_cell(cellvectors,axis,angle):
    """Rotate a set of cell vectors according to a psecified axis and angle.

    :param cellvectors: description of cell in terms of basis vectors
    :type cellvectors: ndarray((3,3),dtype=float)
    :param axis: axis
    :type axis: ndarray(3,dtype=float)
    :param angle: angle, in radians
    :type angle: float
    :return: rotated cell in terms of basis vectors
    :rtype: ndarray((3,3),dtype=float)
    """    
    R = rotation_matrix(axis,angle)
    return np.transpose(np.dot(R,np.transpose(cellvectors)))
        
def get_ortho_bounding_box(boxvectors):
    """Calculate the smallest orthorhombic bounding box (aligned with the cartesian axes) that circumscribes a parallelipiped.

    :param boxvectors: vectors (in rows of an array) defining a parallelipiped
    :type boxvectors: ndarray((3,3),dtype=float)
    :return: max and min of bounding box in three cartesian directions
    :rtype: ndarray((3,2),dtype=float)
    """    
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
    initmin = 999999.0
    boundingbox = np.array([[initmin,-initmin],[initmin,-initmin],[initmin,-initmin]])
    for i in range(8):
        corner = corners[i,0]*boxvectors[0] + corners[i,1]*boxvectors[1] + corners[i,2]*boxvectors[2]
        for s in range(3):
            if corner[s] < boundingbox[s,0]:
                boundingbox[s,0] = corner[s]
            if corner[s] > boundingbox[s,1]:
                boundingbox[s,1] = corner[s]
    return boundingbox

def get_bounding_box(boxvectors, cellvectors):
    """Calculate the smallest bounding box in the specified basis that circumscribes a parallelipiped.

    :param boxvectors: vectors (in rows of an array) defining a parallelipiped
    :type boxvectors: ndarray((3,3),dtype=float)
    :param cellvectors: basis in which bounding box is to be specified
    :type cellvectors: ndarray((3,3),dtype=float)
    :return: max and min of bounding box in three basis vector directions
    :rtype: ndarray((3,2),dtype=float)
    """    
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
    initmin = 999999.0
    boundingbox = np.array([[initmin,-initmin],[initmin,-initmin],[initmin,-initmin]])
    for i in range(8):
        corner = corners[i,0]*boxvectors[0] + corners[i,1]*boxvectors[1] + corners[i,2]*boxvectors[2]
        for s in range(3):
            dist = np.dot(cellvectors[s,:],corner)/np.linalg.norm(cellvectors[s,:])**2
            if dist < boundingbox[s,0]:
                boundingbox[s,0] = dist
            if dist > boundingbox[s,1]:
                boundingbox[s,1] = dist
    #print(boundingbox)
    return boundingbox

def get_repeats(supercellvectors,cellvectors, axis=None,angle=None):
    """Return the number of repeats of a cell along each lattice vector required to fill a supercell box with atoms. 
    Optionally the cell can be rotated (as specified by an axis-angle combination) prior to filling the supercell.

    :param supercellvectors: vectors describing supercell to fill
    :type supercellvectors: ndarray((3,3),dtype=float)
    :param cellvectors: vectors describing cell to be used to fill supercell
    :type cellvectors: ndarray((3,3),dtype=float)
    :param axis: axis about which to rotate the cell, defaults to None
    :type axis: ndarray(3,dtype=float), optional
    :param angle: angle by which to rotate the cell, defaults to None
    :type angle: float, optional
    :return: max and min of bounding box in three basis vector directions
    :rtype: ndarray((3,2),dtype=float)
    """    
    if axis is not None:
        rcellvectors = rotate_cell(cellvectors,axis,angle)
    else:
        rcellvectors = cellvectors
    boundingbox = get_bounding_box(supercellvectors, rcellvectors)
    for s in range(3):
        boundingbox[s,0] = int(boundingbox[s,0] - 1)
        boundingbox[s,1] = int(boundingbox[s,1] + 1)
    return boundingbox
    
def fill_box(supercellvectors, cellvectors, basis, repeats, basis_types=None, overfill=False):
    """Return a list of atom coordinates filling a box defined by supercellvectors, using a set of lattice
    vectors given by cellvectors and a basis. Repeats contains the search bounds required to fill the box.

    :param supercellvectors: vectors describing supercell to fill
    :type supercellvectors: ndarray((3,3),dtype=float)
    :param cellvectors: vectors describing cell to be used to fill supercell
    :type cellvectors: ndarray((3,3),dtype=float)
    :param basis: positions of atoms within the unit cell
    :type basis: ndarray((3,3),dtype=float)
    :param repeats: min and max values in direction of each cell vector in order to tile supercell
    :type repeats: ndarray((3,2),dtype=float)
    :param basis_types: array of integers specifiying type of atoms within unit cell, defaults to None
    :type basis_types: ndarray(M,dtype=int), optional
    :param overfill: set to True to overfill the box. WARNING! useful for visualisations, but will probably break periodicity with periodic boundaries! Defaults to False
    :type overfill: bool, optional
    :raises RuntimeError: _description_
    :return: (number of atoms, coordinates, types) or (number of atoms, coordinates)
    :rtype: (int, ndarray((N,3),dtype=float), ndarray(N,dtype=float)) or (int, ndarray((N,3),dtype=float))
    """    
    nbasis = len(basis)
    if basis_types is not None:
        if len(basis_types) != nbasis:
            raise RuntimeError("Array of atom types must match length of basis")
    r = []
    t = []
    
    x = np.arange(int(repeats[0,0]),int(repeats[0,1]), 1)
    y = np.arange(int(repeats[1,0]),int(repeats[1,1]), 1)
    z = np.arange(int(repeats[2,0]),int(repeats[2,1]), 1)
    V = len(x) * len(y) * len(z)
    indices = (np.stack(np.meshgrid(x, y, z)).T).reshape(V, 3)
    for i in range(V):
        for p in range(nbasis):
            pos = (indices[i,0] + basis[p,0])*cellvectors[0,:] + (indices[i,1] + basis[p,1])*cellvectors[1,:] + (indices[i,2] + basis[p,2])*cellvectors[2,:]
            if is_in_cell(supercellvectors, pos, overfill=overfill):
                r.append(pos.tolist())
                if basis_types is not None:
                    t.append(basis_types[p])
    if basis_types is not None:
        return len(r),r,t
    else:
        return len(r),r
    
def is_in_cell(cellvectors, pos, tol=1e-6, overfill=False):
    """Test whether a poition vector lies within a cell (specified via edges of a parallelipiped).

    :param cellvectors: parallelipiped to test
    :type cellvectors: ndarray((3,3),dtype=float)
    :param pos: position to test
    :type pos: ndarray(3,dtype=float)
    :param tol: tolerance for testing, defaults to 1e-6
    :type tol: float, optional
    :param overfill: set to True to overfill the box. WARNING! useful for visualisations, but will probably break periodicity with periodic boundaries! Defaults to False
    :type overfill: bool, optional
    :return: is the position within the cell?
    :rtype: boolean
    """    
    incell = True
    for i in range(3):
        cross = np.cross(cellvectors[(i+1)%3,:],cellvectors[(i+2)%3,:])
        test = np.dot(pos,cross)/np.dot(cellvectors[i,:],cross)
        if not overfill:
            if test < 0.0-tol or test >= 1.0-tol:
                incell = False
        else:
            if test < 0.0-tol or test > 1.0+tol:
                incell = False
    return incell
    
def wrap_to_cell(cellvectors, pos):
    """Wrap a coordinate into a cell assuming periodic boundary conditions.

    :param cellvectors: parallelipiped to test
    :type cellvectors: ndarray((3,3),dtype=float)
    :param pos: position to test
    :type pos: ndarray(3,dtype=float)
    :return: wrapped position
    :rtype: ndarray(3,dtype=float)
    """    
    for i in range(3):
        cross = np.cross(cellvectors[(i+1)%3,:],cellvectors[(i+2)%3,:])
        test = np.dot(pos,cross)/np.dot(cellvectors[i,:],cross)
        if test < 0.0:
            pos = pos + cellvectors[i,:]
        elif test >= 1.0:
            pos = pos - cellvectors[i,:]
    return pos
    
def translate_coordinates(r, dr, cellvectors):
    """Translate all the coordinates in the array r by the vector dr, then rewrap to the box specified by cellvectors.

    :param r: original positions
    :type r: ndarray((N,3),dtype=float)
    :param dr: translation vector
    :type dr: ndarray(3,dtype=float)
    :param cellvectors: parallelipiped to test
    :type cellvectors: ndarray((3,3),dtype=float)
    :return: translated positions
    :rtype: ndarray((N,3),dtype=float)
    """    
    rp = np.array(r) + dr
    for i in range(np.shape(rp)[0]):
        rp[i,:] = wrap_to_cell(cellvectors, rp[i,:])
    return rp

def wrap_vector_to_cell(cellvectors, vec):
    """Wrap a vector into a cell assuming periodic boundary conditions.

    :param cellvectors: cell to wrap to
    :type cellvectors: ndarray((3,3),dtype=float)
    :param vec: original vector
    :type vec: ndarray(3,dtype=float)
    :return: wrapped vector
    :rtype: ndarray(3,dtype=float)
    """    
    for i in range(3):
        cross = np.cross(cellvectors[(i+1)%3,:],cellvectors[(i+2)%3,:])
        test = np.dot(vec,cross)/np.dot(cellvectors[i,:],cross)
        if test < -0.5:
            vec = vec + cellvectors[i,:]
        elif test >= 0.5:
            vec = vec - cellvectors[i,:]
    return vec

def calculate_corners_ppp(cellvectors,repeats):
    """Calculate the coordinates of the corners of a parallelipiped defined by three vectors.

    :param cellvectors: vectors defining edges of parallelipiped
    :type cellvectors: ndarray((3,3),dtype=float)
    :param repeats: min and max of repeats in three basis vector directions (see bounding box functions)
    :type repeats: ndarray((3,2),dtype=float)
    :return: cartesian coordinates of corners
    :rtype: ndarray((8,3),dtype=float)
    """    
    corners = np.zeros([8,3], dtype=float)
    cornerindex = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
    for i in range(8):
        for s in range(3):
            corners[i,:] = corners[i,:] + repeats[s,cornerindex[i,s]]*cellvectors[s,:]
    return corners
    
def calculate_edges_ppp():
    """Define the edges of a parallelipiped consistent with the ordering in calculate_corners_ppp function.

    :return: edges of parallelipiped specified as pairs of integers indexing corners at either end of edge
    :rtype: ndarray((8,3),dtype=int)
    """    
    edges = np.array([[0,1],[0,2],[0,3],[1,4],[2,4],[4,7],[5,7],[3,5],[3,6],[6,7],[1,6],[2,5]])
    return edges

def vis_data_lines(points):
    """Return data for visualising a series of line segments in a plotly scatter plot segments run from points[i,0,:] to points[i,1,:].

    :param points: array of starting and ending points in pairs defining line segments
    :type points: ndarray((N,2,3),dtype=float)
    :return: lists of coordinates of points spearated by cartesian direction
    :rtype: (list[N,2], list[N,2], list[N,2])
    """    
    data = []
    for i in range(len(points)):
        data.append(points[i,0,:].tolist())
        data.append(points[i,1,:].tolist())
        data.append([None,None,None]) 
    dataarray = np.array(data)
    return dataarray[:,0].tolist(), dataarray[:,1].tolist(), dataarray[:,2].tolist()
    
def vis_data_vectors(points, vectors):
    """Return data for visualising a set of vectors in a plotly scatter plot
    vectors run from points[i,:] to points[i,:]+vectors[i,:].

    :param points: array of starting points for vectors
    :type points: ndarray((N,3),dtype=float)
    :param vectors: array of vectors
    :type vectors: ndarray((N,3),dtype=float)
    :return: lists of coordinates of points spearated by cartesian direction
    :rtype: (list[N,2], list[N,2], list[N,2])
    """    
    data = []
    for i in range(len(points)):
        data.append(points[i,:].tolist())
        data.append((points[i,:] + vectors[i,:]).tolist())
        data.append([None,None,None]) 
    dataarray = np.array(data)
    return dataarray[:,0].tolist(), dataarray[:,1].tolist(), dataarray[:,2].tolist()
    
def vis_data_box(corners, edges):
    """Return data for visualising a box in a plotly scatter plot.

    :param corners: array of coordinates of corners
    :type corners: ndarray((N,3),dtype=float)
    :param edges: array holding starting and ending corner indices for edges
    :type edges: ndarray((N,2),dtype=int)
    :return: lists of coordinates of points spearated by cartesian direction
    :rtype: (list[N,2], list[N,2], list[N,2])
    """    
    data = []
    for i in range(len(edges)):
        data.append(corners[edges[i,0],:])
        data.append(corners[edges[i,1],:])
        data.append([None,None,None])
    dataarray = np.array(data)
    return dataarray[:,0].tolist(), dataarray[:,1].tolist(),dataarray[:,2].tolist()

def vis_data_box_ppp(cellvectors, repeats):
    """Return a list of points for visualising a parallelipiped in a plotly scatter plot.

    :param cellvectors: vectors defining edges of cell
    :type cellvectors: ndarray((3,3),dtype=float)
    :param repeats: number of repeats of cell
    :type repeats: ndarray((3,2),dtype=float)
    :return: lists of coordinates of points spearated by cartesian direction
    :rtype: (list[N,2], list[N,2], list[N,2])
    """    
    corners = calculate_corners_ppp(cellvectors, repeats)
    edges = calculate_edges_ppp()
    x,y,z = vis_data_box(corners, edges)
    return x,y,z
    
def lammps_dump_to_ppd(dump_cell):
    """Take a supercell as specified in a lammps dump file and convert to three edge vectors of a parallelipiped.

    :param dump_cell: array describing cell in lammps dump format
    :type dump_cell: ndarray((3,3),dtype=float)
    :return: cell described as array of edge vectors of a parallelipiped
    :rtype: ndarray((3,3),dtype=float)
    """    
    xy = dump_cell[0,2]; xz = dump_cell[1,2]; yz = dump_cell[2,2]
    xlo = dump_cell[0,0] - min(0.0, xy, xz, xy+xz)
    xhi = dump_cell[0,1] - max(0.0, xy, xz, xy+xz)
    ylo = dump_cell[1,0] - min(0.0, yz)
    yhi = dump_cell[1,1] - max(0.0, yz)
    zlo = dump_cell[2,0]
    zhi = dump_cell[2,1]
    ppd = np.array([
        [xhi-xlo, 0.0, 0.0],
        [xy, yhi-ylo, 0.0],
        [xz, yz, zhi-zlo]
    ])
    return ppd
    
def lammps_box_to_ppd(box_cell):
    """Take a supercell as specified in a lammps input file and convert to three edge vectors of a parallelipiped.

    :param box_cell: array describing cell in lammps input format
    :type box_cell: ndarray((3,3),dtype=float)
    :return: cell described as array of edge vectors of a parallelipiped
    :rtype: ndarray((3,3),dtype=float)
    """
    xy = box_cell[0,2]; xz = box_cell[1,2]; yz = box_cell[2,2]
    xlo = box_cell[0,0]
    xhi = box_cell[0,1]
    ylo = box_cell[1,0]
    yhi = box_cell[1,1]
    zlo = box_cell[2,0]
    zhi = box_cell[2,1]
    ppd = np.array([
        [xhi-xlo, 0.0, 0.0],
        [xy, yhi-ylo, 0.0],
        [xz, yz, zhi-zlo]
    ])
    return ppd

def mirror_boundary_atoms(cell, atom_data, tol=0.02):
    """Augment atom list so that atoms on boundaries and corners are reproduced on opposite side (a cosmetic adjustment).

    :param cell: cell
    :type cell: ndarray((3,3),dtype=float)
    :param atom_data: array of coordinates of atoms
    :type atom_data: ndarray((N,3),dtype=float)
    :param tol: tolerance to determine if an atom is on the boundary surface of the cell, defaults to 0.02
    :type tol: float, optional
    :return: array of coordinates of atoms augmented with copies of atoms on the cell surface
    :rtype: ndarray((>=N,3),dtype=float)
    """    
    n_atoms = np.shape(atom_data)[0]
    r_new = []
    for i in range(n_atoms):
        r_new.append(atom_data[i,:].tolist())
        pattern = []
        for s in range(3):
            cross = np.cross(cell[(s+1)%3,:],cell[(s+2)%3,:])
            test = np.dot(atom_data[i,0:3],cross)/np.dot(cell[s,:],cross)
            this_pattern = np.array([0,0,0], dtype=int)
            if test < tol:
                this_pattern[s] = 1
                pattern.append(this_pattern.tolist())
            elif test > (1.0-tol):
                this_pattern[s] = -1
                pattern.append(this_pattern.tolist())
        if np.shape(pattern)[0]==2:
            pattern.append((np.array(pattern[0]) + np.array(pattern[1])).tolist())
        elif np.shape(pattern)[0]==3:
            pattern.append((np.array(pattern[0]) + np.array(pattern[1])).tolist())
            pattern.append((np.array(pattern[1]) + np.array(pattern[2])).tolist())
            pattern.append((np.array(pattern[2]) + np.array(pattern[0])).tolist())
            pattern.append((np.array(pattern[0]) + np.array(pattern[1]) +  + np.array(pattern[2])).tolist())
        pattern = np.array(pattern)
        for s in range(np.shape(pattern)[0]):
            r_shift = np.copy(atom_data[i,:])
            shift=np.zeros(3)
            for t in range(3):
                r_shift[0:3] = r_shift[0:3] + pattern[s,t]*cell[t,:]
                shift = shift + pattern[s,t]*cell[t,:]
            r_new.append(r_shift.tolist())
    return np.array(r_new)
    
def locate_nearest_atom(r0, r):
    """return the index and position of the atom within an array of atom coordinates nearest to a test point.

    :param r0: coordinate of test point
    :type r0: ndarray(3,dtype=float)
    :param r: coordinates of atoms 
    :type r: ndarray((3,3),dtype=float)
    :return: index of closest atom, coordinates of closest atom
    :rtype: int, ndarray((3),dtype=float)
    """    
    idx = np.linalg.norm(r - r0, axis=1).argmin()
    return idx, r[idx,:]
    
def find_bonds(cell, r, rnn, wrap=False):
    """Return an array of atom pairs that are closer together than a specified separation. Only considers each pair once.

    :param cell: array describing cell (required in order to account for periodic boundaries if wrap=True)
    :type cell: ndarray((3,3),dtype=float)
    :param r: atom coordinates
    :type r: ndarray((N,3),dtype=float)
    :param rnn: cut-off distance for determining neighbours
    :type rnn: float
    :param wrap: flag to select if periodic boundaries applied, defaults to False
    :type wrap: bool, optional
    :return: array of indices of atom pairs
    :rtype: ndarray((M,2),dtype=float)
    """    
    n_atoms = np.shape(r)[0]
    pairs = []
    for i in range(n_atoms):
        for j in range(i+1,n_atoms):
            if wrap:
                d = r[i,:]-r[j,:]
                d = wrap_vector_to_cell(cell, d)
                d = np.linalg.norm(d)
            else:
                d = np.linalg.norm(r[i,:]-r[j,:])
            if d <= rnn:
                pairs.append([i+1,j+1])
    return np.array(pairs)

def find_neighbours(cell, r, r_nn, wrap=True, max_neigh=16):
    """Find a list of neighbours and their relative positions for all the atoms in an array of position coordinates. 
    Use find_neighbours_single() to get this information for a single atom only.

    :param cell: Supercell cell description. Required in order to treat wrapping at boundaries
    :type cell: ndarray((3,3),dtype=float)
    :param r: Array of atom coordinates to examine
    :type r: ndarray((N,3),dtype=float)
    :param r_nn: Cut-off distance within which to treat atoms as neighbours
    :type r_nn: float
    :param wrap: Flag to indicate if boundaries are periodic, defaults to True
    :type wrap: bool, optional
    :param max_neigh: Maximum nunber of neighbours to record, defaults to 16. Ensure this is large enough to avoid missing neighbours within r_nn
    :type max_neigh: int, optional
    :return: Number of neighbours of each atom, indices of these neighbouring atoms in input array, distance to neighbours, displacement vectors of neighbours
    :rtype: ndarray((N),dtype=int), ndarray((N,max_neigh),dtype=int), ndarray((N,max_neigh),dtype=float), ndarray((N,max_neigh,3),dtype=float)
    """    
    n_atoms = np.shape(r)[0]
    n_neigh = np.zeros(n_atoms, dtype=int)
    neigh = np.zeros((n_atoms,max_neigh), dtype=int)
    neigh_sep = np.zeros((n_atoms,max_neigh), dtype=float)
    neigh_disp = np.zeros((n_atoms,max_neigh,3), dtype=float)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                d = r[i,:]-r[j,:]
                if wrap:
                    sep = np.linalg.norm(wrap_vector_to_cell(cell, d))
                else:
                    sep = np.linalg.norm(d)
                if sep <= r_nn:
                    neigh[i,n_neigh[i]] = j
                    neigh_sep[i,n_neigh[i]] = sep
                    neigh_disp[i,n_neigh[i],:] = d
                    n_neigh[i] = n_neigh[i] + 1
    return n_neigh, neigh, neigh_sep, neigh_disp
    
def find_neighbours_single(atom, cell, r, r_nn, wrap=True, max_neigh=16):
    """Find a list of neighbours and their relative positions for a specified atom in an array of position coordinates. 
    Use find_neighbours() to get this information for all atoms in the array.

    :param atom: Index of atom of interest
    :type atom: int
    :param cell: Supercell cell description. Required in order to treat wrapping at boundaries
    :type cell: ndarray((3,3),dtype=float)
    :param r: Array of atom coordinates to examine
    :type r: ndarray((N,3),dtype=float)
    :param r_nn: Cut-off distance within which to treat atoms as neighbours
    :type r_nn: float
    :param wrap: Flag to indicate if boundaries are periodic, defaults to True
    :type wrap: bool, optional
    :param max_neigh: Maximum nunber of neighbours to record, defaults to 16. Ensure this is large enough to avoid missing neighbours within r_nn
    :type max_neigh: int, optional
    :return: Number of neighbours of atom, indices of these neighbouring atoms in input array, distance to neighbours, displacement vectors of neighbours
    :rtype: int, ndarray((max_neigh),dtype=int), ndarray((max_neigh),dtype=float), ndarray((max_neigh,3),dtype=float)
    """    
    n_atoms = np.shape(r)[0]
    n_neigh = 0
    neigh = np.zeros(max_neigh, dtype=int)
    neigh_sep = np.zeros(max_neigh, dtype=float)
    neigh_disp = np.zeros((max_neigh,3), dtype=float)
    i = atom
    for j in range(n_atoms):
        if j != i:
            d = r[i,:]-r[j,:]
            if wrap:
                sep = np.linalg.norm(wrap_vector_to_cell(cell, d))
            else:
                sep = np.linalg.norm(d)
            if sep <= r_nn:
                neigh[n_neigh] = j
                neigh_sep[n_neigh] = sep
                neigh_disp[n_neigh,:] = d
                n_neigh = n_neigh + 1
    return n_neigh, neigh, neigh_sep, neigh_disp

def get_interstices(cell,r,pad,r_cluster):
    """Take a cell specification and list of atom positions and find the interstices via a voronoi tesselation.
    The atom array is augmented by an amount specified via <pad> on all faces and edges to mimic periodic boundaries
    in a way that Voronoi can handle. 
    The detected vertices of the voronoi mesh are then pruned if they are closer than <r_cluster>.

    :param cell: Supercell cell description. Required in order to treat wrapping at boundaries
    :type cell: ndarray((3,3),dtype=float)
    :param r: Array of atom coordinates to examine
    :type r: ndarray((N,3),dtype=float)
    :param pad: Width of padding region to apply at all surfaces of supercell
    :type pad: float
    :param r_cluster: Distance below which two voronoi points are considered to coincide and are pruned
    :type r_cluster: float
    :return: Array of coordinates of interstices
    :rtype: ndarray((M,3),dtype=float)
    """    
    # Generate a padded cell and augmented atom list
    trans = np.array([
        [1,0,0], [0,1,0], [0,0,1], 
        [1,1,0], [0,1,1], [1,0,1], [-1,1,0], [0,-1,1], [1,0,-1],
        [1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]
    ])
    cell_pad = np.zeros((3,3))
    shift = np.zeros(3)
    for s in range(3):
        cell_pad[s,:] = cell[s,:] * (1.0 + 2.0*pad/np.linalg.norm(cell[s,:]))
        shift = shift + cell[s,:] * 1.0*pad/np.linalg.norm(cell[s,:])
    r_pad = r.tolist()
    for i in range(np.shape(trans)[0]):
        r_pad.extend((r+np.dot(trans[i,:],cell)).tolist())
        r_pad.extend((r-np.dot(trans[i,:],cell)).tolist())
    r_to_prune = np.array(r_pad) + shift
    r_pad = []
    for i in range(np.shape(r_to_prune)[0]):
        if is_in_cell(cell_pad, r_to_prune[i,:], tol=1e-6):
            r_pad.extend([r_to_prune[i,:].tolist()])
    # Get voronoi tesselation
    vor = Voronoi(np.array(r_pad))
    # First trim out distant vertices
    v_to_prune = vor.vertices
    v_trim = []
    for i in range(np.shape(v_to_prune)[0]):
        if is_in_cell(cell_pad, v_to_prune[i,:], tol=1e-6):
            v_trim.extend([v_to_prune[i,:].tolist()])
    # Now check for clusters of vertices close together
    v_to_cluster = np.array(v_trim)
    num_v = np.shape(v_to_cluster)[0]
    check_v = np.full(num_v, True, dtype=bool)
    v_clustered = []
    for i in range(num_v):
        if check_v[i]:
            cluster_r = [v_to_cluster[i,:].tolist()]
            n_cluster = 1
            check_v[i] = False
            for j in range(i+1,num_v):
                if check_v[j]:
                    if np.abs(np.linalg.norm(v_to_cluster[i,:] - v_to_cluster[j,:])) < r_cluster:
                        cluster_r.extend([v_to_cluster[j,:].tolist()])
                        n_cluster = n_cluster + 1
                        check_v[j] = False
            #print(n_cluster,cluster_r)
            if n_cluster > 1:
                v_clustered.extend([np.mean(np.array(cluster_r),axis=0).tolist()])
            else:
                v_clustered.extend(cluster_r)
    # Now trim atoms and vertices back to original cell
    v_to_prune = np.array(v_clustered) - shift
    v_trim = []
    for i in range(np.shape(v_to_prune)[0]):
        if is_in_cell(cell, v_to_prune[i,:], tol=1e-6):
            v_trim.extend([v_to_prune[i,:].tolist()])
    return np.array(v_trim)
    
def get_voronoi_volumes(cell,r,pad):
    """Take a cell specification and list of atom positions and find the volumes associated with each atom via a voronoi tesselation.
    The atom array is augmented by an amount specified via <pad> on all faces and edges to mimic periodic boundaries in a way that Voronoi can handle.

    :param cell: Supercell cell description. Required in order to treat wrapping at boundaries
    :type cell: ndarray((3,3),dtype=float)
    :param r: Array of atom coordinates to examine
    :type r: ndarray((N,3),dtype=float)
    :param pad: Width of padding region to apply at all surfaces of supercell
    :type pad: float
    :return: Array of atom (voronoi cell) volumes
    :rtype: ndarray((N),dtype=float)
    """    
    n_atoms = np.shape(r)[0]
    # Generate a padded cell and augmented atom list
    trans = np.array([
        [1,0,0], [0,1,0], [0,0,1], 
        [1,1,0], [0,1,1], [1,0,1], [-1,1,0], [0,-1,1], [1,0,-1],
        [1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]
    ])
    cell_pad = np.zeros((3,3))
    shift = np.zeros(3)
    for s in range(3):
        cell_pad[s,:] = cell[s,:] * (1.0 + 2.0*pad/np.linalg.norm(cell[s,:]))
        shift = shift + cell[s,:] * 1.0*pad/np.linalg.norm(cell[s,:])
    r_pad = r.tolist()
    for i in range(np.shape(trans)[0]):
        r_pad.extend((r+np.dot(trans[i,:],cell)).tolist())
        r_pad.extend((r-np.dot(trans[i,:],cell)).tolist())
    r_to_prune = np.array(r_pad) + shift
    r_pad = []
    for i in range(np.shape(r_to_prune)[0]):
        if is_in_cell(cell_pad, r_to_prune[i,:], tol=1e-6):
            r_pad.extend([r_to_prune[i,:].tolist()])
    # Get voronoi tesselation
    vor = Voronoi(np.array(r_pad))
    # Now get volume for each point
    vol = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(vor.vertices[indices]).volume
    return np.array(vol)[:n_atoms]
    
def multiply_cell(cell,r,mults):
    """Take a cell specification and list of atom positions and return a larger cell with integer numbers of copies of the original positions.
    If other attributes are included in the array r, these will also be replicated.

    :param cell: Supercell cell description. Required in order to treat wrapping at boundaries
    :type cell: ndarray((3,3),dtype=float)
    :param r: Array of atom coordinates to examine
    :type r: ndarray((N,3),dtype=float)
    :param mults: Number of multiples of cell to return in each of the three cell directions
    :type pad: ndarray((3),dtype=int)
    :return: Enlarged supercell cell description, Augmented array of atom coordinates
    :rtype: ndarray((3,3),dtype=float), ndarray((N,3),dtype=float)
    """    
    n_cols = np.shape(r)[1]
    cell_pad = np.zeros((3,3))
    for s in range(3):
        cell_pad[s,:] = cell[s,:] * mults[s]
    r_pad = []
    for i in range(mults[0]):
        for j in range(mults[1]):
            for k in range(mults[2]):
                shift = np.zeros(n_cols)
                shift[:3] = i*cell[0,:] + j*cell[1,:] + k*cell[2,:]
                r_pad.extend( (r + shift).tolist() )
    return cell_pad, np.array(r_pad)

def read_lammps_dump(file):
    """Read a lammps dump file and return the number of atoms, the supercell specification and a subset of atom data.
    (Warning! This is not a well written or general purpose parser for Lammps dump files. It will work if the format 
    matches that expected, but may well fall over if not!)

    :param file: Name of file to read
    :type file: str
    :return: Number of atoms, supercell cell description, Array of atom coordinates (plus Type, ID, PE if in file)
    :rtype: int, ndarray((3,3),dtype=float), ndarray((N,M),dtype=float)
    """    
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>3 and words[3] == 'ATOMS' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex+1):
        f.readline()
    num_atoms = int(f.readline().split()[0])
    f.close()
    # Get supercell shape
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>1 and words[1] == 'BOX' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    supercell_shape = np.zeros((3,3), dtype=float)
    f = open(file)
    for s in range(lineIndex+1):
        f.readline()
    for s in range(3):
        words = f.readline().split()
        if len(words)==3:
            supercell_shape[s,:] = [words[0], words[1], words[2]]
        elif len(words)==2:
            supercell_shape[s,:2] = [words[0], words[1]]
    f.close()
    supercell = lammps_dump_to_ppd(supercell_shape)
    # Get atom data
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>1 and words[1] == 'ATOMS' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex):
        f.readline()
    words = f.readline().split()
    fields = 6
    x_in = None; y_in = None; z_in = None; type_in = None; pe_in = None; id_in = None
    for s in range(len(words)):
        if words[s] == 'x':
            x_in = s-2
        elif words[s] == 'y':
            y_in = s-2
        elif words[s] == 'z':
            z_in = s-2
        elif words[s] == 'type':
            type_in = s-2
        elif words[s] == 'c_peatom':
            pe_in = s-2
        elif words[s] == 'id':
            id_in = s-2
    atom_data = np.zeros((num_atoms,fields), dtype=float)
    for s in range(num_atoms):
        words = f.readline().split()
        if x_in is not None:
            atom_data[s,0] = words[x_in]
        if y_in is not None:
            atom_data[s,1] = words[y_in]
        if z_in is not None:
            atom_data[s,2] = words[z_in]
        if type_in is not None:
            atom_data[s,3] = words[type_in]
        if pe_in is not None:
            atom_data[s,4] = words[pe_in]
        if id_in is not None:
            atom_data[s,5] = words[id_in]
    f.close()
    return num_atoms, supercell, atom_data
    
def read_lammps_input(file):
    """Read structure and coordinates from a Lammps input fiule and return the number of atoms, the supercell specification and a subset of atom data.
    (Warning! This is not a well written or general purpose parser for Lammps input files. It will work if the format 
    matches that expected, but may well fall over if not!)

    :param file: Name of file to read
    :type file: str
    :return: Number of atoms, supercell cell description, Array of atom coordinates, ID and Type
    :rtype: int, ndarray((3,3),dtype=float), ndarray((N,5),dtype=float)
    """   
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>1 and words[1] == 'atoms' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex):
        f.readline()
    num_atoms = int(f.readline().split()[0])
    f.close()
    
    box_cell = np.zeros((3,3), dtype=float)
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>2 and words[2] == 'xlo' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex):
        f.readline()
    for s in range(3):
        words = f.readline().split()
        box_cell[s,0] = float(words[0])
        box_cell[s,1] = float(words[1])
    f.close()
    supercell = lammps_box_to_ppd(box_cell)
    
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>3 and words[3] == 'xy' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    if found_line:
        f = open(file)
        for s in range(lineIndex):
            f.readline()
        words = f.readline().split()
        for s in range(3):
            box_cell[s,2] = float(words[s])
        f.close()
        
    atom_data = np.zeros((num_atoms,6), dtype=float)
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>0 and words[0] == 'Atoms' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex+2):
        f.readline()
    for s in range(num_atoms):
        words = f.readline().split()
        atom_data[s,0] = float(words[2]); atom_data[s,1] = float(words[3]); atom_data[s,2] = float(words[4])
        atom_data[s,3] = float(words[1])
        atom_data[s,5] = float(words[0])
    f.close()    
    return num_atoms, supercell, atom_data

# def write_print_file(r, bond, filename, r_atom=0.15, r_bond=0.05):
#     fo = open(filename, 'w')
#     fo.write('atoms=[\n')
#     for i in range(np.shape(r)[0]):
#         fo.write('[ ')
#         for s in range(3):
#             fo.write(str(r[i,s])+ ', ')
#         fo.write(str(r_atom) + ' ]')
#         if i<np.shape(r)[0]-1:
#             fo.write(',\n')
#         else:
#             fo.write('\n')
#     fo.write('];\n')
#     fo.write('\n')
#     fo.write('bonds=[\n')
#     for i in range(np.shape(bond)[0]):
#         fo.write('[ ')
#         for s in range(2):
#             fo.write(str(bond[i,s])+ ', ')
#         fo.write(str(r_bond) + ' ]')
#         if i<np.shape(r)[0]-1:
#             fo.write(',\n')
#         else:
#             fo.write('\n')
#     fo.write('];\n')
#     fo.close()
    

def write_lammps(supercell, atom_pos, atom_type=None, filename='lammps.txt', num_types=1):
    """Write out a supercell in Lammps format

    :param supercell: Supercell cell description
    :type supercell: ndarray((3,3),dtype=float)
    :param atom_pos: Array of atom coordinates
    :type atom_pos: ndarray((N,3),dtype=float)
    :param atom_type: Array of integers indicating atom type, defaults to None
    :type atom_type: ndarray((N),dtype=int), optional, all atom types set to 1 if not supplied
    :param filename: name of file to write, defaults to 'lammps.txt'
    :type filename: str, optional
    :param num_types: number of atom types in file, defaults to 1
    :type num_types: int, optional
    """ 

    fo = open(filename,'w')
    header = '#Lammps coordinate file'
    fo.write(header)
    fo.write('\n')
    fo.write(str(np.shape(atom_pos)[0]) + ' atoms\n')
    fo.write('\n')
    fo.write(str(num_types) + ' atom types\n')
    fo.write('\n')
    fo.write('0.0 ' + str(supercell[0,0]) + ' xlo xhi\n')
    fo.write('0.0 ' + str(supercell[1,1]) + ' ylo yhi\n')
    fo.write('0.0 ' + str(supercell[2,2]) + ' zlo zhi\n')
    if abs(supercell[1,0]) + abs(supercell[2,0]) + abs(supercell[2,1]) > CRYSTALTOOLS_TOL:
        fo.write(str(supercell[1,0]) + ' ' + str(supercell[2,0]) + ' ' + str(supercell[2,1]) + ' xy xz yz\n')
    fo.write('\n')
    fo.write('Atoms\n')
    fo.write('\n')
    count = 1
    for i in range(np.shape(atom_pos)[0]):
        fo.write(str(count) + ' ')
        if atom_type is not None: 
            fo.write(str(int(atom_type[i])) + ' ') 
        else:
            fo.write('1 ') 
        fo.write(str(atom_pos[i,0]) + ' ' + str(atom_pos[i,1]) + ' ' + str(atom_pos[i,2]) + '\n')
        count = count + 1
    fo.flush()
    fo.close()
    return

def write_poscar(supercell, atom_pos, filename='POSCAR', type_string=None, constraints=None):
    """Write out a supercell in Vasp POSCAR format

    :param supercell: Supercell cell description
    :type supercell: ndarray((3,3),dtype=float)
    :param atom_pos: Array of atom coordinates
    :type atom_pos: ndarray((N,3),dtype=float)
    :param filename: Name of file to write, defaults to 'lammps.txt'
    :type filename: str, optional
    :param type_string: String to indicate numbers of atoms in Vasp format, defaults to None, resulting in all atoms of single species
    :type type_string: str, optional
    :param constraints: Constraints to apply to individual atoms, defaults to None, resulting in T T T for all atoms
    :type constraints: N*3 multi-dimensional list of strings, optional

    """ 
    supercell = supercell
    r = atom_pos
    num_atoms = np.shape(r)[0]
        
    fo = open(filename,'w')
    header = '#VASP coordinate file'
    fo.write(header)
    fo.write('\n')
    # Lattice parameter already factored into supercell and coordinates
    fo.write('1.0\n')
    # Supercell shape
    order = [0,1,2]
    if np.dot(supercell[0,:],np.cross(supercell[1,:],supercell[2,:])) < 0:
         order = [1,0,2]
    for s in range(3):
        for t in range(3):
            fo.write(str(supercell[order[s],t]) + ' ')
        fo.write('\n')
    # Atom positions
    if type_string is None:
        fo.write(str(num_atoms) + '\n')
    else:
        fo.write(type_string + '\n')
    fo.write('Selective dynamics\n')
    fo.write('Cartesian\n')
    for i in range(num_atoms):
        fo.write(str(r[i,0]) + ' ' + str(r[i,1]) + ' ' + str(r[i,2]))
        if constraints is not None:
            fo.write(' ' + constraints[i,0] + ' ' + constraints[i,1] + ' ' + constraints[i,2])
        else:
            fo.write(' T T T')
        fo.write('\n')           
    fo.flush()
    fo.close()