# symmetry.py
""" Module to define and store symmetry operations of crystal lattices
Author:  Chris Race
Date:    3rd January 2017
Contact: christopher.race@manchester.ac.uk
"""

import numpy as np

SYMMETRY_TOL = 1e-3

class Symmetry(object):
    """Calculates and stores a set of symmettry operations (as matrices) for a given crystal lattice

    :ivar ndarray((N,3),dtype=float) mirror_planes: numpy array plane normals of N mirror planes
    :ivar ndarray((N,3),dtype=float) rot_2: numpy array storing axes of N two-fold rotations
    :ivar ndarray((N,3),dtype=float) rot_3: numpy array storing axes of N three-fold rotations
    :ivar ndarray((N,3),dtype=float) rot_4: numpy array storing axes of N four-fold rotations
    :ivar ndarray((N,3,3),dtype=float) symops: numpy array storing transformation matrices for full set of N syymetry operations (including identity and inversion)
    :ivar integer n_trans: Number of symmetry operations in symops
    :ivar equiv_vecs: CODE CURRENTLY COMMENTED OUT Probably defunct
    :ivar boolean symops_set: Set to true when symmetry operations calcualated (i.e. when symops populated)
    """
    
    def __init__(self,lattice_type):
        """Initialises the symmetry properties lattice of a specified type.
        Currently supported types are: fcc, sc, bcc

        :param lattice_type: lattice type to consider
        :type lattice_type: string
        :raises RuntimeError: "Lattice type must be specified." where no lattice type specified
        :raises RuntimeError: "Lattice type " + lattice_type + " not supported." where specified lattice type not supported
        """        
        test = np.zeros(3)
        if lattice_type == 'null':
            raise RuntimeError("Lattice type must be specified.")
        elif lattice_type == 'fcc' or lattice_type == 'bcc' or lattice_type == 'sc':
            self.mirror_planes = np.array([
                [1, 0, 0],[0, 1, 0],[0, 0, 1],
                [1, 1, 0]/np.sqrt(2),[0, 1, 1]/np.sqrt(2),[1, 0, 1]/np.sqrt(2),
                [-1, 1, 0]/np.sqrt(2),[0, -1, 1]/np.sqrt(2),[1, 0, -1]/np.sqrt(2),
            ], dtype='float')
            self.rot_2 = np.array([
                [1,1,0]/np.sqrt(2),[0,1,1]/np.sqrt(2),[1,0,1]/np.sqrt(2),
                [1,-1,0]/np.sqrt(2),[0,1,-1]/np.sqrt(2),[-1,0,1]/np.sqrt(2)
            ], dtype='float')
            self.rot_3 = np.array([
                [1,1,1]/np.sqrt(3),[-1,1,1]/np.sqrt(3),[1,-1,1]/np.sqrt(3),[1,1,-1]/np.sqrt(3)
            ], dtype='float')
            self.rot_4 = np.array([
                [1,0,0],[0,1,0],[0,0,1]
            ], dtype='float') 
            
            # self.equiv_vecs = np.array([
            #     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            #     [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            #     [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            #     [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            #     [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
            #     [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            #     [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            #     [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
            #     [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            #     [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
            #     [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            #     [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
            #     [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            #     [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
            #     [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            #     [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
            #     [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            #     [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            #     [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
            #     [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            #     [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
            #     [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
            #     [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
            #     [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
            # ])
            
            # permutations = np.array([[0,1,2], [2,0,1], [1,2,0], [2,1,0], [1,0,2], [0,2,1]])
            # nps = 6
            # signs = np.array([[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]])
            # nss = 4
            # self.equiv_vecs = np.zeros((2*nps*nss,3,3), dtype=int)
            # for p in range(nps):
            #     for s in range(nss):
            #         op = np.zeros((3,3), dtype=int)
            #         for t in range(3):
            #             op[t,permutations[p,t]] = signs[s,t]
            #         self.equiv_vecs[p*nss + s,:,:] = op
            #         self.equiv_vecs[nps*nss + p*nss + s,:,:] = -op
        else:
            raise RuntimeError("Lattice type " + lattice_type + " not supported.")
        
        
        self.n_trans = np.shape(self.mirror_planes)[0] + np.shape(self.rot_2)[0] + 2*np.shape(self.rot_3)[0] + 3*np.shape(self.rot_4)[0] + 2
        self.symops = np.zeros((self.n_trans,3,3), dtype=float)
        self.symops_set = False
        if self.n_trans > 2:
            self.symops_set = True
        opcount = 0
        for i in range(np.shape(self.mirror_planes)[0]): # Mirror planes
            self.symops[opcount,:,:] = np.identity(3)-2*np.outer(self.mirror_planes[i,:],self.mirror_planes[i,:])
            opcount = opcount + 1
        
        for i in range(len(self.rot_2)): # Two-fold rotations
            x,y,z = self.rot_2[i,:]
            angles = [np.pi]
            cs = np.cos(angles)
            ss = np.sin(angles)
            for j in range(len(angles)):
                c = cs[j]; s = ss[j]
                self.symops[opcount,:,:] = np.array([[c+x*x*(1.0-c), x*y*(1.0-c)-z*s, x*z*(1.0-c)+y*s],
                                 [y*x*(1.0-c)+z*s, c+y*y*(1.0-c), y*z*(1.0-c)-x*s], 
                                 [z*x*(1.0-c)-y*s, z*y*(1.0-c)+x*s, c+z*z*(1.0-c)]])
                opcount = opcount + 1
        
        for i in range(len(self.rot_3)): # Three-fold rotations
            x,y,z = self.rot_3[i,:]
            angles = [2*np.pi/3, 4*np.pi/3]
            cs = np.cos(angles)
            ss = np.sin(angles)
            for j in range(len(angles)):
                c = cs[j]; s = ss[j]
                self.symops[opcount,:,:] = np.array([[c+x*x*(1.0-c), x*y*(1.0-c)-z*s, x*z*(1.0-c)+y*s],
                                 [y*x*(1.0-c)+z*s, c+y*y*(1.0-c), y*z*(1.0-c)-x*s], 
                                 [z*x*(1.0-c)-y*s, z*y*(1.0-c)+x*s, c+z*z*(1.0-c)]])
                opcount = opcount + 1
        
        for i in range(len(self.rot_4)): # Four-fold rotations
            x,y,z = self.rot_4[i,:]
            angles = [np.pi/2, 2*np.pi/2, 3*np.pi/2]
            cs = np.cos(angles)
            ss = np.sin(angles)
            for j in range(len(angles)):
                c = cs[j]; s = ss[j]
                self.symops[opcount,:,:] = np.array([[c+x*x*(1.0-c), x*y*(1.0-c)-z*s, x*z*(1.0-c)+y*s],
                                 [y*x*(1.0-c)+z*s, c+y*y*(1.0-c), y*z*(1.0-c)-x*s], 
                                 [z*x*(1.0-c)-y*s, z*y*(1.0-c)+x*s, c+z*z*(1.0-c)]])
                opcount = opcount + 1
        self.symops[opcount,:,:] = np.identity(3) # Identity
        opcount = opcount + 1
        self.symops[opcount,:,:] = -np.identity(3) # Inversion
        opcount = opcount + 1
        #print(self.n_trans)
        #print(self.symops)

        
        # round down any near zeros
        self.symops = np.around(self.symops, 6)

    # def sym_equiv(self, v1,v2, tol=SYMMETRY_TOL):
    #     """ Check to see if two vectors are symmetrically equivalent"""
    #     equiv = False
    #     for i in range(self.n_trans):
    #         trans_v1 = np.dot(self.symops[i,:,:],v1)
    #         if np.abs(np.linalg.norm(trans_v1 - v2)) < tol:
    #             equiv = True
    #             break
    #     return equiv

    # def vector_equiv(self, v1,v2, tol=SYMMETRY_TOL):
    #     """ Check to see if two vectors are symmetrically equivalent"""
    #     equiv = False
    #     for i in range(np.shape(self.equiv_vecs)[0]):
    #         trans_v1 = np.dot(self.equiv_vecs[i,:,:],v1)
    #         #if np.abs(np.linalg.norm(trans_v1 - v2)) < tol:
    #         if np.abs(1.0-np.abs(np.dot(trans_v1,v2)/np.linalg.norm(trans_v1)/np.linalg.norm(v2))) < tol:
    #             equiv = True
    #             break
    #     return equiv
        


