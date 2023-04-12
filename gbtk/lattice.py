# lattice.py
""" Module to store and manipulate crystal lattices
Author:  Chris Race
Date:    3rd January 2017
Contact: christopher.race@manchester.ac.uk
"""

import numpy as np
from . import symmetry as symmetry
from . import crystaltools as ct

# For visualisation only
try:
    import plotly
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)
except ModuleNotFoundError:
    pass

LATTICE_TOL = 1e-3

def print_supported_types():
    print('currently supported lattice types are:')
    supported_types = ['fcc', 'bcc', 'sc', 'fcc-bct', 'hcp', 'hcp-ortho', 'CsCl', 'ZrO2-fct']
    print(supported_types)

class Lattice(object):
    """A lattice holds a description of the basic crystal structure. This description
    includes the vectors defining the edges of the unit cell and the fractional coordinates
    of the atoms within that cell.

    :ivar string lattice_type: Type of lattice. Can be fcc, bcc, sc, fcc-bct, hcp, hcp-ortho, CsCl, ZrO2-fct
    :ivar ndarray((3,3),dtype=float) cell_vectors: numpy array storing vectors defining unit cell parallelipiped in rows
    :ivar ndarray((num_basis,3),dtype=float) basis_coords: Basis - positions of atoms in unit cell in fractional coordinates (fractions of cell_vectors)
    :ivar integer num_basis: Number of atoms in basis
    :type num_basis: integer
    :ivar ndarray(num_basis,dtype=int) atom_types: Types of atoms in basis, specified as consecutive integers indicating differnt types from 1 to num_atom_types
    :ivar int num_atom_types: Number of atom types in unit cell
    :ivar object symmetries: Symmetries of the lattice type, currently only populated for fcc lattice type
    """    

    def __init__(self,lattice_type,lengths=None,angles=None):
        """Initialises a lattice of a specified type and dimensions
        Dimensions are relative to the a-direction cell length
        Shape of the cell is specified by a series of optional values for lengths and angles. 
        The number of these values will be different for different lattice types as follows:

                fcc, bcc, sc, fcc-bct, CsCl: 0

                hcp, hcp-ortho, ZrO2-fct: 0 => ideal, 1 => c/a ratio, 

        :param lattice_type: Type of lattice. Can be fcc, bcc, sc, fcc-bct, hcp, hcp-ortho, CsCl, ZrO2-fct
        :type lattice_type: string
        :param lengths: Lengths of unit cell vectors, defaults to None
        :type lengths: list of floats, optional
        :param angles: Angles between unit cell vectors, defaults to None
        :type angles: list of floats, optional
        :raises RuntimeError: "Cell shape specification not required for latticetype", if angles or lengths specified when neither required
        :raises RuntimeError: "Cell angle specification not required for latticetype", if angles specified when not required
        :raises RuntimeError: "If specifying cell shape specify only c/a ratio for latticetype", if shape specified, but incorrect number of lengths given
        :raises RuntimeError: "Unknown lattice type", if unsupported value specified for lattice_type
        """        

        if lattice_type == 'null':
            if (lengths is not None) or (angles is not None):
                raise RuntimeError("Cell shape specification not required for latticetype" + lattice_type)
        elif lattice_type == 'fcc':
            if (lengths is not None) or (angles is not None):
                raise RuntimeError("Cell shape specification not required for latticetype" + lattice_type)
            if (angles is not None):
                raise RuntimeError("Cell angle specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0,0.0,0.0],
                                        [0.0,1.0,0.0],
                                        [0.0,0.0,1.0]
                                        ])
            self.basis_coords = (np.array([
                                        [0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.0],
                                        [0.0, 0.5, 0.5],
                                        [0.5, 0.0, 0.5]
                                        ]))
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1, 1, 1, 1])
            self.num_atom_types = 1
            self.symmetries = symmetry.Symmetry(lattice_type)
        elif lattice_type == 'sc':
            if (lengths is not None) or (angles is not None):
                raise RuntimeError("Cell shape specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0,0.0,0.0],
                                        [0.0,1.0,0.0],
                                        [0.0,0.0,1.0]
                                        ])
            self.basis_coords = (np.array([
                                        [0.0, 0.0, 0.0]
                                        ]))
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1])
            self.num_atom_types = 1
        elif lattice_type == 'bcc':
            if (lengths is not None) or (angles is not None):
                raise RuntimeError("Cell shape specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0,0.0,0.0],
                                        [0.0,1.0,0.0],
                                        [0.0,0.0,1.0]
                                        ])
            self.basis_coords = (np.array([
                                        [0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.5]
                                        ]))
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1,1])
            self.num_atom_types = 1
        elif lattice_type == 'hcp':
            if (lengths is not None):
                if (len(lengths) == 1):
                    c_over_a = lengths[0]
                else:
                    raise RuntimeError("If specifying cell shape specify only c/a ratio for latticetype" + lattice_type)
            else:
                c_over_a = np.sqrt(8.0/3.0)
            if (angles is not None):
                raise RuntimeError("Cell angle specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0,0.0,0.0],
                                        [-0.5,np.sqrt(3.0)/2.0,0.0],
                                        [0.0,0.0,c_over_a]
                                        ])
            # Basis below places centre of inversion at origin after shift
            self.basis_coords = (np.array([
                                        [0.0, 0.0, 0.0],
                                        [2.0/3.0, 1.0/3.0, 0.5],
                                        ]))
            self.basis_coords = self.basis_coords + np.array([[1.0-1.0/3.0, 1.0-1.0/6.0, 1.0-0.25],[-1.0/3.0, -1.0/6.0, -0.25]])
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1,1])
            self.num_atom_types = 1
        elif lattice_type == 'hcp-ortho':
            if (lengths is not None):
                if (len(lengths) == 1):
                    c_over_a = lengths[0]
                else:
                    raise RuntimeError("If specifying cell shape specify only c/a ratio for latticetype" + lattice_type)
            else:
                c_over_a = np.sqrt(8.0/3.0)
            if (angles is not None):
                raise RuntimeError("Cell angle specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0,0.0,0.0],
                                        [0.0,np.sqrt(3.0),0.0],
                                        [0.0,0.0,c_over_a]
                                        ])
            # Basis below places centre of inversion at origin after shift
            self.basis_coords = (np.array([
                                        [0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.0],
                                        [0.5, 1.0/6.0, 0.5],
                                        [0.0, 4.0/6.0, 0.5]
                                        ]))
            self.basis_coords = self.basis_coords + np.array([ [1.0-0.25, 1.0-1.0/12.0, 1.0-0.25], [-0.25, -1.0/12.0, 1.0-0.25], [-0.25, -1.0/12.0, -0.25], [1.0-0.25, -1.0/12.0, -0.25] ])
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1,1,1,1])
            self.num_atom_types = 1
        elif lattice_type == 'fcc-bct':
            if (lengths is not None) or (angles is not None):
                raise RuntimeError("Cell shape specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0/np.sqrt(2),0.0,0.0],
                                        [0.0,1.0/np.sqrt(2),0.0],
                                        [0.0,0.0,1.0]
                                        ])
            self.basis_coords = (np.array([
                                        [0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.5]
                                        ]))
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1,1])
            self.num_atom_types = 1
        elif lattice_type == 'CsCl':
            if (lengths is not None) or (angles is not None):
                raise RuntimeError("Cell shape specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0,0.0,0.0],
                                        [0.0,1.0,0.0],
                                        [0.0,0.0,1.0]
                                        ])
            self.basis_coords = (np.array([
                                        [0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.5]
                                        ]))
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1,2])
            self.num_atom_types = 2
        elif lattice_type == 'ZrO2-fct':
            if (lengths is not None):
                if (len(lengths) == 1):
                    c_over_a = lengths[0]
                else:
                    raise RuntimeError("If specifying cell shape specify only c/a ratio for latticetype" + lattice_type)
            else:
                c_over_a = 5.278128659040220 / 5.115796532474760
            if (angles is not None):
                raise RuntimeError("Cell angle specification not required for latticetype" + lattice_type)
            self.cell_vectors = np.array([
                                        [1.0,0.0,0.0],
                                        [0.0,1.0,0.0],
                                        [0.0,0.0,c_over_a]
                                        ])
            # Basis below places centre of inversion at origin
            self.basis_coords = (np.array([
                                [ 0.50,    0.75,    0.750],
                                [ 0.00,    0.75,    0.250],
                                [ 0.00,    0.25,    0.750],
                                [ 0.50,    0.25,    0.250],
                                [ 0.75,    0.00,    0.957],
                                [ 0.25,    0.50,    0.957],
                                [ 0.25,    0.00,    0.543],
                                [ 0.75,    0.50,    0.543],
                                [ 0.75,    0.00,    0.457],
                                [ 0.25,    0.50,    0.457],
                                [ 0.25,    0.00,    0.043],
                                [ 0.75,    0.50,    0.043]
                                        ]))
            # Basis below places a Zr atom at origin
            # self.basis_coords = (np.array([
            #                     [0.000000000000000,       0.000000000000000,       0.000000000000000],
            #                     [0.500000000000000,       0.000000000000000,      0.500000000000000],
            #                     [0.500000000000000,       0.500000000000000,       0.000000000000000],
            #                     [0.000000000000000,       0.50000000000000,        0.500000000000000],
            #                     [0.250000000000000,       0.250000000000000,       0.207000000000000],
            #                     [0.750000000000000,       0.750000000000000,       0.207000000000000],
            #                     [0.750000000000000,       0.250000000000000,      -0.207000000000000],
            #                     [0.250000000000000,       0.750000000000000,      -0.207000000000000],
            #                     [0.250000000000000,       0.250000000000000,       0.707000000000000],
            #                     [0.750000000000000,       0.750000000000000,       0.707000000000000],
            #                     [0.750000000000000,       0.250000000000000,       0.293000000000000],
            #                     [0.250000000000000,       0.750000000000000,       0.293000000000000]
            #                             ]))
            self.num_basis = len(self.basis_coords)
            self.atom_types = np.array([1,1,1,1,2,2,2,2,2,2,2,2])
            #self.atom_types = np.array([1,1,1,1])
            self.num_atom_types = 2
        else:
            raise RuntimeError("Unknown lattice type")
            
        self.lattice_type = lattice_type
    
    def copy_lattice(self, lattice):
        """Copy attributes from supplied lattice object

        :param lattice: lattice object to copy attributes from
        :type lattice: lattice object
        """        

        self.cell_vectors = np.copy(lattice.cell_vectors)
        self.basis_coords = np.copy(lattice.basis_coords)
        self.num_basis = lattice.num_basis
        self.atom_types = np.copy(lattice.atom_types)
        self.num_atom_types = lattice.num_atom_types
    
    try:
        def visualise_3d(self):
            """Generate a plot showing the lattice unit cell
            """            
            data = []
            colours = ['rgb(155, 0, 0)', 'rgb(0, 155, 0)', 'rgb(0, 0, 155)',  'rgb(155, 155, 0)', 'rgb(0, 155, 155)', 'rgb(155, 0, 155)']
            # Define scatter plot trace for atoms
            for s in range(self.num_atom_types):
                x,y,z = [],[],[]
                for i in range(self.num_basis):
                    if self.atom_types[i] == s+1:
                        new_point = ct.vector_in_basis(self.basis_coords[i,:],self.cell_vectors)
                        x.append(new_point[0])
                        y.append(new_point[1])
                        z.append(new_point[2])
                name = 'Atom type ' + str(s+1)
                trace = go.Scatter3d(
                    x = x, y = y, z = z, mode = 'markers', name = name,
                    marker = dict(size = 5, color = colours[s])
                )
                data.append(trace)
        
            # Define scatter plot trace for unit cell
            x,y,z = ct.vis_data_box_ppp(self.cell_vectors, np.array([[0,1],[0,1],[0,1]]))
            trace = go.Scatter3d(
                x = x, y = y, z = z,  mode = 'lines', name = 'Unit cell',
                line = dict(width = 2, color = 'rgb(0, 0, 0)')
            )
            data.append(trace)
    
            layout = go.Layout(
                width = 800, height = 500,
                title = "Lattice unit cell",
                xaxis = dict( nticks = 10, domain = [0, 0.9]),
                yaxis = dict(scaleanchor = "x")
            )

            plotly.offline.iplot({
                "data": data,
                "layout": layout
            })
    except NameError:
        pass

    def print_lattice_spec(self):
        """Print details of lattice object
        """
        print('Details of lattice')
        print('------------------')
        print('Lattice type:')
        print(self.lattice_type)
        print()
        print('Cell vectors:')
        print(self.cell_vectors)
        print()
        print('Number of atoms in basis:')
        print(self.num_basis)
        print()
        print('Atom types:')
        print(self.atom_types)
        print()
        print('Number of atom types')
        print(self.num_atom_types)
        print('Positions of atoms')
        print(self.basis_coords)
        print()


# def vector_equivalence(v1,v2, T, tol=LATTICE_TOL):
#     """Test if two vectors are equivalent under one of the transformations in T"""
#     equiv = False
#     for i in range(np.shape(T)[0]):
#         v = np.dot(T,v2)
#         if abs(np.linalg.norm(v-v1)) < tol:
#             equiv = True
#     return equiv

if __name__ == "__main__":
    # Test code
    print('Testing lattice.py')
    mylattice = Lattice('fcc')
    mylattice.print_lattice_spec()