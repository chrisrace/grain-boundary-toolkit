# spatialsearch.py
""" Module to define a spatialseaerch object for increasing the efficiency of grid-based searches
Author:  Chris Race
Date:    3rd January 2017
Contact: christopher.race@manchester.ac.uk
"""
import numpy as np

class SpatialSearch(object):
    """A spatialsearch object holds an array of position vectors on a grid
    defined by integer multiples of a set of basis vectors
    in cartesian space within a circle or sphere of a given radius and 
    their distances from the origin, sorted by distance. Note that these distances 
    assume a cubic basis. Application in a non cubic system will mean that vectors are
    not considered in completely correct size order.

    The purpose of this object is to speed up grid-based searches

    :ivar nrarray((nspace,[3,4]),dtype=float) space: Numpy array containing vectors and lengths (in fourth column) of those vectors from origin to points on a simple square or cubic grid
    :ivar int nspace: Number of vectors in array
    """
    
    def __init__(self, maxcomponent, basis_vectors = None, dimension = 3):
        """Populates the array of search points in a spatialsearch object

        :param maxcomponent: Radius of search sphere. Corresponds to largest component of any vectors returned
        :type maxcomponent: int
        :param basis_vectors: Array of basis vectors in rows, defaults to None, in which case a square grid (cubic basis) is assumed
        :type basis_vectors: ndarray((3,3), dtype=float), optional
        :param dimension: dimensionality of space, defaults to 3
        :type dimension: int, optional
        :raises RuntimeError: "Dimension for search space must be 2 or 3" if incorrect value for dimension specified
        """        
        # Ordered list of relative vectors to help speed up searches later on
        
        if dimension == 3:
            spaceradius = maxcomponent
            self.nspace = (2*spaceradius+1)*(2*spaceradius+1)*(2*spaceradius+1)-1
            self.space = np.zeros((self.nspace,4))
            tempspace = np.zeros((self.nspace+1,4))
            for x in range(-spaceradius,spaceradius+1):
                #print 'Building space array:', x
                for y in range(-spaceradius,spaceradius+1):
                    for z in range(-spaceradius,spaceradius+1):
                        tempspace[(2*spaceradius+1)*((x+spaceradius)*(2*spaceradius+1)+(y+spaceradius))+(z+spaceradius),0] = x
                        tempspace[(2*spaceradius+1)*((x+spaceradius)*(2*spaceradius+1)+(y+spaceradius))+(z+spaceradius),1] = y
                        tempspace[(2*spaceradius+1)*((x+spaceradius)*(2*spaceradius+1)+(y+spaceradius))+(z+spaceradius),2] = z
                        if basis_vectors is not None:
                            tempspace[(2*spaceradius+1)*((x+spaceradius)*(2*spaceradius+1)+(y+spaceradius))+(z+spaceradius),3] = np.linalg.norm(np.einsum('i,ij',[x,y,z],basis_vectors))
                        else:
                            tempspace[(2*spaceradius+1)*((x+spaceradius)*(2*spaceradius+1)+(y+spaceradius))+(z+spaceradius),3] = np.sqrt(x*x+y*y+z*z)
            # sort array by last column (length)
            idx = np.argsort(tempspace[:,3],0)
            for i in range(1,self.nspace+1):
                self.space[i-1,:] = tempspace[idx[i],:]
        elif dimension == 2:
            spaceradius = maxcomponent
            self.nspace = (2*spaceradius+1)*(2*spaceradius+1)-1
            self.space = np.zeros((self.nspace,3))
            tempspace = np.zeros((self.nspace+1,3))
            for x in range(-spaceradius,spaceradius+1):
                #print 'Building space array:', x
                for y in range(-spaceradius,spaceradius+1):
                    tempspace[(2*spaceradius+1)*((x+spaceradius)*(2*spaceradius+1)+(y+spaceradius))+(z+spaceradius),0] = x
                    tempspace[(2*spaceradius+1)*((x+spaceradius)*(2*spaceradius+1)+(y+spaceradius))+(z+spaceradius),1] = y
                    if basis_vectors is not None:
                        tempspace[(2*spaceradius+1)*(x+spaceradius)+(y+spaceradius),2] = np.linalg.norm(np.einsum('i,ij',[x,y],basis_vectors))
                    else:
                        tempspace[(2*spaceradius+1)*(x+spaceradius)+(y+spaceradius),2] = np.sqrt(x*x+y*y)
            # sort array by last column (length)
            idx = np.argsort(tempspace[:,2],0)
            for i in range(1,self.nspace+1):
                self.space[i-1,:] = tempspace[idx[i],:]
        else:
            raise RuntimeError("Dimension for search space must be 2 or 3")
            
if __name__ == "__main__":
    # Test code
    print('Testing spatialsearch.py with a simple square grid of radius 5')
    mysearch = SpatialSearch(5)
    print(mysearch.space)
