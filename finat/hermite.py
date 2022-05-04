import numpy

import FIAT
from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 3:
            raise ValueError("Degree must be 3 for Hermite element")
        if Citations is not None:
            Citations().register("Ciarlet1972")
        super().__init__(FIAT.CubicHermite(cell))

    def basis_transformation(self, coordinate_mapping):
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        h = coordinate_mapping.cell_size()

        d = self.cell.get_dimension()
        numbf = self.space_dimension()

        M = numpy.eye(numbf, dtype=object)

        for multiindex in numpy.ndindex(M.shape):
            M[multiindex] = Literal(M[multiindex])

        cur = 0
        for i in range(d+1):
            cur += 1  # skip the vertex
            J = Js[i]
            for j in range(d):
                for k in range(d):
                    M[cur+j, cur+k] = J[j, k] / h[i]
            cur += d

        return ListTensor(M)


class HighOrderHermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if cell.get_dimension() != 1:
            raise ValueError("High-order Hermite element currently only implemented in 1 dimension")
        if degree < 3:
            raise ValueError("Degree must be at least 3 for high-order Hermite element")
        super().__init__(FIAT.HighOrderHermite(cell, degree))
        
    def basis_transformation(self, coordinate_mapping):
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        h = coordinate_mapping.cell_size()

        numbf = self.space_dimension()

        M = numpy.eye(numbf, dtype=object)

        for multiindex in numpy.ndindex(M.shape):
            M[multiindex] = Literal(M[multiindex])

        J = Js[0]
        M[2, 2] = J[1, 1] / h[0]
        J = Js[1]
        M[self.degree + 1, self.degree + 1] = J[1, 1] / h[1]

        return ListTensor(M)
