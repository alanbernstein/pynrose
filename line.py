import numpy as np
from sympy import Point
from sympy.geometry import Line

"""
| Section          | SimpleLine | LineWrapper |
| define object    |   0.000273 |    0.000241 |
| define lines     |   0.000273 |    2.956732 |
| define vertices  |   0.000217 |    0.795405 |
| connect vertices |   0.000038 |    0.000033 |
| get traces       |   0.000042 |    0.000040 |
| plot             |   0.126805 |    0.190878 |
"""


class SimpleLine(object):
    """
    A simple line class that only handles intersections and origin distance.
    replaces sympy's Line, because that's too slow.
    """
    def __init__(self, **kwargs):
        if 'p1' in kwargs and 'p2' in kwargs:
            self._init_point_point(**kwargs)
        elif 'a' in kwargs and 'b' in kwargs and 'c' in kwargs:
            self._init_standard(**kwargs)
        else:
            raise Exception('Line format not supported')

    def _init_point_point(self, **kwargs):
        self.p1 = kwargs['p1']
        self.p2 = kwargs['p2']
        # if self.p1.imag != 0:
        #     self.p1 = [self.p1.real, self.p1.imag]
        self.a = self.p1[1] - self.p2[1]
        self.b = self.p2[0] - self.p1[0]
        self.c = (self.p2[0] - self.p1[0])*self.p1[1] - (self.p2[1]-self.p1[1])*self.p1[0]

    def _init_standard(self, **kwargs):
        self.a = kwargs['a']
        self.b = kwargs['b']
        self.c = kwargs['c']

    def intersect(self, other):
        a1, b1, c1 = self.a, self.b, self.c
        a2, b2, c2 = other.a, other.b, other.c
        denom = a1*b2 - b1*a2
        x = (c1*b2 - b1*c2)/denom
        y = (a1*c2 - c1*a2)/denom
        return (x, y)

    def origin_distance(self):
        return np.abs(self.c) / np.sqrt(self.b**2 + self.a**2)


class LineWrapper(object):
    """
    A wrapper around sympy's line class that uses the same interface as SimpleLine
    for easy switching
    """
    def __init__(self, **kwargs):
        if 'p1' in kwargs and 'p2' in kwargs:
            self._init_point_point(**kwargs)
        elif 'a' in kwargs and 'b' in kwargs and 'c' in kwargs:
            self._init_general(**kwargs)
        else:
            raise Exception('Line format not supported')

    def _init_general(self, **kwargs):
        self.a = kwargs['a']
        self.b = kwargs['b']
        self.c = kwargs['c']

    def _init_point_point(self, **kwargs):
        self.p1 = kwargs['p1']
        self.p2 = kwargs['p2']
        self.line = Line(Point(kwargs['p1'][0], kwargs['p1'][1]),
                         Point(kwargs['p2'][0], kwargs['p2'][1]))

    def intersect(self, other):
        pt = self.line.intersect(other.line)
        return map(float, list(set(pt))[0])


def test_simple_line():
    p1 = np.random.random(2)
    p2 = np.random.random(2)
    p3 = np.random.random(2)
    #l1 = Line(Point(p1[0], p1[1]), Point(p2[0], p2[1]))
    #l2 = Line(Point(p2[0], p2[1]), Point(p3[0], p3[1]))
    l1 = LineWrapper(p1=p1, p2=p2)
    l2 = LineWrapper(p1=p2, p2=p3)
    L1 = SimpleLine(p1=p1, p2=p2)
    L2 = SimpleLine(p1=p2, p2=p3)

    # v1 = list(set(l1.intersect(l2)))[0]
    v1 = l1.intersect(l2)
    print(p2)
    print(L1.intersect(L2))
    print([float(v) for v in v1])


# test_simple_line()
