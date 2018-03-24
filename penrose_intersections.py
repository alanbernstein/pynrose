#!/usr/bin/env python
import sys
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from plotlytools import category10

# from sympy import Point
# from sympy.geometry import Line


"""
This is an ASCII drawing of a Penrose triangle, or impossible triangle.
In the terminology of the Penrose class below, it has
sides = 3
segments = 2

It is hard to use this diagram to explain the meaning of the "segment_ratio" variable...

 /\
|\ \
| \ \
|  \ \
|   \ \
| |\ \ \
| | \ \ \
| | |\ \ \
| | | \ \ \
| | |  \ \ \
| | |   \/ /|
| | |   / / |
| | |  / / /
| | | / / /
| | |/ / /
| | | / /
| | |/ /
| |   /
| |  /
| | /
 \|/
"""

# there were two other short-lived approaches:
# turtle:
# define all turn angles and lengths, directly, in terms of a handful of base
# variables. then just run those through a "turtle draw" controller.
# this got unwieldy quickly.
#
# 3d:
# draw a polyhedron in 3d space, then render it on a depth buffer with some
# clever non-euclidean occlusion rules. i still like this idea, but i'm
# not really sure if it's feasible. i didn't get too far before coming up
# with the Right Way for 2d drawings...
#
# instead of these, just draw a grid of lines emanating from the base polygon
# then find the appropriate intersections and connect them.
# bonus: works with any polygon.

plot_args = dict(
    auto_open=True,
    # output_type='div',
    show_link=False,
    config={'displayModeBar': False}
)


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
            self._init_general(**kwargs)
        else:
            raise Exception('Line format not supported')

    def _init_point_point(self, **kwargs):
        """
        y-y1 = (y2-y1)/(x2-x1) (x-x1)
        (x2-x1)(y-y1) = (y2-y1)(x-x1)
        (x2-x1)(y-y1) - (y2-y1)(x-x1) = 0
        a = y1 - y2
        b = x2 - x1
        c = (x2-x1)*y1 - (y2-y1)*x1

        """
        self.p1 = kwargs['p1']
        self.p2 = kwargs['p2']
        self.a = self.p1[1] - self.p2[1]
        self.b = self.p2[0] - self.p1[0]
        self.c = (self.p2[0] - self.p1[0])*self.p1[1] - (self.p2[1]-self.p1[1])*self.p1[0]

    def _init_general(self, **kwargs):
        self.a = kwargs['a']
        self.b = kwargs['b']
        self.c = kwargs['c']

    def intersect(self, other):
        """
        a1 x + b1 y = c1
        a2 x + b2 y = c2
        """
        a1, b1, c1 = self.a, self.b, self.c
        a2, b2, c2 = other.a, other.b, other.c
        denom = a1*b2 - b1*a2
        x = (c1*b2 - b1*c2)/denom
        y = (a1*c2 - c1*a2)/denom
        return (x, y)

    def origin_distance(self):
        # ax + by = c
        # y = c/b - a/b x
        # y = b/a x    (perpendicular through origin)
        # -b x + a y = 0
        #
        # compute intersection of these
        #
        # denom = a*a + b*b
        # x = (c*a)/denom
        # y = (c*b)/denom
        #
        # sqrt((ccaa+ccbb)/(aa+bb)^2)
        # sqrt(ccaa+ccbb)/(aa+bb)
        # c*sqrt(aa+bb)/(aa+bb)
        # c / sqrt(aa+bb)
        # TODO: should be abs(c)
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

def get_polygon_convexity(pts):
    # input:  Nx2 array
    # output: [is_convex, is_convex, ...] x N

    # idea 1
    # 0. figure out an interior point?? (to the left of the rightmost point?)
    # 1. sort by angle
    # 2. walk vertices in order (ascending angle), starting at smallest angle (break ties arbitrarily)
    # 3. check whether "turning left" (convex) or "turning right" (concave)

    # idea 2
    # compute subtended angle for each vertex
    # sum them, take supplementary angle if it's too big
    # if <=180, convex, else concave

    # idea 3
    # 1. compute turn angle via cos(theta) = dot(a,b)/(norm(a)*norm(b))
    i0 = np.arange(len(pts))
    iprev = (i0 - 1) % len(pts)
    inext = (i0 + 1) % len(pts)
    v0 = pts[inext] - pts[i0]
    v1 = pts[i0] - pts[iprev]
    dots = np.sum(v0 * v1, axis=1)
    m0 = np.sqrt(np.sum(v0 * v0, axis=1))
    m1 = np.sqrt(np.sum(v1 * v1, axis=1))
    turn_angle = np.arccos(dots/(m0*m1))
    # 2. convexity is simply defined, assuming CCW traversal of polygon
    is_convex = list(turn_angle - np.pi/2 > 0)
    return is_convex


class Penrose(object):
    line_class = SimpleLine  # SimpleLine for self-contained, LineWrapper to use sympy's Line

    def __init__(self, polygon=None, sides=3, segments=2, segment_ratio=None):
        # TODO: make segment width an explicit parameter, instead of segment_ratio
        # TODO: handle nonconvex polygons?
        # TODO: handle arbitrary "phase"
        # TODO: handle "pass-through vertices" for more complex shapes, e.g. curved sides
        # TODO: variable spacing for multiple segments
        #       (so it looks like it has a more cylindrical cross section)
        # TODO: handle different "skip values" of connecting segments, instead of 1 by default
        # TODO: handle genus 2+ shapes? E.G the letter A instead of a triangle
        # TODO: rewrite/simplify Penrose class to compute intersections lazily (hard)
        # DONE: scale segments equally based on line-origin distance
        # DONE: use lightweight Line class so intersections are faster
        self.Nsides = sides
        self.Nsegments = segments
        if polygon is None:
            # use regular Nsides-sided polygon
            pts = np.exp(np.arange(self.Nsides) * 2j * np.pi / self.Nsides)
            self.polygon = [(p.real, p.imag) for p in pts]
        else:
            if type(polygon) == np.ndarray and polygon.dtype == 'complex128':
                # convert complex to tuples
                self.polygon = [(p.real, p.imag) for p in polygon]
            else:
                self.polygon = polygon
            self.Nsides = len(self.polygon)
        self.segment_ratio = segment_ratio or 1./((self.Nsides-2))
        self.convexity = get_polygon_convexity(np.array(self.polygon))

    def _define_lines(self):
        """
        create a dict of lines like this:
        {
            (side, dist): Line,
            ...
        }
        where:
        side refers to the index of one of the Nsides sides of the base polygon
        dist refers to the index of one of the Nsegments segments coming out of the base polygon
        (0 is the side of the base polygon, 1 is the first segment, etc)
        this generates all the lines needed for finding the intersection lattices at the polygon vertices
        """
        # TODO: this works for any shape such that "moving the line away from the origin"
        # is equivalent to "moving the line away from the interior of the shape".
        # if i want to make this work for any arbitrary polygon, it will need to
        # understand what the interior is and choose the correct direction

        def S(n, k):
            return (n+k) % self.Nsides

        v = self.polygon
        self.lines = {}

        # compute line-origin distances
        ds = []
        for n in range(self.Nsides):
            L = self.line_class(p1=v[n], p2=v[S(n, 1)])
            ds.append(L.origin_distance())
        d_min = np.min(ds)
        # NOTE: could adjust the spacing right here
        k_ratios = [d_min/d for d in ds]

        print('define_lines')
        for n in range(self.Nsides):
            n1 = (n+1) % self.Nsides
            for m in range(self.Nsegments+1):
                # k = 1+self.segment_ratio*m            # naive scale
                k = 1+self.segment_ratio*m*k_ratios[n]  # scale relative to origin distance
                print('  side=%d, segment=%d, k=%f: vertices %d,%d' % (n, m, k, n, n1))

                self.lines[(n, m)] = self.line_class(
                    p1=(k*v[n][0], k*v[n][1]),
                    p2=(k*v[n1][0], k*v[n1][1]),
                )

    def _define_vertices(self):
        """
        create a dict of vertices like this:
        {
            (base-vertex, dist1, dist2): Point2d,
            ...
        }
        where:
        base-vertex refers to the index of one of the Nsides vertices of the base polygon
        (dist1, dist2) is a coordinate on an affine-tranformed cartesian grid:
                       - origin at that vertex
                       - axes aligned with the two adjacent sides
        """
        print('define_vertices')
        self.vertices = {}
        for n1 in range(0, self.Nsides):
            n2 = (n1 + 1) % self.Nsides
            for m1 in range(0, self.Nsegments+1):
                l1 = self.lines[(n1, m1)]
                for m2 in range(0, self.Nsegments+1):
                    l2 = self.lines[(n2, m2)]
                    n = (n1 + 1) % self.Nsides  # increment so vertex keys align with polygon indices
                    # this is because line[n] is defined with points from polygon[n], polygon[n+1]
                    # and then vertex[(n, -, -)] is defined from line[n], line[n+1]
                    # TODO: there is a cleaner way to do this, but have to change other indexes as well...
                    ki = (n, m1, m2)
                    li = l1.intersect(l2)
                    print('  (%d, %d, %d): (%.4g, %.4g)' % (n, m1, m2, li[0], li[1]))
                    self.vertices[ki] = li

    def get_polygon_traces(self, **kwargs):
        # get_*_traces functions are all compatible with plotly:
        # returns a list of dicts that are usable as go.Scatter objects
        # kwargs can be any set of go.Scatter inputs
        #
        # returns a trace of the generating polygon
        x = [p[0] for p in self.polygon + [self.polygon[0]]]
        y = [p[1] for p in self.polygon + [self.polygon[0]]]
        trace = dict(x=x, y=y)
        trace.update(**kwargs)
        return [trace]

    def get_line_traces_dumb(self, **kwargs):
        # returns a list of traces of the generating lines, based on
        # how those lines were originally defined, which might not be useful
        data = []
        for key, line in self.lines.items():
            p1, p2 = line.p1, line.p2
            trace = dict(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                name='%s' % (key,),
            )
            trace.update(**kwargs)
            data.append(trace)
        return data

    def get_line_traces(self, **kwargs):
        # returns a list of traces of the generating lines, by connecting vertices
        # more useful than the "dumb" version, but doesn't work as intended for
        # concave vertices...
        # {
        #   (base-vertex, dist1, dist2): Point2d,
        # }

        print('get_line_traces')
        data = []
        for n in range(self.Nsides):
            n1 = (n+1) % self.Nsides
            print('  side %d (convex=%s, next=%d)' % (n, self.convexity[n], n1))
            for m in range(self.Nsegments+1):
                print('    segment %d' % m)
                # select coordinates in the vertex-frame based on whether its convex or not
                if self.convexity[n]:
                    i0 = (n, 2, m)
                else:
                    i0 = (n, 0, m)
                if self.convexity[n1]:
                    i1 = (n1, m, 2)
                else:
                    i1 = (n1, m, 0)

                trace = dict(
                    x=[self.vertices[i0][0], self.vertices[i1][0]],
                    y=[self.vertices[i0][1], self.vertices[i1][1]],
                    name='(%d, %d)' % (n, m),
                )
                print('      vertices %s, %s' % (i0, i1))

                trace.update(**kwargs)
                data.append(trace)

        return data

    def get_vertex_traces(self, **kwargs):
        # returns a list of named traces of the generating vertices
        data = []
        for k, (x, y) in self.vertices.items():
            trace = dict(x=[x], y=[y], name='%s' % (k,))
            trace.update(**kwargs)
            data.append(trace)
        data.sort(key=lambda x: x['name'])
        return data

    def get_vertex_trace(self, **kwargs):
        # returns a trace of the generating vertices
        x = [v[0] for v in self.vertices.values()]
        y = [v[1] for v in self.vertices.values()]
        trace = dict(x=x, y=y)
        trace.update(**kwargs)
        return [trace]

    def get_segment_traces(self, **kwargs):
        # returns a list of traces of the segments of the penrose object
        data = []
        for part in self.segments:
            trace = dict(
                x=[self.vertices[k][0] for k in part],
                y=[self.vertices[k][1] for k in part],
            )
            trace.update(**kwargs)
            data.append(trace)
        return data

    def _connect_vertices(self):
        # self._connect_vertices_convex()
        self._connect_vertices_nonconvex2()

    def _connect_vertices_nonconvex2(self):
        # connects vertices for any Nsides, but only two segments. convex or nonconvex.
        # given what we learned from manually connected segments for the
        # "complicated" shape, it seems clear how to do this now:
        # - corners behave normally when a segment passes through concave vertices
        # - corners behave differently when a segment starts or ends at a concave vertex
        # - just need to keep track of which vertex the segment is currently at,
        #   and adjust the corner that is added to the segment accordingly
        def S(n, k):
            return (n+k) % self.Nsides

        self.segments = []
        for i in range(self.Nsides):
            # first corner depends on convexity
            if self.convexity[i]:
                part = [(i, 0, 0)]
            else:
                part = [(i, 0, 1), (i, 1, 0)]

            # middle corners are always the same
            for j in range(self.Nsegments):
                part.append((S(i, j+1), j, j+1))

            # last corner depends on convexity
            end = S(i, j+2)
            if self.convexity[end]:
                part.append((end, j+1, j))
                part.append((end, j, j+1))
            else:
                part.append((end, j+1, j+1))

            self.segments.append(part)

    def _connect_vertices_convex(self):
        # connects vertices for any (Nsides, Nsegments)
        # convex only
        def S(n, k):
            return (n+k) % self.Nsides
        self.segments = []
        for i in range(self.Nsides):
            part = [(i, 0, 0)]
            for j in range(self.Nsegments):
                part.append((S(i, j+1), j, j+1))

            part.append((S(i, j+2), j+1, j))
            part.append((S(i, j+2), j, j+1))

            self.segments.append(part)

    def _connect_vertices2(self):
        # for posterity
        # connects vertices for any number of sides, but only for two segments
        # convex only
        def S(n, k):
            return (n+k) % self.Nsides
        self.segments = []
        for i in range(self.Nsides):
            self.segments.append([
                (i, 0, 0),
                (S(i, 1), 0, 1),
                (S(i, 2), 1, 2),
                (S(i, 3), 2, 1),
                (S(i, 3), 1, 2)
            ])

    def _connect_vertices3(self):
        # for posterity
        # connects vertices for any number of sides, but only for three segments
        # convex only
        def S(n, k):
            return (n+k) % self.Nsides
        self.segments = []
        for i in range(self.Nsides):
            self.segments.append([
                (i, 0, 0),
                (S(i, 1), 0, 1),
                (S(i, 2), 1, 2),
                (S(i, 3), 2, 3),
                (S(i, 4), 3, 2),
                (S(i, 4), 2, 3)
            ])

    def _connect_vertices_manual_complicated(self):
        # manually define the connections for a complicated shape with
        # vertex 1 is concave and isolated
        # vertexes 4,5 are concave and adjacent
        # TODO: might also need to investigate what happens when two
        # concave vertices are separated by one convex vertex
        def S(n, k):
            return (n+k) % self.Nsides
        self.segments = [
            # 0 concave
            # 1 convex
            # 2 convex
            # 3 concave
            # 4 concave
            # 5 convex
            # 6 convex
            # normal
            # (i, 0, 0),
            # (S(i, 1), 0, 1),
            # (S(i, 2), 1, 2),
            # (S(i, 3), 2, 1),
            # (S(i, 3), 1, 2)
            [
                # blue. starts and ends at concave
                (0, 0, 1),        #
                (0, 1, 0),        # 0. (0, 0, 0) -> (0, 0, 1), (0, 1, 0)
                (S(0, 1), 0, 1),  # 1. no change
                (S(0, 2), 1, 2),  # 2. no change
                (S(0, 3), 2, 2),  # 3. (3, 2, 1), (3, 1, 2) -> (3, 2, 2)
            ],
            [
                # yellow. ends at concave
                (1, 0, 0),        # 1. no change
                (S(1, 1), 0, 1),  # 2. no change
                (S(1, 2), 1, 2),  # 3. no change
                (S(1, 3), 2, 2),  # 4. (4, 2, 1), (4, 1, 2) -> (4, 2, 2)
            ],
            [
                # green. passes through concave, but doesn't start or end at concave
                (2, 0, 0),        # 2. no change
                (S(2, 1), 0, 1),  # 3. no change
                (S(2, 2), 1, 2),  # 4. no change
                (S(2, 3), 2, 1),  # 5. no change
                (S(2, 3), 1, 2),  # 5. no change
            ],
            [
                # red. starts at concave
                (3, 0, 1),        #
                (3, 1, 0),        # 3. (3, 0, 0) -> (3, 0, 1), (3, 1, 0)
                (S(3, 1), 0, 1),  # 4. no change
                (S(3, 2), 1, 2),  # 5. no change
                (S(3, 3), 2, 1),  # 6. no change
                (S(3, 3), 1, 2),  # 6. no change
            ],
            [
                # purple. starts and ends at concave
                (4, 0, 1),        #
                (4, 1, 0),        # 4. (4, 0, 0) -> (4, 0, 1), (4, 1, 0)
                (S(4, 1), 0, 1),  # 5. no change
                (S(4, 2), 1, 2),  # 6. no change
                (S(4, 3), 2, 2),  # 0. (0, 2, 1), (0, 1, 2) -> (0, 2, 2)
            ],
            [
                # brown. passes through concave
                (5, 0, 0),        # 5. no change
                (S(5, 1), 0, 1),  # 6. no change
                (S(5, 2), 1, 2),  # 0. no change
                (S(5, 3), 2, 1),  # 1. no change
                (S(5, 3), 1, 2),  # 1. no change
            ],
            [
                # pink. passes through concave
                (6, 0, 0),        # 6. no change
                (S(6, 1), 0, 1),  # 0. no change
                (S(6, 2), 1, 2),  # 1. no change
                (S(6, 3), 2, 1),  # 2. no change
                (S(6, 3), 1, 2),  # 2. no change
            ],
        ]

    def _connect_vertices_manual_kite(self):
        # manually define the connections for the "kite", simplest non-convex polygon
        # vertex #2 is non-convex
        def S(n, k):
            return (n+k) % self.Nsides
        self.segments = [
            [
                # blue      no change
                (0, 0, 0),        # convex
                (S(0, 1), 0, 1),  # convex
                (S(0, 2), 1, 2),  # concave
                (S(0, 3), 2, 1),  # convex
                (S(0, 3), 1, 2),  # convex
            ],
            [
                # yellow    no change
                (1, 0, 0),        # convex
                (S(1, 1), 0, 1),  # concave
                (S(1, 2), 1, 2),  # convex
                (S(1, 3), 2, 1),  # convex
                (S(1, 3), 1, 2),  # convex
            ],
            [
                # green. starts at concave, needs to adapt
                (2, 0, 1),    # concave
                (2, 1, 0),    # (2, 0, 0) -> (2, 0, 1), (2, 1, 0)  concave
                (S(2, 1), 0, 1),  # convex
                (S(2, 2), 1, 2),  # convex
                (S(2, 3), 2, 1),  # convex
                (S(2, 3), 1, 2),  # convex
            ],
            [
                # red. ends at concave, needs to adapt
                (3, 0, 0),        # convex
                (S(3, 1), 0, 1),  # convex
                (S(3, 2), 1, 2),  # convex
                (S(3, 3), 2, 2),  # (2, 2, 1) -> (2, 2, 2)      concave
                #(S(3, 3), 1, 2)  # (2, 1, 2) -> {}
            ],
        ]


def random_convex_polygon(sides=3, debias=True):
    angles = np.arange(0, 1, 1./sides)
    noise = (np.random.random(sides)-0.5)/sides
    angles = (angles + noise) * 2 * np.pi
    # angles = np.random.random(sides) * 2 * np.pi
    angles.sort()
    v = np.exp(1j*angles)
    if debias:
        v -= np.mean(v)
    return v


def main():
    kite = np.array([1, -0.7-0.7j, -.2, -0.7+0.7j])
    pentathing = np.array([.7+.7j, -.7+.7j, -.3, -.7-.7j, .7-.7j])
    complicated_shape = 2 * np.array([.3, .7+.7j, -.7+.9j, -.4+.3j, -.4-.3j, -.7-.9j, .7-.7j])
    # p = Penrose(sides=5, segments=4, segment_ratio=0.1)

    if len(sys.argv) < 2:
        mode = '3'
    else:
        mode = sys.argv[1]
    if mode == 'random' and len(sys.argv) > 2:
        sides = int(sys.argv[2])
    else:
        sides = 3

    if mode == '3':
        p = Penrose(sides=3, segments=2)
    elif mode == 'random':
        p = Penrose(polygon=random_convex_polygon(sides=sides), segments=2)
    elif mode == 'pentathing':
        p = Penrose(polygon=pentathing, segments=2)
    elif mode == 'complicated':
        p = Penrose(polygon=complicated_shape, segments=2)
    elif mode == 'kite':
        p = Penrose(polygon=kite, segments=2)

    print('base polygon:')
    for n, pt in enumerate(p.polygon):
        print('  %d: %s' % (n, pt))

    p._define_lines()
    p._define_vertices()

    if mode == 'kite':
        p._connect_vertices_manual_kite()
    elif mode == 'complicated':
        # p._connect_vertices_manual_complicated()
        p._connect_vertices()
    else:
        p._connect_vertices()

    data = []
    # data.extend(p.get_polygon_traces(mode='lines', line=dict(color='blue')))
    # data.extend(p.get_vertex_traces(mode='markers', line=dict(color='black')))
    data.extend(p.get_segment_traces(mode='lines'))
    # data.extend(p.get_vertex_trace(mode='markers', line=dict(color='black')))
    # data.extend(p.get_line_traces(mode='lines', line=dict(color='gray', dash='dash', width=1)))

    layout = go.Layout(
        title='foobar',
        xaxis=dict(title='x', zeroline=False),
        yaxis=dict(title='y', scaleanchor='x', zeroline=False),
        # showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='penrose-lines.html', **plot_args)


main()
