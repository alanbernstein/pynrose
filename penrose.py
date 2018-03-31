#!/usr/bin/env python
import sys
import time
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from plotlytools import category10

from line import SimpleLine

"""
This is an ASCII drawing of a Penrose triangle, or impossible triangle.
In the terminology of the Penrose class below, it has
sides = 3
segments = 2

It is hard to use this diagram to explain the meaning of the "segment_ratio" variable...

       _
      / /\
     / /  \
    / / /\ \
   / / /\ \ \
  / / /  \ \ \
 / /_/____\ \ \
/__________\ \ \
\_____________\/

there were two other short-lived approaches:
turtle:
define all turn angles and lengths, directly, in terms of a handful of base
variables. then just run those through a "turtle draw" controller.
this got unwieldy quickly.

3d:
draw a polyhedron in 3d space, then render it on a depth buffer with some
clever non-euclidean occlusion rules. i still like this idea, but i'm
not really sure if it's feasible. i didn't get too far before coming up
with the Right Way for 2d drawings...

instead of these, just draw a grid of lines emanating from the base polygon
then find the appropriate intersections and connect them.
bonus: works with any polygon.
"""

plot_args = dict(
    auto_open=True,
    # output_type='div',
    show_link=False,
    config={'displayModeBar': False}
)


def get_polygon_convexity_ccw(pts):
    # input:  Nx2 array
    # output: [is_convex, is_convex, ...] x N
    # only works for ccw polygons (cross(v0, v1) < 0)
    # but same approach will work for both if just determine whether ccw or not
    i0 = np.arange(len(pts))
    iprev = (i0 - 1) % len(pts)
    inext = (i0 + 1) % len(pts)
    v0 = pts[inext] - pts[i0]
    v1 = pts[i0] - pts[iprev]
    return np.cross(v0, v1) < 0


def get_polygon_convexity(pts):
    # work for both ccw and cw
    # TODO
    pass


class Penrose(object):
    """
    build a "penrose polyhegon" out of any simple polygon
    assumes polygon is traversed CCW
    """
    line_class = SimpleLine  # SimpleLine for self-contained, LineWrapper to use sympy's Line

    def __init__(self, polygon=None, sides=3, segments=2, segment_size=None):
        # TODO: handle arbitrary "phase"
        # TODO: handle "pass-through vertices" for more complex shapes, e.g. curved sides
        # TODO: handle different "skip values" of connecting segments, instead of 1 by default
        # TODO: handle genus 2+ shapes? E.G the letter A instead of a triangle
        # IDEA: compute intersections lazily (fast enough already)
        # DONE: variable spacing for multiple segments (cylindrical appearance)
        # DONE: make segment width an explicit parameter, instead of segment_ratio
        # DONE: handle nonconvex polygons
        # DONE: scale segments equally based on line-origin distance
        # DONE: use lightweight Line class so intersections are faster
        t0 = time.time()
        self.Nsides = sides
        self.Nsegments = segments
        if polygon is None:
            # use regular Nsides-sided polygon
            pts = regular_polygon(self.Nsides)
            self.polygon = [(p.real, p.imag) for p in pts]
        else:
            if type(polygon) == np.ndarray and polygon.dtype == 'complex128':
                # convert complex to tuples
                self.polygon = [(p.real, p.imag) for p in polygon]
            else:
                self.polygon = polygon
            self.Nsides = len(self.polygon)
        self.segment_size = segment_size or [1./(self.Nsides)] * self.Nsegments
        self.convexity = get_polygon_convexity_ccw(np.array(self.polygon))

        self._define_lines()      # define Nsegments additional lines emanating from base polygon
        self._define_vertices()   # define lattice of (Nsegments+1)^2 points at each vertex
        self._connect_vertices()  # the hard part: connect the points in the right sequence
        t1 = time.time()
        print('geometry calculated in %f sec' % (t1-t0))

    def S(self, n, k):
        # cyclic Successor convenience function
        return (n+k) % self.Nsides

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
        # TODO redo this more numpyly

        v = self.polygon
        self.lines = {}
        mm = np.cumsum([0] + self.segment_size)

        # print('define_lines')
        for n in range(self.Nsides):
            n1 = self.S(n, 1)
            dx, dy = v[n1][0] - v[n][0], v[n1][1] - v[n][1]
            mag = np.hypot(dx, dy)
            perp = [dy/mag, -dx/mag]  # this is where CW vs CCW matters
            for m in range(self.Nsegments+1):
                # print('  side=%d, segment=%d, k=%f: vertices %d,%d' % (n, m, k, n, n1))

                self.lines[(n, m)] = self.line_class(
                    p1=(v[n][0] + mm[m]*perp[0], v[n][1] + mm[m]*perp[1]),
                    p2=(v[n1][0] + mm[m]*perp[0], v[n1][1] + mm[m]*perp[1]),
                )

    def _get_vertex(self, k):
        """
        function to support lazy computation of vertex locations
        use input key k to compute (and store) only those vertices that are required
        calculation is more than fast enough without this though
        """
        if k in self.vertices:
            return self.vertices[k]

        # k = (base-vertex, dist1, dist2) = (n1, m1, m2)
        # n2 = self.S(n1, 1)
        # intersection of the lines (TODO verify)
        # l1 = self.lines[(n1, m1)]
        # l2 = self.lines[(n2, m2)]
        pass

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
        # print('define_vertices')
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
                    # print('  (%d, %d, %d): (%.4g, %.4g)' % (n, m1, m2, li[0], li[1]))
                    self.vertices[ki] = li

    def get_polygon_traces(self, **kwargs):
        # get_*_traces functions are all compatible with plotly:
        # returns a list of dicts that are usable as go.Scatter objects
        # kwargs can be any set of go.Scatter inputs (debug)
        #
        # returns a trace of the generating polygon
        x = [p[0] for p in self.polygon + [self.polygon[0]]]
        y = [p[1] for p in self.polygon + [self.polygon[0]]]
        trace = dict(x=x, y=y)
        trace.update(**kwargs)
        return [trace]

    def get_line_traces(self, **kwargs):
        # returns a list of traces of the generating lines (debug), by connecting vertices
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
        # returns a list of named traces of the generating vertices (debug)
        data = []
        for k, (x, y) in self.vertices.items():
            trace = dict(x=[x], y=[y], name='%s' % (k,))
            trace.update(**kwargs)
            data.append(trace)
        data.sort(key=lambda x: x['name'])
        return data

    def get_vertex_trace(self, **kwargs):
        # returns a single trace of the generating vertices (debug)
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
            # trace = dict(
            #     x=[self.get_vertex(k)[0] for k in part],
            #     y=[self.get_vertex(k)[1] for k in part],
            # )
            trace.update(**kwargs)
            data.append(trace)
        return data

    def _connect_vertices(self):
        self._connect_vertices_nonconvex()

    def _connect_vertices_nonconvex(self):
        # connects vertices for any Nsides, Nsegments. convex or nonconvex.
        # given what we learned from manually connected segments for the
        # "complicated" shape, it seems clear how to do this now:
        # - corners behave normally when a segment passes through concave vertices
        # - corners behave differently when a segment starts or ends at a concave vertex
        # - just need to keep track of which vertex the segment is currently at,
        #   and adjust the corner that is added to the segment accordingly

        # print('connect_vertices_nonconvex')

        self.segments = []
        for i in range(self.Nsides):
            # first corner depends on convexity
            # TODO here is an opportunity to switch convexity bool to
            # vertex_type enum: {convex, concave, pass-through}
            # pass-through means don't treat it like one of the
            # penrose corner vertices, but just pass the line through it
            # so you can create curved segments
            if self.convexity[i]:
                part = [(i, 0, 0)]
            else:
                part = [(i, 0, 1), (i, 1, 0)]

            # middle corners are always the same
            for j in range(self.Nsegments):
                part.append((self.S(i, j+1), j, j+1))

            # last corner depends on convexity
            end = self.S(i, j+2)
            if self.convexity[end]:
                part.append((end, j+1, j))
                part.append((end, j, j+1))
            else:
                part.append((end, j+1, j+1))

            self.segments.append(part)


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


def regular_polygon(sides=3):
    return np.exp(np.arange(sides) * 2j * np.pi / sides)


def regular_star(p=5, q=2, r=1.0):
    # see e.g. http://mathworld.wolfram.com/StarPolygon.html
    # in that context,
    # p = number of points
    # q = "density" - how many vertices are skipped. 1 < q < p/2
    # -
    # r = distance from origin to star point
    if p < 5 or not 1 < q < p/2.0:
        raise ValueError

    r_mid = (np.cos(np.pi/p) - np.tan(np.pi * (q-1.0) / p) * np.sin(np.pi/p))
    n = np.arange(0., 2*p)
    angles = n / p * np.pi
    mags = r_mid + (1 - r_mid) * (n % 2)
    return r * mags * np.exp(1j*angles)


def random_simple_polygon(sides=6, debias=True):
    # simple, as in, non-self-intersecting. possibly non-convex
    # http://jeffe.cs.illinois.edu/open/randompoly.html
    # trivial by brute force: there are N! orderings of N points,
    # so just check each ordering for self-intersecting sides
    # each of the N sides has to be compared with N-3 other sides,
    # for N(N-3)/2 total intersection checks (0, 2, 5, 9, 14, ...)
    # note that we only need to compute N(N-3)/2 intersection checks,
    # then use those same checks for every ordering
    # n=5: {02, 03, 13, 14, 24}
    # n=6: {02, 03, 04, 13, 14, 15, 24, 25, 35}
    import itertools
    pts = np.random.random(sides) + 1j * np.random.random(sides)

    # compute all intersection points
    intersection_pts = {}
    for n1 in range(sides-1):
        for n2 in range(n1+1, sides):
            # line(x) connects points x and x+1
            # intersection(x, y) is where line x intersects line y
            l1 = SimpleLine(pts[n1], pts[n1+1])
            l2 = SimpleLine(pts[n2], pts[n2+1])
            intersection_pts[(n1, n2)] = l1.intersect(l2)

    def between((p1, p2), pi):
        # TODO finish
        return True

    for edges in itertools.permutations(range(sides)):
        # check intersections
        for n1 in range(sides):
            # upper = sides if n1 else sides-1
            upper = sides - (n1 == 0)
            for n2 in range(n1+2, upper):
                # TODO finish
                bounds1 = (intersection_pts[(n1-1, n1)], intersection_pts[(n1, n1+1)])
                bounds2 = (intersection_pts[(n2-1, n2)], intersection_pts[(n2, n2+1)])
                this = intersection_pts[(n1, n2)]
                if between(bounds1, this) or between(bounds2, this):
                    continue
                print(n1, n2)
        return pts[edges]


def benchmark():
    pts = regular_polygon(sides=7)
    t0 = time.time()
    iters = 0
    while True:
        t1 = time.time()
        p = Penrose(polygon=pts, segments=4)
        traces = p.get_segment_traces()
        t2 = time.time()

        if int(t2) != int(t1):
            print('%f FPS' % (iters/(t2-t0)))

        pts += 0.01
        iters += 1


def main():
    kite = np.array([1, -0.7+0.7j, -.2, -0.7-0.7j])
    pentathing = np.array([.7+.7j, -.7+.7j, -.3, -.7-.7j, .7-.7j])
    complicated_shape = 2 * np.array([.3, .7+.7j, -.7+.9j, -.4+.3j, -.4-.3j, -.7-.9j, .7-.7j])
    u_shape = np.array([1+1j, 1-1j, -1-1j, -1-2j, 2-2j, 2+2j, -1+2j, -1+1j])

    if len(sys.argv) < 2:
        mode = '3'
    else:
        mode = sys.argv[1]

    if mode == 'random' and len(sys.argv) > 2:
        sides = int(sys.argv[2])
    elif mode == 'star' and len(sys.argv) > 2:
        sides = int(sys.argv[2])
        q = 2
        if len(sys.argv) > 3:
            q = int(sys.argv[3])
    else:
        sides = 3

    segments = 2

    if mode in '3456789':
        p = Penrose(sides=int(mode), segments=segments)
    elif mode == 'random':
        p = Penrose(polygon=random_convex_polygon(sides=sides), segments=segments)
    elif mode == 'star':
        p = Penrose(polygon=regular_star(p=sides, q=q), segments=segments)
    elif mode == 'pentathing':
        p = Penrose(polygon=pentathing, segments=segments)
    elif mode == 'complicated':
        p = Penrose(polygon=complicated_shape, segments=segments)
    elif mode == 'kite':
        p = Penrose(polygon=kite, segments=segments)
    elif mode == 'u':
        p = Penrose(polygon=u_shape, segments=segments)
    elif mode == 'test1':
        p = Penrose(sides=8, segments=6, segment_size=[0.1, 0.2, 0.3, 0.3, 0.2, 0.1])

    print('base polygon:')
    for n, pt in enumerate(p.polygon):
        print('  %d: %s' % (n, pt))

    data = []
    # data.extend(p.get_polygon_traces(mode='lines', line=dict(color='blue')))
    # data.extend(p.get_vertex_traces(mode='markers', line=dict(color='black')))
    # data.extend(p.get_segment_traces(mode='lines'))
    data.extend(p.get_segment_traces(mode='lines', line=dict(color='black')))
    # data.extend(p.get_vertex_trace(mode='markers', line=dict(color='black')))
    # data.extend(p.get_line_traces(mode='lines', line=dict(color='gray', dash='dash', width=1)))

    layout = go.Layout(
        title='impossible polyhegon',
        xaxis=dict(title='x', zeroline=False),
        yaxis=dict(title='y', scaleanchor='x', zeroline=False),
        # showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='penrose.html', **plot_args)


main()
# benchmark()
# random_simple_polygon(6)
