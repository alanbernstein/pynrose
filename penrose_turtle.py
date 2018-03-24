#!/usr/bin/env python
from plotly.offline import plot
import plotly.graph_objs as go
import numpy as np


def evaluate_turtles(l_vec, dt_vec, t0):
    z = np.cumsum(l_vec * np.exp(1j*(np.cumsum(dt_vec) + t0)))
    return z


def rotate(z, a):
    return z * np.exp(1j*a)


def get_turtles_3_2():
    # generate distance + turn-angle sequence that can be passed
    # to evaluate_turtles to create a conventional penrose triangle

    N = 3

    # this ratio is somewhat arbitrary, 3:1 looks decent
    d1 = 3  # side-length of interior polygon
    d2 = 1  # slant-length of side segment

    # hardcoded radius+turn-angles
    l_vec = np.array([d1*np.sqrt(3)/3, d1+d2, d1+d2*3, d1+d2*4, d2])
    dt_vec = np.array([0, 150, 120, 120, 60]) * np.pi/180
    # l_vec = np.array([d1*np.sqrt(3)/3, d1+d2, d1+d2*3, d1+d2*4, d2])
    # dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a]) * np.pi/180

    return l_vec, dt_vec


def get_turtles(sides=3, segments=2):
    # TODO
    # - support arbitrary "sides"
    # - support arbitrary "segments"
    # - support "phase" - the rotation of the segments

    a = 180 - 360/sides      # base angle

    # this ratio is somewhat arbitrary, 3:1 looks decent
    d1 = 3  # side-length of interior polygon
    d2 = 1  # slant-length of side segment

    if sides == 3:
        k = np.sqrt(3)/3
        if segments == 2:
            A, B, C = 3, 4, 1
            l_vec = np.array([d1*k, d1+d2, d1+d2*A, d1+d2*B, d2*C])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a]) * np.pi/180
        if segments == 3:
            A, B, C, D = 3, 6, 7, 1
            l_vec = np.array([d1*k, d1+d2, d1+d2*A, d1+d2*B, d1+d2*C, d2*D])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, 180-a, a]) * np.pi/180
        if segments == 4:
            A, B, C, D, E = 3, 6, 9, 10, 1
            l_vec = np.array([d1*k, d1+d2, d1+d2*A, d1+d2*B, d1+d2*C, d1+d2*D, d2*E])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, 180-a, 180-a, a]) * np.pi/180

        # A, B = 3, 1
        # l_vec = np.array([d1*k, d1+d2, d1+d2*A*1, d1+d2*A*2, d1+d2*(A*3), d1+d2*(A*3+1), d2*B])
        # dt_vec = np.array([0, 180-a/2, 180-a, 180-a, 180-a, 180-a, a]) * np.pi/180
    elif sides == 4:
        k = np.sqrt(2)/2
        if segments == 2:
            A, B, C = 2, 2, 2*k
            l_vec = np.array([d1*k, d1+d2, d1+d2*A, d1+d2*B, d2*C])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a/2]) * np.pi/180
        if segments == 3:
            A, B, C = 2, 2, 2*k
            l_vec = np.array([d1*k, d1+d2, d1+d2*2, d1+d2*4, d1+d2*4, d2*C])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, 180-a, a/2]) * np.pi/180
        if segments == 4:
            A, B, C = 2, 2, 2*k
            l_vec = np.array([d1*k, d1+d2, d1+d2*2, d1+d2*4, d1+d2*6, d1+d2*6, d2*C])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, 180-a, 180-a, a/2]) * np.pi/180
    elif sides == 5:
        k = (np.sqrt(5)+1)/4
        A, B, C = 1.2, 1.5, k*2
        A = 2 * (1 - np.sin(18 * np.pi/180))
        # l_vec = np.array([d1*k, d1+d2, d1+d2*1.43805025, d1+.728, d2*1.857])
        # dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a/3]) * np.pi/180
        A = 2*(1-np.sin(np.pi/(5*2)))
        l_vec = np.array([d1*k, d1+d2, d1+d2*A, d1+.728, d2*1.857])
        dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a/3]) * np.pi/180
    elif sides == 6:
        k = 1
        if segments == 2:
            A, B, C = 1, 1, k
            # l_vec = np.array([d1*k, d1+d2, d1+d2*A, d1+d2*B, d2*C])
            # dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a/2]) * np.pi/180
            l_vec = np.array([d1*k, d1+d2, d1+d2*A, d1+d2*0, d2*np.sqrt(3)])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a/4]) * np.pi/180
        if segments == 3:
            l_vec = np.array([d1*k, d1+d2, d1+d2*1, d1+d2*2, d1+d2, d2*np.sqrt(3)])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, 180-a, a/4]) * np.pi/180
        if segments == 4:
            l_vec = np.array([d1*k, d1+d2, d1+d2*1, d1+d2*2, d1+d2*3, d1+d2*2, d2*np.sqrt(3)])
            dt_vec = np.array([0, 180-a/2, 180-a, 180-a, 180-a, 180-a, a/4]) * np.pi/180


    # l_vec = np.array([d1*k, d1+d2, d1+d2*3, d1+d2*4, d2])
    # dt_vec = np.array([0, 180-a/2, 180-a, 180-a, a]) * np.pi/180

    return l_vec, dt_vec


def get_penrose_shape(sides=3, segments=2):
    # these should be defined in terms of faces and sides
    t0 = np.pi/2

    # l_vec, dt_vec = get_turtles_3_2()  # get turtle definitions
    l_vec, dt_vec = get_turtles(sides, segments)  # get turtle definitions
    z = evaluate_turtles(l_vec, dt_vec, t0)  # evaluate to trajectories
    zs = [z * np.exp(1j*n*2*np.pi/sides) for n in range(sides)]  # create N rotated copies
    pts = np.array([zs[2][0], zs[1][1], zs[0][2]])
    print(pts)
    print(pts.imag)
    print(np.diff(pts.imag))
    return zs


def plot_stuff(zs):
    data = [go.Scatter(x=z.real, y=z.imag) for z in zs]
    layout = dict(
        xaxis=dict(title='foo', zeroline=False, showgrid=False),
        yaxis=dict(scaleanchor='x', title='bar', zeroline=False, showgrid=False)
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='penrose.html')


zs = get_penrose_shape(sides=5, segments=2)
plot_stuff(zs)


############################
## deprecated

def rotate_real(x, y, a):
    return x * np.cos(a) - y*np.sin(a), x*np.sin(a) + y*np.cos(a)


def evaluate_turtles_real(l_vec, dt_vec, t0):
    # evaluate sequences of distances + turn-angles to generate
    # a trajectory
    x_vec, y_vec = [], []
    x, y = 0, 0
    t = t0
    for l, dt in zip(l_vec, dt_vec):
        t += dt
        x += l * np.cos(t)
        y += l * np.sin(t)
        x_vec.append(x)
        y_vec.append(y)

    x = np.array(x_vec)
    y = np.array(y_vec)
    return x, y


def get_penrose_shape_real(faces=3, sides=3):
    # return a list of N xy trajectories
    sides = 3

    # these should be defined in terms of faces and sides
    t0 = np.pi/2

    l_vec, dt_vec = get_turtles_3_2()
    x, y = evaluate_turtles_real(l_vec, dt_vec, t0)
    x1, y1 = rotate_real(x, y, 2*np.pi/3)
    x2, y2 = rotate_real(x, y, 4*np.pi/3)
    return [[x, y], [x1, y1], [x2, y2]]



def plot_stuff_real(xys):
    data = [go.Scatter(x=x, y=y) for x, y in xys]
    layout = dict(
        xaxis=dict(title='foo'),
        yaxis=dict(scaleanchor='x', title='bar')
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='penrose.html')
