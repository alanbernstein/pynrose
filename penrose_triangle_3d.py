import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go

plot_args = dict(
    auto_open=False,
    # output_type='div',
    show_link=False,
    config={'displayModeBar': False}
)


def main():
    x, y, z = torus_vertices()
    edges = torus_edges_standard()
    edges = torus_edges_mobius()

    traces = []
    for e in edges:
        traces.append(go.Scatter(
            x=[x[e[0][0], e[0][1]], x[e[1][0], e[1][1]]],
            y=[y[e[0][0], e[0][1]], y[e[1][0], e[1][1]]],
            mode='lines',
            line=dict(color='black'),
        ))

    layout = go.Layout(
        title='penrose experiments',
        xaxis=dict(title='x'),
        yaxis=dict(title='y', scaleanchor='x'),
        showlegend=False,
    )

    fig = go.Figure(data=traces, layout=layout)
    plot(fig, filename='penrose.html', **plot_args)


def torus_edges_mobius(n1=4, n2=4):
    edges = []
    for u in range(n1):
        for v in range(n2):
            edges.append(((u, v), (u, (v+1) % n2)))
            edges.append(((u, v), ((u+1) % n1, v)))

    return edges


def torus_edges_standard(n1=4, n2=4):
    """
    0,0 - o-o-o-o - 3,0
          | | | |
          o-o-o-o
          | | | |
    0,2-  o-o-o-o - 3,2
    """
    edges = []
    for u in range(n1):
        for v in range(n2):
            edges.append(((u, v), (u, (v+1) % n2)))
            edges.append(((u, v), ((u+1) % n1, v)))

    return edges


def torus_vertices(r1=1, r2=0.25, n1=4, n2=4, p1=np.pi/4, p2=np.pi/2):
    # generate a polygonal torus
    # inputs:
    # - main radius r1
    # - cross-section radius r2
    # - number of main segments
    # - number of cross-section segments
    # outputs:
    # - x, y, z in u-v parametric meshgrid format

    u, v = np.meshgrid(np.arange(n1), np.arange(n2))

    x = (r1 + r2 * np.cos(2*np.pi/n2*v + p2)) * np.cos(2*np.pi/n1*u + p1)
    y = (r1 + r2 * np.cos(2*np.pi/n2*v + p2)) * np.sin(2*np.pi/n1*u + p1)
    z = r2 * np.sin(2*np.pi/n2*v + p2)

    return x, y, z


main()
