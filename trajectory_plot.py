
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.collections as mcoll


def draw_lines(output,output_i,linestyle='-',alpha=1,darker=False,linewidth=2):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    loc = np.array(output[output_i,:,:,0:2])
    loc = np.transpose( loc, [1,2,0] )

    x = loc[:,0,:]
    y = loc[:,1,:]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    max_range = max( y_max-y_min, x_max-x_min )
    xmin = (x_min+x_max)/2-max_range/2-0.1
    xmax = (x_min+x_max)/2+max_range/2+0.1
    ymin = (y_min+y_max)/2-max_range/2-0.1
    ymax = (y_min+y_max)/2+max_range/2+0.1

    cmaps = [ 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds', 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds'  ]
    cmaps = [ matplotlib.cm.get_cmap(cmap, 512) for cmap in cmaps ]
    cmaps = [ ListedColormap(cmap(np.linspace(0., 0.8, 256))) for cmap in cmaps ]
    if darker:
        cmaps = [ ListedColormap(cmap(np.linspace(0.2, 0.8, 256))) for cmap in cmaps ]

    for i in range(loc.shape[-1]):
        lc = colorline(loc[:,0,i], loc[:,1,i], cmap=cmaps[i],linestyle=linestyle,alpha=alpha,linewidth=linewidth)
    return xmin, ymin, xmax, ymax

def colorline(
        x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
        linewidth=2, alpha=0.8, linestyle='-'):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)

    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, alpha=alpha, linestyle=linestyle)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

