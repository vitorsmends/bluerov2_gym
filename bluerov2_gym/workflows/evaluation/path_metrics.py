"""Classical geometric path distance and sampled time within a tolerance."""
import numpy as np
from .trajectories import create_trajectory


def distance_to_polyline(points, vertices):
    points=np.asarray(points,dtype=float);vertices=np.asarray(vertices,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or vertices.ndim!=2 or vertices.shape[1]!=3:
        raise ValueError('Expected 3-D position arrays.')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(vertices)) or len(vertices)<2:
        raise ValueError('Finite positions and at least one reference segment are required.')
    starts=vertices[:-1];edges=np.diff(vertices,axis=0);lengths=np.sum(edges*edges,axis=1)
    keep=lengths>0;starts,edges,lengths=starts[keep],edges[keep],lengths[keep]
    if not len(lengths):raise ValueError('Degenerate reference path.')
    distances=[]
    for first in range(0,len(points),64):
        delta=points[first:first+64,None,:]-starts[None,:,:]
        alpha=np.clip(np.einsum('ijk,jk->ij',delta,edges)/lengths,0,1)
        residual=delta-alpha[:,:,None]*edges[None,:,:]
        distances.extend(np.sqrt(np.min(np.einsum('ijk,ijk->ij',residual,residual),axis=1)))
    return np.asarray(distances)


def geometric_path_distance(points,name,params):
    points=np.asarray(points,dtype=float);reference=create_trajectory(name,params)
    if not np.all(np.isfinite(points)):raise ValueError('Non-finite vehicle positions.')
    if name=='circle':
        return np.sqrt((np.hypot(points[:,0],points[:,1])-reference.radius)**2+
                       (points[:,2]-reference.depth)**2)
    if hasattr(reference,'points'):
        return distance_to_polyline(points,reference.points)
    if name=='figure_eight_depth' and params.get('depth_amp',.25)==0:
        period=float(params['period'])
        vertices=np.asarray([reference.position(t) for t in np.linspace(0,period,4097)])
        # Constant-depth sine figure eight: maximum chord approximation error
        # <= sqrt(5)*a*(2*pi/4096)^2/8, under 0.7 micrometres for a=1 m.
        return distance_to_polyline(points,vertices)
    raise ValueError('Metric implemented for constant-depth eight, circle, square and star.')


def time_within_tolerance(errors,tolerance=.2):
    errors=np.asarray(errors,dtype=float)
    if not len(errors) or tolerance<=0 or not np.all(np.isfinite(errors)):
        raise ValueError('Finite errors and a positive tolerance are required.')
    return float(100*np.mean(errors<=tolerance))
