# render.py <prefix> <mesh.obj> <out.png> [cx cy cz r]: the input mesh, the open pose and the closed pose (with its sphere).
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
pre, obj, out = sys.argv[1:4]
sph = list(map(float, sys.argv[4:8])) if len(sys.argv) >= 8 else None
def load(path):
    with open(path, 'rb') as f:
        hdr = f.readline().split(); n = int(hdr[0]); b = list(map(float, hdr[1:]))
        v = np.frombuffer(f.read(), dtype=np.float32).reshape(n, n, n)  # z, y, x
    v = v.transpose(2, 1, 0)
    sp = [(b[3+i]-b[i])/(n-1) for i in range(3)]
    if v.min() > 0: return np.zeros((0,3)), np.zeros((0,3),int)
    vert, face, _, _ = measure.marching_cubes(v, 0.0, spacing=sp)
    return vert + np.array(b[:3]), face
def obj_load(p):
    V=[];F=[]
    for l in open(p):
        s=l.split()
        if not s: continue
        if s[0]=='v': V.append(list(map(float,s[1:4])))
        elif s[0]=='f': F.append([int(x.split('/')[0])-1 for x in s[1:4]])
    return np.array(V), np.array(F)
def draw(ax, v, f, col, title, sphere=None):
    # y is up in the editor: plot (x, z, y)
    if len(f):
        P = v[:, [0, 2, 1]]
        tri = P[f]
        n = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]); n /= np.linalg.norm(n,axis=1,keepdims=True)+1e-9
        shade = 0.35 + 0.65*np.clip(np.abs(n @ np.array([0.4,-0.5,0.75])),0,1)
        c = np.array(col)[None,:]*shade[:,None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(c,0,1), edgecolor='none'))
    if sphere:
        cx,cy,cz,r = sphere
        u,w = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
        ax.plot_wireframe(cx+r*np.cos(u)*np.sin(w), cz+r*np.sin(u)*np.sin(w), cy+r*np.cos(w), color='k', linewidth=0.3, alpha=0.5)
    ax.set_xlim(-45,45); ax.set_ylim(-45,45); ax.set_zlim(-25,65); ax.set_box_aspect((1,1,1)); ax.set_title(title); ax.view_init(22,-60)
    ax.set_axis_off()
fig = plt.figure(figsize=(15,5.2))
mv, mf = obj_load(obj)
draw(fig.add_subplot(131, projection='3d'), mv, mf, (0.75,0.75,0.8), 'input mesh')
v,f = load(pre+'_open.raw'); draw(fig.add_subplot(132, projection='3d'), v, f, (0.95,0.65,0.35), 'design (open)')
v,f = load(pre+'_closed.raw'); draw(fig.add_subplot(133, projection='3d'), v, f, (0.4,0.7,0.95), 'closed', sph)
plt.tight_layout(); plt.savefig(out, dpi=80)
