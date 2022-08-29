import numpy as np

############### loading file ###################
def get_pred(path):
    data = np.load(path, allow_pickle=True).item()
    points = data["input"]
    pred_affs = data["pred_affs"]
    fixed_affs_local = data["fixed_affs_local"]
    fixed_affs_global = data["fixed_affs_global"]
    print(points.shape)
    return points.squeeze(0), pred_affs.squeeze(0), fixed_affs_local.squeeze(0), fixed_affs_global.squeeze(0)

pts, pre,fixl,fixg = get_pred("spatial_pattern_30.npy")

############### spatial pattern statistics  ###################
num_sp = 6

global_patten = []
local_patten = []
mask = np.linalg.norm(pts,axis=1)<1.0
for i in range(num_sp):
    ps1 = fixg[:, 0 + 3 * i:3 + 3 * i]
    ps2 = pre[:, 0 + 3 * i:3 + 3 * i]
    d = (np.linalg.norm(ps1-ps2, axis=1)*mask).sum()/mask.sum()
    global_patten += [d]

    ps1 = fixl[:, 0 + 3 * i:3 + 3 * i]
    ps2 = 0 #pre[:, 0 + 3 * i:3 + 3 * i] # need to load uniform spatial pattern...
    d = (np.linalg.norm(ps1-ps2, axis=1)*mask).sum()/mask.sum()
    local_patten += [d]

import matplotlib.pyplot as plt
labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, global_patten, width, color='plum', label='Non-uniform Spatial Pattern')
rects2 = ax.bar(x + width/2, local_patten, width, color='lightblue', label='Uniform Spatial Pattern')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Offsets = Prediction - Initialization')
#ax.set_title('Spatial Pattern Points')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format("%.3f" %height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()


############### spatial pattern visualization  ###################
import trimesh
mesh = trimesh.load("lamp.obj")
v = mesh.vertices
v = np.array(v)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)

valstep = 20
def update_slider(val):
    ax.cla()
    # Hide grid lines
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(10,30)
    X, Y, Z = 0, 2, 1  #switch the axis for better visualization
    ax.scatter(v[::valstep, X], v[::valstep, Y], v[::valstep, Z], c='grey', s=1, marker='o')

    vid = int(val)  # np.round(val).astype(np.int)
    ind = np.argmin(np.linalg.norm(pts - v[vid], axis=1))

    num_point_to_show = 1
    id = [i for i in range(ind,ind+num_point_to_show,1)]
    psize = 50

    x = pts[id, X]
    y = pts[id, Y]
    z = pts[id, Z]
    ax.scatter(x, y, z, c='black', s=psize, marker='+') #'#000000'
    for i in range(num_sp):
        x = fixg[id, X+3*i]
        y = fixg[id, Y+3*i]
        z = fixg[id, Z+3*i]
        ax.scatter(x, y, z, c='magenta', s=psize, marker='+') #'#FF00FF'
    for i in range(num_sp):
        x = fixl[id, X+3*i]
        y = fixl[id, Y+3*i]
        z = fixl[id, Z+3*i]
        ax.scatter(x, y, z, c='deepskyblue', s=psize, marker='+')  #'#00BFFF'
    for i in range(num_sp):
        x = pre[id, X+3*i]
        y = pre[id, Y+3*i]
        z = pre[id, Z+3*i]
        ax.scatter(x, y, z, c = '', edgecolors='green', s=psize, marker='o') #'#008000'
    if 0: #show predicted pattern from uniform initialization
        for i in range(0,num_sp):
            x = pre2[id, X+3*i]
            y = pre2[id, Y+3*i]
            z = pre2[id, Z+3*i]
            ax.scatter(x, y, z, c = '', edgecolors='orange', s=psize, marker='o') #'#FFA500'

    fig.canvas.draw_idle()
    rang = 0.67
    ax.set_xlim(-rang, rang)
    ax.set_ylim(-rang, rang)
    ax.set_zlim(-rang, rang)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

from matplotlib.widgets import Slider,RadioButtons
axcolor = 'lightgoldenrodyellow'  # slider color
om = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)  # slider position
som = Slider(om, r'Point index', 0, v.shape[0]-1, valinit=0,valstep=valstep)
som.on_changed(update_slider)
update_slider(0)
plt.show()


#python vis.py
