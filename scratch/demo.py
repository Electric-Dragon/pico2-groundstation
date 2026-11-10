"""End-to-end demo on a synthetic scene: depth -> cloud -> floor removal -> clusters -> polytopes."""
import matplotlib.pyplot as plt
import numpy as np

from polytope3d.clustering import euclidean_clusters
from polytope3d.polytopes import aabb, contains, convex_hull, volume_of_aabb
from polytope3d.preprocessing import remove_plane_ransac
from polytope3d.reconstruction import depth_to_points
from polytope3d.synthetic import render_depth

CAM = dict(fx=300.0, fy=300.0, cx=160.0, cy=120.0)
BOXES = [((-0.5, 0.5, 2.0), (0.5, 1.0, 3.0)), ((1.0, 0.6, 3.5), (1.6, 1.0, 4.0))]

depth = render_depth(240, 320, **CAM, cam_height=1.0, boxes=BOXES)
points = depth_to_points(depth, **CAM)
points = points[points[:, 2] < 8.0]
rest, floor, (normal, d) = remove_plane_ransac(points)
clusters = euclidean_clusters(rest, eps=0.1)

print(f"points: {len(points)}  floor: {len(floor)}  obstacle points: {len(rest)}")
print(f"floor normal: {np.round(normal, 3)}  clusters found: {len(clusters)}")
for i, c in enumerate(clusters):
    A1, b1 = aabb(c)
    A2, b2, verts = convex_hull(c)
    ok = contains(A1, b1, c).all() and contains(A2, b2, c).all()
    print(f"cluster {i}: {len(c)} pts | AABB planes {len(b1)} vol {volume_of_aabb(A1, b1):.3f} m3"
          f" | hull planes {len(b2)} | all points contained: {ok}")


def draw_aabb(ax, pts, color):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                p, q = corners[i], corners[j]
                ax.plot([p[0], q[0]], [p[2], q[2]], [-p[1], -q[1]], color=color)


fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(depth, vmax=8)
fig.colorbar(im, ax=ax1, label="depth (m)")
ax1.set_title("Synthetic depth image")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(floor[::10, 0], floor[::10, 2], -floor[::10, 1], s=1, c="lightgray")
for c, color in zip(clusters, ["tab:red", "tab:blue", "tab:green"]):
    ax2.scatter(c[:, 0], c[:, 2], -c[:, 1], s=2, c=color)
    draw_aabb(ax2, c, color)
ax2.set_xlabel("X right (m)")
ax2.set_ylabel("Z forward (m)")
ax2.set_zlabel("height (-Y, m)")
ax2.set_title("Floor (grey), clusters, AABBs")
plt.tight_layout()
plt.savefig("scratch/demo.png", dpi=150)