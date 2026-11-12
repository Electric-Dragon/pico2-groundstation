import matplotlib.pyplot as plt

from polytope3d.synthetic import render_depth

depth = render_depth(
    240, 320, 300.0, 300.0, 160.0, 120.0, 1.0, [((-0.5, 0.5, 2.0), (0.5, 1.0, 3.0))]
)
plt.imshow(depth)
plt.colorbar(label="depth (m)")
plt.savefig("scratch/depth.png")
