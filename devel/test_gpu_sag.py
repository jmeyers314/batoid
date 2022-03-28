import numpy as np
import batoid

para = batoid.Paraboloid(1.0)
plane = batoid.Plane()
tilted = batoid.Tilted(0.01, -0.02)

x = np.linspace(-0.1, 0.1, 1000)
x, y = np.meshgrid(x, x)

for surf in [
    batoid.Paraboloid(1.0),
    batoid.Plane(),
    batoid.Tilted(0.01, -0.02),
    batoid.Sphere(-0.3),
    batoid.Quadric(-0.2, 0.01)
]:
    out_cpu = surf.sag(x, y)
    out_gpu = surf._sagGPU(x, y)
    np.testing.assert_allclose(out_cpu, out_gpu)
