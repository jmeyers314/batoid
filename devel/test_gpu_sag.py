import numpy as np
import batoid

x = np.linspace(-0.1, 0.1, 1000)
x, y = np.meshgrid(x, x)

for surf in [
    batoid.Paraboloid(1.0),
    batoid.Plane(),
    batoid.Tilted(0.01, -0.02),
    batoid.Sphere(-0.3),
    batoid.Quadric(-0.2, 0.01),
    batoid.Asphere(-0.2, 0.01, [1e-3, 1e-5, -2e-7]),
    batoid.Sum([batoid.Paraboloid(1.0), batoid.Sphere(0.01)])
]:
    print(surf)
    out_cpu = surf.sag(x, y)
    out_gpu = surf._sagGPU(x, y)
    np.testing.assert_allclose(out_cpu, out_gpu, atol=1e-15, rtol=1e-15)
