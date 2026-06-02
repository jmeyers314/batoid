import batoid
import numpy as np

for f in "ugrizy":
    telescope = batoid.Optic.fromYaml(f"Rubin_v1000_{f}.yaml")
    camera = telescope["LSSTCamera"]
    cs = camera.coordSys.shiftLocal([0.0, 0.0, 0.1045])
    rx, ry, rz = cs.euler()
    print(f"------ {f} ------")
    print(camera.coordSys)
    print(camera.coordSys.shiftLocal([0.0, 0.0, 0.1045]))

    print(
f"""
    -
      type: Baffle
      name: CameraBody
      parent: LSSTCamera  # physically attached to camera; moves with it under rigid-body ops
      surface:
        type: Plane
      obscuration:
        type: ObscCircle
        radius: 0.80469  # = 1.60938/2 m; camera body outer diameter
      coordSys:
        x: {cs.origin[0]}
        y: {cs.origin[1]}
        z: {cs.origin[2]}
        rotX: {rx}
        rotY: {ry}
        rotZ: {rz}
"""
    )

    print()