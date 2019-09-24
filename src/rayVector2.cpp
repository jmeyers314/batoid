#include "rayVector2.h"

namespace batoid {
    RayVector2::RayVector2(
        Ref<MatrixX3d> _r, Ref<MatrixX3d> _v, Ref<VectorXd> _t,
        Ref<VectorXd> _wavelength, Ref<VectorXd> _flux,
        Ref<VectorXb> _vignetted, Ref<VectorXb> _failed
    ) : r(_r), v(_v), t(_t),
        wavelength(_wavelength), flux(_flux),
        vignetted(_vignetted), failed(_failed) {
            // Let's rearrange things to see if they make sense back in python...
            for (int i=0; i<101; i++) {
                r(i, 0) += i;
            }
            for (int i=0; i<3; i++) {
                v(0, i) += i;
            }
        }
}
