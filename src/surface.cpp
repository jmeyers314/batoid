#include "surface.h"
#include "transformation.h"

namespace jtrace {
    Transformation Surface::shift(double dx, double dy, double dz) const {
        return Transformation(this, dx, dy, dz);
    }
}
