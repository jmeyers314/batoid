#include "surface.h"
#include "utils.h"

namespace batoid {
    std::ostream& operator<<(std::ostream& os, const Surface& s) {
        return os << s.repr();
    }

    RayVector Surface::intersect(const RayVector& rv) const {
        std::vector<Ray> rays(rv.rays.size());

        parallelTransform(rv.rays.cbegin(), rv.rays.cend(), rays.begin(),
            [this](const Ray& ray)
            { return intersect(ray); }
        );
        return RayVector(std::move(rays), rv.wavelength);
    }

    void Surface::intersectInPlace(RayVector& rv) const {
        parallel_for_each(
            rv.rays.begin(), rv.rays.end(),
            [this](Ray& r) { intersectInPlace(r); }
        );
    }
}
