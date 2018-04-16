#include "surface.h"
#include "utils.h"

namespace batoid {
    std::ostream& operator<<(std::ostream& os, const Surface& s) {
        return os << s.repr();
    }

    std::vector<Ray> Surface::intersect(const std::vector<Ray>& rays) const {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [this](const Ray& ray)
            { return intersect(ray); }
        );
        return result;
    }

    void Surface::intersectInPlace(std::vector<Ray>& rays) const {
        parallel_for_each(
            rays.begin(), rays.end(),
            [this](Ray& r) { intersectInPlace(r); }
        );
    }
}
