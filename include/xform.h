#ifndef batoid_xform_h
#define batoid_xform_h

#include "vec3.h"
#include "ray.h"

namespace batoid {
    struct XForm {
        XForm(Rot3 _R, Vec3 _dr);
        Ray to(const Ray& ray) const;
        Ray from(const Ray& ray) const;
        std::vector<Ray> to(const std::vector<Ray>& rays) const;
        std::vector<Ray> from(const std::vector<Ray>& rays) const;
        XForm inverse() const;
        const Rot3 R;
        const Vec3 dr;
    };

    XForm operator*(const XForm&, const XForm&);
}

#endif
