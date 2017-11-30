#include "xform.h"
#include "ray.h"
#include "vec3.h"
#include "utils.h"

namespace batoid{
    XForm::XForm(Rot3 _R, Vec3 _dr) : R(_R), dr(_dr) {}

    Ray XForm::to(const Ray& r) const {
        if (r.failed) return r;
        return Ray{RotVec(R, r.p0-dr),
                   RotVec(R, r.v),
                   r.t0,
                   r.wavelength,
                   r.isVignetted};
    }

    Ray XForm::from(const Ray& r) const {
        if (r.failed) return r;
        return Ray{UnRotVec(R, r.p0)+dr,
                   UnRotVec(R, r.v),
                   r.t0,
                   r.wavelength,
                   r.isVignetted};
    }

    std::vector<Ray> XForm::to(const std::vector<Ray>& rays) const {
        std::vector<Ray> result(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [this](const Ray& ray) { return to(ray); },
            2000
        );
        return result;
    }

    std::vector<Ray> XForm::from(const std::vector<Ray>& rays) const {
        std::vector<Ray> result(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [this](const Ray& ray) { return from(ray); },
            2000
        );
        return result;
    }

    XForm XForm::inverse() const {
        return XForm(R.inverse(), -RotVec(R, dr));
    }

    // Composition: xf2 followed by xf1
    XForm operator*(const XForm& xf1, const XForm& xf2) {
        Rot3 R = xf1.R * xf2.R;
        Vec3 dr = xf2.dr + UnRotVec(xf2.R, xf1.dr);
        return XForm(R, dr);
    }
}
