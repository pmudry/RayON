#ifndef HITTABLE_H
#define HITTABLE_H

#include "utils.h"

/**
 * An abstract class for something that can be hit by a ray
 */
class hit_record
{
public:
    point3 p; // The point where the ray hits the object
    vec3 normal; // The normal vector at the hit point
    double t; // The ray distance at the hit point
    bool frontFacing; // True if the ray hits the front face of the object

    void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.
        frontFacing = dot(r.direction(), outward_normal) < 0;
        normal = frontFacing ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    virtual ~hittable() = default;

    virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const = 0;
};

#endif