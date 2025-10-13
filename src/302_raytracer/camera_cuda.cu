/**
 * @file camera_cuda.cu
 * @brief CUDA-accelerated ray tracing implementation with support for various materials
 *        including displacement-mapped spheres for organic surface variations.
 *
 * This file contains the GPU kernels and device functions for ray tracing, including:
 * - Basic geometric primitives (spheres, rectangles)
 * - Material handling (Lambertian, Mirror, Rough Mirror, Glass, Light)
 * - Noise-based surface displacement for organic sphere deformation
 * - Multi-threaded rendering with anti-aliasing
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cfloat>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Golf ball debug modes:
// 0 = off (regular shading)
// 1 = show normals as colors (0.5*(n+1))
// 2 = show displacement value (grayscale, brighter = deeper dimple)
// 3 = show gradient magnitude (grayscale)
#define DEBUG_GOLF_NORMALS 0

//==============================================================================
// VECTOR MATH AND UTILITY STRUCTURES
//==============================================================================

/**
 * @brief Simple 3D vector structure optimized for CUDA
 * Provides basic vector operations for ray tracing computations
 */
struct float3_simple
{
    float x, y, z;
    __device__ __host__ float3_simple() : x(0), y(0), z(0) {}
    __device__ __host__ float3_simple(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __device__ __host__ float3_simple operator+(const float3_simple &other) const
    {
        return float3_simple(x + other.x, y + other.y, z + other.z);
    }

    __device__ __host__ float3_simple operator-(const float3_simple &other) const
    {
        return float3_simple(x - other.x, y - other.y, z - other.z);
    }

    __device__ __host__ float3_simple operator*(float t) const
    {
        return float3_simple(x * t, y * t, z * t);
    }

    __device__ __host__ float3_simple operator/(float t) const
    {
        return float3_simple(x / t, y / t, z / t);
    }

    __device__ __host__ float3_simple operator-() const
    {
        return float3_simple(-x, -y, -z);
    }

    __device__ __host__ float length() const
    {
        return sqrtf(x * x + y * y + z * z);
    }

    __device__ __host__ float length_squared() const
    {
        return x * x + y * y + z * z;
    }
};

__device__ __host__ float3_simple operator*(float t, const float3_simple &v)
{
    return v * t;
}

/** @brief Compute dot product of two vectors */
__device__ __host__ float dot(const float3_simple &a, const float3_simple &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** @brief Compute cross product of two vectors */
__device__ __host__ float3_simple cross(const float3_simple &a, const float3_simple &b)
{
    return float3_simple(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

/** @brief Normalize a vector to unit length */
__device__ __host__ float3_simple unit_vector(const float3_simple &v)
{
    return v / v.length();
}

/** @brief Convert a normal to a debug RGB color */
__device__ __host__ inline float3_simple normal_to_color(const float3_simple &n)
{
    return float3_simple(0.5f * (n.x + 1.0f), 0.5f * (n.y + 1.0f), 0.5f * (n.z + 1.0f));
}

__device__ __host__ inline float3_simple grayscale(float v)
{
    v = fmaxf(0.0f, fminf(1.0f, v));
    return float3_simple(v, v, v);
}

//==============================================================================
// OPTICAL PHYSICS FUNCTIONS
//==============================================================================

/**
 * @brief Calculate reflection direction using the law of reflection
 * @param v Incident ray direction (should be normalized)
 * @param n Surface normal (should be normalized)
 * @return Reflected ray direction
 */
__device__ float3_simple reflect(const float3_simple &v, const float3_simple &n)
{
    return v - 2 * dot(v, n) * n;
}

/**
 * @brief Calculate refraction direction using Snell's law
 * @param uv Incident ray direction (normalized)
 * @param n Surface normal (normalized)
 * @param etai_over_etat Ratio of refractive indices (n1/n2)
 * @return Refracted ray direction
 */
__device__ float3_simple refract(const float3_simple &uv, const float3_simple &n, float etai_over_etat)
{
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    float3_simple r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float3_simple r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

/**
 * @brief Schlick's approximation for Fresnel reflectance
 * Used to determine reflection probability for dielectric materials
 * @param cosine Cosine of angle between incident ray and normal
 * @param ref_idx Refractive index of the material
 * @return Probability of reflection (0-1)
 */
__device__ float reflectance(float cosine, float ref_idx)
{
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

//==============================================================================
// RAY TRACING DATA STRUCTURES
//==============================================================================

/**
 * @brief Simple ray structure for ray tracing calculations
 */
struct ray_simple
{
    float3_simple orig, dir; ///< Ray origin and direction

    __device__ __host__ ray_simple() {}
    __device__ __host__ ray_simple(const float3_simple &origin, const float3_simple &direction)
        : orig(origin), dir(direction) {}

    /** @brief Get point along ray at parameter t */
    __device__ __host__ float3_simple at(float t) const
    {
        return orig + t * dir;
    }
};

/**
 * @brief Material types supported by the ray tracer
 */
enum MaterialType
{
    LAMBERTIAN = 0,  ///< Diffuse/matte surfaces
    MIRROR = 1,      ///< Perfect mirror surfaces
    GLASS = 2,       ///< Transparent dielectric materials
    LIGHT = 3,       ///< Emissive light sources
    ROUGH_MIRROR = 4 ///< Imperfect mirror with surface roughness
};

/**
 * @brief Hit record containing intersection information
 * Stores all data needed for shading at intersection points
 */
struct hit_record_simple
{
    float3_simple p, normal; ///< Intersection point and surface normal
    float t;                 ///< Ray parameter at intersection
    bool front_face;         ///< Whether ray hits front face
    MaterialType material;   ///< Material type at intersection
    float3_simple color;     ///< Base color for Lambertian materials
    float refractive_index;  ///< Refractive index for glass materials
    float3_simple emission;  ///< Emission color for light materials
    float roughness;         ///< Surface roughness for rough mirrors
};

//==============================================================================
// RANDOM NUMBER GENERATION AND SAMPLING
//==============================================================================

/** @brief Generate random float in range [0,1) using CUDA's curand */
__device__ float random_float(curandState *state)
{
    return curand_uniform(state);
}

/**
 * @brief Generate fuzzy reflection direction for rough mirror surfaces
 * @param v Incident ray direction (normalized)
 * @param n Surface normal (normalized)
 * @param roughness Surface roughness factor (0=perfect mirror, 1=very rough)
 * @param state Random number generator state
 * @return Fuzzy reflected ray direction
 */
__device__ float3_simple reflect_fuzzy(const float3_simple &v, const float3_simple &n, float roughness, curandState *state)
{
    // Generate random perturbation vector in unit sphere
    float3_simple random_in_sphere;
    do
    {
        random_in_sphere = 2.0f * float3_simple(random_float(state), random_float(state), random_float(state)) - float3_simple(1.0f, 1.0f, 1.0f);
    } while (random_in_sphere.length_squared() >= 1.0f);

    // Perturb the normal and reflect off the modified surface
    float3_simple perturbed_normal = unit_vector(n + roughness * random_in_sphere);
    return reflect(v, perturbed_normal);
}

/**
 * @brief Generate random direction in hemisphere around surface normal
 * Used for Lambertian (diffuse) material sampling
 * @param normal Surface normal direction
 * @param state Random number generator state
 * @return Random direction in hemisphere
 */
__device__ float3_simple random_in_hemisphere(const float3_simple &normal, curandState *state)
{
    // Generate random point in unit sphere using rejection sampling
    float3_simple in_unit_sphere;
    do
    {
        in_unit_sphere = 2.0f * float3_simple(random_float(state), random_float(state), random_float(state)) - float3_simple(1.0f, 1.0f, 1.0f);
    } while (in_unit_sphere.length_squared() >= 1.0f);

    // Ensure the direction is in the same hemisphere as the normal
    if (dot(in_unit_sphere, normal) > 0.0f)
        return in_unit_sphere;
    else
        return float3_simple(-in_unit_sphere.x, -in_unit_sphere.y, -in_unit_sphere.z);
}

/**
 * @brief Sample random point on rectangular area light for soft shadows
 * @param state Random number generator state
 * @return Random point on the area light surface
 */
__device__ float3_simple sample_area_light(curandState *state)
{
    float3_simple light_corner(-1.0f, 3.0f, -2.0f); // Light position
    float3_simple light_u(2.0f, 0.0f, 0.0f);        // Width vector
    float3_simple light_v(0.0f, 0.0f, 1.0f);        // Height vector

    float alpha = random_float(state);
    float beta = random_float(state);

    return light_corner + alpha * light_u + beta * light_v;
}

/** @brief Smooth interpolation function for gradual transitions */
__device__ float smoothstep(float edge0, float edge1, float x)
{
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

//==============================================================================
// PROCEDURAL GENERATION UTILITIES
//==============================================================================

/**
 * @brief Generate random position on sphere surface using spherical coordinates
 * @param seed Deterministic seed for consistent placement
 * @param center Sphere center
 * @param radius Sphere radius
 * @return Random point on sphere surface
 */
__device__ float3_simple random_sphere_position(int seed, float3_simple center, float radius)
{
    curandState local_state;
    curand_init(seed * 1234567 + 987654321, 0, 0, &local_state);

    float theta = 2.0f * M_PI * random_float(&local_state);      // Azimuth [0, 2π]
    float phi = acosf(1.0f - 2.0f * random_float(&local_state)); // Polar [0, π]

    // Convert spherical to cartesian coordinates
    float x = sinf(phi) * cosf(theta);
    float y = sinf(phi) * sinf(theta);
    float z = cosf(phi);

    return center + float3_simple(x * radius, y * radius, z * radius);
}

// Device function for sphere intersection
__device__ bool hit_sphere(const float3_simple &center, float radius, const ray_simple &r, float t_min, float t_max, hit_record_simple &rec)
{
    // Calculate vector from ray origin to sphere center
    float3_simple oc = r.orig - center;

    // Quadratic equation coefficients for ray-sphere intersection
    // Ray equation: P(t) = A + t*B, where A is origin, B is direction
    // Sphere equation: (P-C)·(P-C) = r², where C is center, r is radius
    // Substituting ray into sphere equation gives: at² + 2bt + c = 0
    float a = dot(r.dir, r.dir);             // Coefficient a: direction·direction
    float half_b = dot(oc, r.dir);           // Half of coefficient b: (origin-center)·direction
    float c = dot(oc, oc) - radius * radius; // Coefficient c: |origin-center|² - r²

    // Calculate discriminant to check if ray intersects sphere
    float discriminant = half_b * half_b - a * c;

    // No intersection if discriminant is negative
    if (discriminant < 0)
        return false;

    // Calculate the two possible intersection points
    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a; // Closer intersection point

    // Check if closer intersection is within valid t range
    if (root < t_min || t_max < root)
    {
        root = (-half_b + sqrtd) / a; // Try farther intersection point
        if (root < t_min || t_max < root)
            return false; // Both intersections outside valid range
    }

    // Fill hit record with intersection details
    rec.t = root;        // Parameter t where intersection occurs
    rec.p = r.at(rec.t); // Point of intersection

    // Calculate outward-pointing normal at intersection point
    float3_simple outward_normal = (rec.p - center) / radius;

    // Determine if ray hits front face (ray and normal point in opposite directions)
    rec.front_face = dot(r.dir, outward_normal) < 0;

    // Set normal to always point against the ray direction
    rec.normal = rec.front_face ? outward_normal : float3_simple(-outward_normal.x, -outward_normal.y, -outward_normal.z);

    return true;
}

/**
 * @brief Ray-rectangle intersection for area lights
 * @param corner One corner of the rectangle
 * @param u Vector defining one edge of the rectangle
 * @param v Vector defining the adjacent edge of the rectangle
 * @param r Ray to intersect
 * @param t_min Minimum valid intersection distance
 * @param t_max Maximum valid intersection distance
 * @param rec Hit record to fill with intersection data
 * @return True if intersection found within rectangle bounds
 */
__device__ bool hit_rectangle(const float3_simple &corner, const float3_simple &u, const float3_simple &v,
                              const ray_simple &r, float t_min, float t_max, hit_record_simple &rec)
{
    // Calculate normal vector (perpendicular to rectangle plane)
    float3_simple normal = unit_vector(float3_simple(
        u.y * v.z - u.z * v.y, // Cross product u × v
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x));

    // Calculate intersection with plane: normal · (P - corner) = 0
    // For ray P = origin + t * direction: t = normal · (corner - origin) / (normal · direction)
    float denom = dot(normal, r.dir);

    // If denominator is close to 0, ray is parallel to plane
    if (fabsf(denom) < 1e-8f)
        return false;

    float t = dot(normal, corner - r.orig) / denom;

    if (t < t_min || t > t_max)
        return false;

    // Find intersection point
    float3_simple intersection = r.at(t);

    // Check if intersection is within rectangle bounds
    float3_simple p = intersection - corner;

    // Project onto rectangle's coordinate system
    float alpha = dot(p, u) / dot(u, u);
    float beta = dot(p, v) / dot(v, v);

    // Check if point is inside rectangle (0 <= alpha <= 1, 0 <= beta <= 1)
    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
        return false;

    // We have a valid hit
    rec.t = t;
    rec.p = intersection;
    rec.front_face = dot(r.dir, normal) < 0;
    rec.normal = rec.front_face ? normal : float3_simple(-normal.x, -normal.y, -normal.z);

    return true;
}

//==============================================================================
// GOLF BALL SURFACE DISPLACEMENT
//==============================================================================

/**
 * @brief Convert 3D point to spherical coordinates with 90-degree rotation around x-axis
 * @param p 3D point on sphere surface
 * @param theta Output azimuthal angle (0 to 2π)
 * @param phi Output polar angle (0 to π)
 */
__device__ void cartesianToSpherical(float3_simple p, float &theta, float &phi)
{
    float r = p.length();
    if (r < 1e-6f)
    {
        theta = 0.0f;
        phi = 0.0f;
        return;
    }

    // Apply 90-degree rotation around x-axis: (x,y,z) -> (x,-z,y)
    float3_simple rotated = float3_simple(p.x, -p.z, p.y);

    theta = atan2f(rotated.y, rotated.x);
    if (theta < 0.0f)
        theta += 2.0f * M_PI;
    phi = acosf(rotated.z / r);
}

/**
 * @brief Find distance to nearest point in a regular icosahedral distribution
 * @param p 3D point on unit sphere surface (normalized)
 * @return Distance to nearest dimple center
 */
// Generate i-th point on a Fibonacci sphere with n points
__device__ inline float3_simple fibonacci_point(int i, int n)
{
    // Golden angle in radians
    const float ga = 2.39996323f; // ~pi * (3 - sqrt(5))
    float k = (float)i + 0.5f;
    float v = k / (float)n;
    float phi = acosf(1.0f - 2.0f * v);
    float theta = ga * k;
    float s = sinf(phi);
    return float3_simple(cosf(theta) * s, sinf(theta) * s, cosf(phi));
}

// Angular distance (radians) to nearest Fibonacci dimple center
__device__ float distanceToNearestDimple(float3_simple p)
{
    float3_simple q = unit_vector(p);
    const int N = 150; // number of dimple centers; adjust for density
    float max_dot = -1.0f;
    for (int i = 0; i < N; ++i)
    {
        float3_simple c = fibonacci_point(i, N);
        float d = dot(q, c);
        if (d > max_dot)
            max_dot = d;
    }
    // Angular distance between q and nearest center
    max_dot = fmaxf(fminf(max_dot, 1.0f), -1.0f);
    return acosf(max_dot);
}

/**
 * @brief Create regular dimple pattern using icosahedral distribution
 * @param p 3D point on unit sphere surface (normalized)
 * @return Dimple depth (negative for inward displacement)
 */
__device__ float hexagonalDimplePattern(float3_simple p)
{
    // Angular distance to the nearest Fibonacci center
    float ang = distanceToNearestDimple(unit_vector(p));

    // Dimple size and depth in angular units
    const float dimple_radius = 0.24f; // ~13.7 degrees
    const float dimple_depth = 0.35f;

    if (ang < dimple_radius)
    {
        float t = ang / dimple_radius; // 0..1
        float depth = dimple_depth * cosf(t * M_PI * 0.5f);
        return -depth; // inward displacement
    }
    return 0.0f;
}

//==============================================================================
// RED SPHERE BLACK DOTS (FIBONACCI GRID)
//==============================================================================

/**
 * @brief Nearest angular distance to a Fibonacci-grid center (parameterized N)
 * @param dir Unit vector on sphere
 * @param N Number of centers (controls spacing)
 */
__device__ float nearestAngularDistanceFibonacci(float3_simple dir, int N)
{
    float3_simple q = unit_vector(dir);
    float max_dp = -1.0f;
    for (int i = 0; i < N; ++i)
    {
        float3_simple c = fibonacci_point(i, N);
        float d = dot(q, c);
        if (d > max_dp)
            max_dp = d;
    }
    max_dp = fmaxf(fminf(max_dp, 1.0f), -1.0f);
    return acosf(max_dp);
}

/**
 * @brief Calculate golf ball surface displacement using regular hexagonal dimple pattern
 * @param p Surface point in world coordinates
 * @param center Sphere center in world coordinates
 * @param radius Sphere radius
 * @return Displacement amount (negative creates dimples)
 */
__device__ float golfBallDisplacement(float3_simple p, float3_simple center, float radius)
{
    // Convert to sphere-local coordinates (relative to sphere center)
    float3_simple local_point = float3_simple(p.x - center.x, p.y - center.y, p.z - center.z);

    // Normalize to get position on unit sphere surface
    float3_simple normalized = unit_vector(local_point);

    // Get regular hexagonal dimple displacement
    float displacement = hexagonalDimplePattern(normalized);

    return displacement;
}

/**
 * @brief Ray-sphere intersection with golf ball surface displacement
 * @param center Sphere center
 * @param radius Sphere radius
 * @param r Ray to intersect
 * @param t_min Minimum intersection distance
 * @param t_max Maximum intersection distance
 * @param rec Hit record to fill
 * @return True if intersection found
 */
__device__ bool hit_golf_ball_sphere(float3_simple center, float radius, const ray_simple &r,
                                     float t_min, float t_max, hit_record_simple &rec)
{
    // First, find intersection with base sphere
    if (!hit_sphere(center, radius, r, t_min, t_max, rec))
    {
        return false;
    }

    // Compute displacement value at the base hit point
    float3_simple surface_point = rec.p;
    float base_displacement = golfBallDisplacement(surface_point, center, radius);

    // Strengths
    const float displacement_scale = 0.2f; // normal perturbation strength
    // Geometric displacement: outward-only push to enhance rim highlight while avoiding self-intersections.
    // We push more outside dimples and less inside them (never inward), keeping the surface stable.
    const float dimple_depth_param = 0.35f; // must match hexagonalDimplePattern()
    const float geo_strength = 0.35f;       // max outward push as a fraction of radius

    // Outward normal of the base sphere at the hit location (stable reference)
    float3_simple base_outward = unit_vector(float3_simple(surface_point.x - center.x, surface_point.y - center.y, surface_point.z - center.z));

    // Apply outward-only geometric displacement to enhance rim highlight without self-trapping
    float d_norm = fminf(1.0f, fmaxf(0.0f, -base_displacement / dimple_depth_param)); // 0 outside -> 1 deepest dimple
    float outward_push = radius * geo_strength * (1.0f - d_norm);                     // more push outside dimples
    rec.p = float3_simple(
        surface_point.x + base_outward.x * outward_push,
        surface_point.y + base_outward.y * outward_push,
        surface_point.z + base_outward.z * outward_push);

    // Base shading normal from displaced position
    float3_simple base_normal = unit_vector(float3_simple(rec.p.x - center.x, rec.p.y - center.y, rec.p.z - center.z));

    // For actual dimple areas, perturb the normal using tangent-space gradient of the displacement field
    if (base_displacement < -0.001f)
    {
        // Build an orthonormal basis (t1, t2) for the tangent plane at base_normal
        float3_simple helper = fabsf(base_normal.x) > 0.8f ? float3_simple(0, 1, 0) : float3_simple(1, 0, 0);
        float3_simple t1 = unit_vector(cross(helper, base_normal));
        float3_simple t2 = cross(base_normal, t1); // already orthogonal, length ~1

        // Sample displacement on the unit sphere in small angular steps along t1 and t2
        const float h = 0.015f;            // small angle in radians (sharper)
        float3_simple p_hat = base_normal; // unit direction from center after displacement
        // Evaluate displacement using the pattern directly in direction space
        float d0 = hexagonalDimplePattern(p_hat);
        float d1 = hexagonalDimplePattern(unit_vector(float3_simple(p_hat.x + h * t1.x, p_hat.y + h * t1.y, p_hat.z + h * t1.z)));
        float d2 = hexagonalDimplePattern(unit_vector(float3_simple(p_hat.x + h * t2.x, p_hat.y + h * t2.y, p_hat.z + h * t2.z)));

        // Tangent gradient of the displacement field on the sphere
        float dd1 = (d1 - d0) / h;
        float dd2 = (d2 - d0) / h;
        float3_simple grad_tan = float3_simple(dd1 * t1.x + dd2 * t2.x,
                                               dd1 * t1.y + dd2 * t2.y,
                                               dd1 * t1.z + dd2 * t2.z);

        // Perturbed normal follows the gradient of the implicit surface F(x)=|x-c|- (R + s*d(p_hat))
        // n ≈ base_normal - s * grad_tan
        float3_simple delta_n = float3_simple(
            -displacement_scale * grad_tan.x,
            -displacement_scale * grad_tan.y,
            -displacement_scale * grad_tan.z);

        // Silhouette-safe attenuation: reduce perturbation at grazing view angles to avoid black edges
        float3_simple view_dir = unit_vector(float3_simple(-r.dir.x, -r.dir.y, -r.dir.z));
        float ndv = fmaxf(0.0f, dot(base_normal, view_dir));
        float atten = smoothstep(0.1f, 0.4f, ndv); // 0 near silhouette, 1 away from it
        delta_n = float3_simple(delta_n.x * atten, delta_n.y * atten, delta_n.z * atten);

        // Clamp perturbation to avoid flipping normals (black shading)
        float max_len = 0.4f; // safe limit
        float len = delta_n.length();
        if (len > max_len && len > 1e-6f)
        {
            delta_n = (max_len / len) * delta_n;
        }

        float3_simple perturbed = unit_vector(float3_simple(
            base_normal.x + delta_n.x,
            base_normal.y + delta_n.y,
            base_normal.z + delta_n.z));

        // Ensure normal points outward relative to sphere center
        if (dot(perturbed, base_normal) < 0.0f)
        {
            perturbed = float3_simple(-perturbed.x, -perturbed.y, -perturbed.z);
        }

        // NaN guard
        if (!(perturbed.x == perturbed.x) || !(perturbed.y == perturbed.y) || !(perturbed.z == perturbed.z))
        {
            perturbed = base_normal;
        }

        rec.normal = perturbed;
    }
    else
    {
        // Use standard outward normal for non-dimple areas
        rec.normal = base_normal;
    }

    // Set front face based on ray direction and flip to oppose incoming ray
    float ndotv = dot(r.dir, rec.normal);
    if (!(ndotv == ndotv))
    { // NaN guard
        rec.normal = base_normal;
        ndotv = dot(r.dir, rec.normal);
    }
    rec.front_face = ndotv < 0;
    if (!rec.front_face)
    {
        rec.normal = float3_simple(-rec.normal.x, -rec.normal.y, -rec.normal.z);
    }

    // Push the hit point slightly along the final normal to avoid self-intersections
    // and starting the next bounce inside the surface (prevents black artifacts)
    const float surface_epsilon = 1e-3f; // in world units
    rec.p = float3_simple(
        rec.p.x + rec.normal.x * surface_epsilon,
        rec.p.y + rec.normal.y * surface_epsilon,
        rec.p.z + rec.normal.z * surface_epsilon);

    return true;
}

//==============================================================================
// SCENE DEFINITION AND INTERSECTION
//==============================================================================

/**
 * @brief Test ray against all objects in the scene
 * @param r Ray to test against scene
 * @param t_min Minimum valid intersection distance
 * @param t_max Maximum valid intersection distance
 * @param rec Hit record to fill with closest intersection
 * @return True if any intersection found, false otherwise
 */
__device__ bool hit_world(const ray_simple &r, float t_min, float t_max, hit_record_simple &rec)
{
    hit_record_simple temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    // Ground sphere with rough mirror surface and slight blue tint
    if (hit_sphere(float3_simple(0, -950.5f, -1), 950.0f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.color = float3_simple(0.44f, 0.7f, .95f); // Slight blue tint (cool metal)
        rec.material = ROUGH_MIRROR;
        rec.roughness = 0.7f; // Higher roughness for ground surface
    }

    // Left sphere (rough mirror with golden tint)
    if (hit_sphere(float3_simple(-3.5, 0.45, -1.8), 0.8f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = ROUGH_MIRROR;
        rec.color = float3_simple(1.0f, 0.85f, 0.47f); // Golden tint (brass/gold color)
        rec.roughness = 0.03f;                         // Moderate surface roughness for imperfect reflection
    }

    // "Golf" ball displaced sphere
    if (hit_golf_ball_sphere(float3_simple(1.2, 0, -2), 0.5f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = ROUGH_MIRROR;
        rec.roughness = 0.3f;                         // Moderate surface roughness for imperfect reflection
        rec.color = float3_simple(0.3f, 0.3f, 0.91f); // Red
    }

    // The red-black dotted sphere
    if (hit_sphere(float3_simple(-1.3f, 0.18, -5), 0.7f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        
        // Base red and dot color (near-black so it still responds to light)
        float3_simple base_color = float3_simple(0.9f, 0.1f, 0.1f);
        float3_simple dot_color = float3_simple(0.02f, 0.02f, 0.02f);

        // Apply regularly spaced black dots in the sphere's local space
        // Local direction from sphere center
        float3_simple red_center = float3_simple(-1.3f, 0.18f, -5.0f);
        float3_simple local = float3_simple(rec.p.x - red_center.x, rec.p.y - red_center.y, rec.p.z - red_center.z);
        float3_simple dir = unit_vector(local);

        // Fibonacci grid parameters: relatively big, regularly spaced
        const int ndots = 12;            // number of centers (spacing)
        const float dot_radius = 0.33f; // angular radius in radians (~13 deg)

        float ang = nearestAngularDistanceFibonacci(dir, ndots);
        float mask = ang < dot_radius ? 0.0f : 1.0f; // 0 inside dot -> dot color, 1 outside -> base red
        rec.color = float3_simple(
            base_color.x * mask + dot_color.x * (1.0f - mask),
            base_color.y * mask + dot_color.y * (1.0f - mask),
            base_color.z * mask + dot_color.z * (1.0f - mask));
    }

    if (hit_sphere(float3_simple(-.7f, .2, -.3f), 0.6f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = GLASS;
        rec.refractive_index = 1.5f; // Typical glass refractive index
    }

    // ISC Logo spheres
    if (hit_sphere(float3_simple(-3.5f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(247 / 255.0f, 241 / 255.0f, 159 / 255.0f); // Yellow
    }

    if (hit_sphere(float3_simple(-3.0f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(140 / 255.0f, 198 / 255.0f, 230 / 255.0f); // Blue
    }

    if (hit_sphere(float3_simple(-2.5f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(168 / 255.0f, 144 / 255.0f, 192 / 255.0f); // Violoet
    }

    if (hit_sphere(float3_simple(-2.0f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(226 / 255.0f, 171 / 255.0f, 186 / 255.0f); // Rose
    }

    if (hit_sphere(float3_simple(-1.5f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(152.0f / 255.0f, 199.0f / 255.0f, 191.0f / 255.0f); // Green
    }

    // 12 Simple spheres in circle around golden sphere
    float3_simple golden_center = float3_simple(-3.5, 0, -2);

    // Area light - rectangular light source above the scene
    float3_simple light_corner(-1.0f, 3.0f, -2.0f); // Corner position
    float3_simple light_u(2.5f, 0.0f, 0.0f);        // Width vector (2 units wide)
    float3_simple light_v(0.0f, 0.0f, 1.5f);        // Height vector (1 unit tall)

    if (hit_rectangle(light_corner, light_u, light_v, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LIGHT;
        rec.emission = float3_simple(5.5f, 4.8f, 4.2f); // Warm white light (intensity 5)
    }

    return hit_anything;
}

//==============================================================================
// RAY TRACING AND SHADING
//==============================================================================

/**
 * @brief Calculate color contribution from a ray using recursive ray tracing
 * @param r Ray to trace through the scene
 * @param state Random number generator state for Monte Carlo sampling
 * @param depth Remaining recursion depth (prevents infinite recursion)
 * @param ray_count Counter for total rays traced (for performance metrics)
 * @return Final color contribution from this ray
 */
__device__ float3_simple ray_color(const ray_simple &r, curandState *state, int depth, unsigned long long *ray_count)
{
    if (depth <= 0)
        return float3_simple(0, 0, 0);

    // Increment ray counter atomically
    atomicAdd(ray_count, 1);

    hit_record_simple rec;
    if (hit_world(r, 0.001f, FLT_MAX, rec))
    {
        // If we hit a light source, return its emission
        if (rec.material == LIGHT)
        {
            return rec.emission;
        }

#if DEBUG_GOLF_NORMALS
        // Approximate golf ball hit detection by center/radius
        const float3_simple golf_center(1.0f, 0.0f, -2.0f);
        const float golf_radius = 0.5f;
        float3_simple to_golf = float3_simple(rec.p.x - golf_center.x, rec.p.y - golf_center.y, rec.p.z - golf_center.z);
        float dist = to_golf.length();
        bool near_golf_surface = fabsf(dist - golf_radius) < 0.25f;
        if (near_golf_surface)
        {
#if DEBUG_GOLF_NORMALS == 1
            return normal_to_color(rec.normal);
#elif DEBUG_GOLF_NORMALS == 2
            // Visualize displacement value on unit sphere
            float3_simple dir = dist > 1e-6f ? (to_golf / dist) : float3_simple(0, 0, 1);
            float d = hexagonalDimplePattern(dir); // negative inside dimple
            // Map: deeper dimple (more negative) -> brighter (abs)
            return grayscale(fminf(fabsf(d) * 3.0f, 1.0f));
#elif DEBUG_GOLF_NORMALS == 3
            // Visualize gradient magnitude of displacement field
            float3_simple dir = dist > 1e-6f ? (to_golf / dist) : float3_simple(0, 0, 1);
            // Build tangent basis
            float3_simple helper = fabsf(dir.x) > 0.8f ? float3_simple(0, 1, 0) : float3_simple(1, 0, 0);
            float3_simple t1 = unit_vector(cross(helper, dir));
            float3_simple t2 = cross(dir, t1);
            const float h = 0.02f;
            float d0 = hexagonalDimplePattern(dir);
            float d1 = hexagonalDimplePattern(unit_vector(float3_simple(dir.x + h * t1.x, dir.y + h * t1.y, dir.z + h * t1.z)));
            float d2 = hexagonalDimplePattern(unit_vector(float3_simple(dir.x + h * t2.x, dir.y + h * t2.y, dir.z + h * t2.z)));
            float dd1 = (d1 - d0) / h;
            float dd2 = (d2 - d0) / h;
            float mag = sqrtf(dd1 * dd1 + dd2 * dd2);
            return grayscale(fminf(mag * 0.5f, 1.0f));
#else
            return normal_to_color(rec.normal);
#endif
        }
#endif

        float3_simple attenuation;
        ray_simple scattered;

        if (rec.material == LAMBERTIAN)
        {
            // Lambertian scattering - different colors for different spheres
            float3_simple target = rec.p + rec.normal + random_in_hemisphere(rec.normal, state);
            scattered = ray_simple(rec.p, target - rec.p);
            attenuation = rec.color;
        }
        else if (rec.material == MIRROR)
        {
            // Mirror reflection
            float3_simple reflected = reflect(unit_vector(r.dir), rec.normal);
            scattered = ray_simple(rec.p, reflected);
            attenuation = float3_simple(.99f, .99f, .99f); // Slightly blue-tinted mirror
        }
        else if (rec.material == ROUGH_MIRROR)
        {
            // Rough mirror reflection with surface imperfections and custom tint
            float3_simple reflected = reflect_fuzzy(unit_vector(r.dir), rec.normal, rec.roughness, state);
            scattered = ray_simple(rec.p, reflected);

            // Rough mirrors use stored color as tint with reduced reflectivity
            float base_reflectivity = 0.8f;
            attenuation = float3_simple(
                rec.color.x * base_reflectivity,
                rec.color.y * base_reflectivity,
                rec.color.z * base_reflectivity);
        }
        else if (rec.material == GLASS)
        {
            // Glass (dielectric) material
            attenuation = float3_simple(1.0f, 1.0f, 1.0f); // Glass doesn't absorb light
            // Monte Carlo approach to glass rendering: Instead of splitting the ray into separate
            // reflection and refraction rays (which would double the ray count exponentially),
            // we probabilistically choose ONE path per ray based on Fresnel reflectance.
            // This maintains constant ray count while still producing physically accurate results
            // through statistical sampling over many rays per pixel.

            // Determine refraction ratio based on which side of the surface we're hitting
            // When ray hits front face: going from air (n=1.0) into glass (n=1.5), so ratio = 1.0/1.5
            // When ray hits back face: going from glass (n=1.5) into air (n=1.0), so ratio = 1.5/1.0
            float refraction_ratio = rec.front_face ? (1.0f / rec.refractive_index) : rec.refractive_index;

            float3_simple unit_direction = unit_vector(r.dir);

            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            float3_simple direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(state))
            {
                // Reflect
                direction = reflect(unit_direction, rec.normal);
            }
            else
            {
                // Refract
                direction = refract(unit_direction, rec.normal, refraction_ratio);
            }

            scattered = ray_simple(rec.p, direction);
        }

        float3_simple color = ray_color(scattered, state, depth - 1, ray_count);
        return float3_simple(attenuation.x * color.x, attenuation.y * color.y, attenuation.z * color.z);
    }

    // Background gradient for the world
    float3_simple unit_direction = unit_vector(r.dir);
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * float3_simple(1.0f, 1.0f, 1.0f) + t * float3_simple(0.5f, 0.7f, 1.0f);
}

//==============================================================================
// CUDA KERNELS
//==============================================================================

/**
 * @brief Initialize random number generator states for each pixel
 * @param states Array of random states (one per pixel)
 * @param seed Base random seed
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
__global__ void init_random_states(curandState *states, unsigned long seed, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        curand_init(seed + idx, idx, 0, &states[idx]);
    }
}

/**
 * @brief Main CUDA kernel for ray tracing entire image
 * Each thread processes one pixel with multiple samples for anti-aliasing
 * @param image Output image buffer (RGB, 8-bit per channel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param samples_per_pixel Number of rays per pixel for anti-aliasing
 * @param max_depth Maximum ray recursion depth
 * @param cam_center_* Camera center position components
 * @param pixel00_* Top-left pixel center position components
 * @param delta_u_* Pixel step in U direction components
 * @param delta_v_* Pixel step in V direction components
 * @param ray_count Global counter for rays traced
 */
__global__ void renderPixelsKernel(unsigned char *image, int width, int height, int samples_per_pixel, int max_depth,
                                   float cam_center_x, float cam_center_y, float cam_center_z,
                                   float pixel00_x, float pixel00_y, float pixel00_z,
                                   float delta_u_x, float delta_u_y, float delta_u_z,
                                   float delta_v_x, float delta_v_y, float delta_v_z,
                                   unsigned long long *ray_count)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Strict bounds checking
    if (x >= width || y >= height)
        return;

    int pixel_idx = y * width + x;
    int base_idx = pixel_idx * 3;

    // Double check bounds for memory access
    if (pixel_idx >= width * height || base_idx + 2 >= width * height * 3)
    {
        return;
    }

    // Initialize local random state with unique seed per pixel to avoid artifacts
    curandState local_rand_state;
    curand_init((blockIdx.x * blockDim.x + threadIdx.x) * 1000 + (blockIdx.y * blockDim.y + threadIdx.y) * 2000 + clock(),
                pixel_idx, 0, &local_rand_state);

    // Convert parameters to float3_simple
    float3_simple camera_center(cam_center_x, cam_center_y, cam_center_z);
    float3_simple pixel00_loc(pixel00_x, pixel00_y, pixel00_z);
    float3_simple pixel_delta_u(delta_u_x, delta_u_y, delta_u_z);
    float3_simple pixel_delta_v(delta_v_x, delta_v_y, delta_v_z);

    float3_simple pixel_color(0, 0, 0);

    // Use full samples but ensure each pixel is completely independent
    int actual_samples = samples_per_pixel;

    for (int s = 0; s < actual_samples; s++)
    {
        // Random offset within pixel for anti-aliasing
        float offset_u = random_float(&local_rand_state) - 0.5f;
        float offset_v = random_float(&local_rand_state) - 0.5f;

        float3_simple pixel_center = pixel00_loc + ((float)x + offset_u) * pixel_delta_u + ((float)y + offset_v) * pixel_delta_v;
        float3_simple ray_direction = pixel_center - camera_center;
        ray_simple r(camera_center, ray_direction);

        pixel_color = pixel_color + ray_color(r, &local_rand_state, min(max_depth, 6), ray_count);
    }

    // Anti-aliasing is done there
    pixel_color = pixel_color / (float)actual_samples;

    // Gamma correction (gamma=2)
    pixel_color.x = sqrtf(fmaxf(pixel_color.x, 0.0f));
    pixel_color.y = sqrtf(fmaxf(pixel_color.y, 0.0f));
    pixel_color.z = sqrtf(fmaxf(pixel_color.z, 0.0f));

    // Convert to bytes with clamping
    unsigned char r = (unsigned char)(255.0f * fminf(fmaxf(pixel_color.x, 0.0f), 1.0f));
    unsigned char g = (unsigned char)(255.0f * fminf(fmaxf(pixel_color.y, 0.0f), 1.0f));
    unsigned char b = (unsigned char)(255.0f * fminf(fmaxf(pixel_color.z, 0.0f), 1.0f));

    // Store in image buffer - each thread writes to its own unique location
    image[base_idx] = r;
    image[base_idx + 1] = g;
    image[base_idx + 2] = b;
}

/**
 * @brief Main CUDA kernel for ray tracing entire image
 * Each thread processes one pixel with multiple samples for anti-aliasing
 * @param image Output image buffer (RGB, 8-bit per channel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param samples_per_pixel Number of rays per pixel for anti-aliasing
 * @param max_depth Maximum ray recursion depth
 * @param cam_center_* Camera center position components
 * @param pixel00_* Top-left pixel center position components
 * @param delta_u_* Pixel step in U direction components
 * @param delta_v_* Pixel step in V direction components
 * @param start_x Tile start X coordinate in global image
 * @param start_y Tile start Y coordinate in global image
 * @param end_x Tile end X coordinate in global image
 * @param end_y Tile end Y coordinate in global image
 * @param ray_count Global counter for rays traced
 */
__global__ void renderPixelsTileKernel(unsigned char *image, int width, int height, int samples_per_pixel, int max_depth,
                                       float cam_center_x, float cam_center_y, float cam_center_z,
                                       float pixel00_x, float pixel00_y, float pixel00_z,
                                       float delta_u_x, float delta_u_y, float delta_u_z,
                                       float delta_v_x, float delta_v_y, float delta_v_z,
                                       int start_x, int start_y, int end_x, int end_y,
                                       unsigned long long *ray_count)
{

    // Calculate global pixel coordinates within the tile
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Convert tile coordinates to global image coordinates
    int x = start_x + tile_x;
    int y = start_y + tile_y;

    // Check if we're within the tile bounds and image bounds
    if (x >= end_x || y >= end_y || x >= width || y >= height)
        return;

    int pixel_idx = y * width + x;
    int base_idx = pixel_idx * 3;

    // Double check bounds for memory access
    if (pixel_idx >= width * height || base_idx + 2 >= width * height * 3)
    {
        return;
    }

    // Initialize local random state with unique seed per pixel
    curandState local_rand_state;
    curand_init((blockIdx.x * blockDim.x + threadIdx.x) * 1000 + (blockIdx.y * blockDim.y + threadIdx.y) * 2000 + clock(),
                pixel_idx, 0, &local_rand_state);

    // Convert parameters to float3_simple
    float3_simple camera_center(cam_center_x, cam_center_y, cam_center_z);
    float3_simple pixel00_loc(pixel00_x, pixel00_y, pixel00_z);
    float3_simple pixel_delta_u(delta_u_x, delta_u_y, delta_u_z);
    float3_simple pixel_delta_v(delta_v_x, delta_v_y, delta_v_z);

    float3_simple pixel_color(0, 0, 0);

    // Use full samples but ensure each pixel is completely independent
    int actual_samples = samples_per_pixel;

    for (int s = 0; s < actual_samples; s++)
    {
        // Random offset within pixel for anti-aliasing
        float offset_u = random_float(&local_rand_state) - 0.5f;
        float offset_v = random_float(&local_rand_state) - 0.5f;

        float3_simple pixel_center = pixel00_loc + ((float)x + offset_u) * pixel_delta_u + ((float)y + offset_v) * pixel_delta_v;
        float3_simple ray_direction = pixel_center - camera_center;
        ray_simple r(camera_center, ray_direction);

        pixel_color = pixel_color + ray_color(r, &local_rand_state, min(max_depth, 6), ray_count);
    }

    // Anti-aliasing is done there
    pixel_color = pixel_color / (float)actual_samples;

    // Gamma correction (gamma=2)
    pixel_color.x = sqrtf(fmaxf(pixel_color.x, 0.0f));
    pixel_color.y = sqrtf(fmaxf(pixel_color.y, 0.0f));
    pixel_color.z = sqrtf(fmaxf(pixel_color.z, 0.0f));

    // Convert to bytes with clamping
    unsigned char r = (unsigned char)(255.0f * fminf(fmaxf(pixel_color.x, 0.0f), 1.0f));
    unsigned char g = (unsigned char)(255.0f * fminf(fmaxf(pixel_color.y, 0.0f), 1.0f));
    unsigned char b = (unsigned char)(255.0f * fminf(fmaxf(pixel_color.z, 0.0f), 1.0f));

    // Store in image buffer - each thread writes to its own unique location
    image[base_idx] = r;
    image[base_idx + 1] = g;
    image[base_idx + 2] = b;
}

//==============================================================================
// HOST INTERFACE FUNCTIONS
//==============================================================================

/**
 * @brief Host function for tile-based rendering (useful for real-time display)
 * Renders only a rectangular portion of the image for progressive rendering
 * @param image Full image buffer (input/output)
 * @param width Full image width in pixels
 * @param height Full image height in pixels
 * @param cam_center_* Camera position components
 * @param pixel00_* Top-left pixel center position components
 * @param delta_u_* Pixel step in U direction components
 * @param delta_v_* Pixel step in V direction components
 * @param samples_per_pixel Number of rays per pixel for anti-aliasing
 * @param max_depth Maximum ray recursion depth
 * @param start_x Starting X coordinate of tile
 * @param start_y Starting Y coordinate of tile
 * @param end_x Ending X coordinate of tile (exclusive)
 * @param end_y Ending Y coordinate of tile (exclusive)
 * @return Total number of rays traced for this tile
 */
extern "C" unsigned long long renderPixelsCUDA(unsigned char *image, int width, int height,
                                               double cam_center_x, double cam_center_y, double cam_center_z,
                                               double pixel00_x, double pixel00_y, double pixel00_z,
                                               double delta_u_x, double delta_u_y, double delta_u_z,
                                               double delta_v_x, double delta_v_y, double delta_v_z,
                                               int samples_per_pixel, int max_depth,
                                               int start_x, int start_y, int end_x, int end_y)
{

    // printf("CUDA tile render: (%d,%d) to (%d,%d) of %dx%d\n", start_x, start_y, end_x, end_y, width, height);

    // Calculate tile dimensions
    int tile_width = end_x - start_x;
    int tile_height = end_y - start_y;

    // Allocate device memory for the full image (we need to maintain the full buffer)
    unsigned char *d_image;
    unsigned long long *d_ray_count;
    size_t image_size = width * height * 3 * sizeof(unsigned char);

    cudaError_t malloc_err1 = cudaMalloc(&d_image, image_size);
    cudaError_t malloc_err2 = cudaMalloc(&d_ray_count, sizeof(unsigned long long));

    if (malloc_err1 != cudaSuccess || malloc_err2 != cudaSuccess)
    {
        printf("CUDA malloc error: %s, %s\n", cudaGetErrorString(malloc_err1), cudaGetErrorString(malloc_err2));
        return 0;
    }

    // Copy current image data to device (to preserve already rendered tiles)
    cudaError_t copy_to_err = cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    if (copy_to_err != cudaSuccess)
    {
        printf("Memory copy to device error: %s\n", cudaGetErrorString(copy_to_err));
        cudaFree(d_image);
        cudaFree(d_ray_count);
        return 0;
    }

    // Initialize ray counter to zero
    cudaMemset(d_ray_count, 0, sizeof(unsigned long long));

    // Set up grid and block dimensions for the tile
    dim3 block_size(32, 4);
    dim3 grid_size((tile_width + block_size.x - 1) / block_size.x,
                   (tile_height + block_size.y - 1) / block_size.y);

    // printf("Tile grid size: (%d, %d), Block size: (%d, %d)\n", grid_size.x, grid_size.y, block_size.x, block_size.y);

    // Launch tile rendering kernel
    renderPixelsTileKernel<<<grid_size, block_size>>>(d_image, width, height, samples_per_pixel, max_depth,
                                                      (float)cam_center_x, (float)cam_center_y, (float)cam_center_z,
                                                      (float)pixel00_x, (float)pixel00_y, (float)pixel00_z,
                                                      (float)delta_u_x, (float)delta_u_y, (float)delta_u_z,
                                                      (float)delta_v_x, (float)delta_v_y, (float)delta_v_z,
                                                      start_x, start_y, end_x, end_y, d_ray_count);

    // Check for kernel errors
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(kernel_err));
        cudaFree(d_image);
        cudaFree(d_ray_count);
        return 0;
    }

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaError_t copy_err = cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);
    if (copy_err != cudaSuccess)
    {
        printf("Memory copy error: %s\n", cudaGetErrorString(copy_err));
        cudaFree(d_image);
        cudaFree(d_ray_count);
        return 0;
    }

    // Copy ray count back to host
    unsigned long long host_ray_count = 0;
    cudaError_t count_copy_err = cudaMemcpy(&host_ray_count, d_ray_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (count_copy_err != cudaSuccess)
    {
        printf("Ray count copy error: %s\n", cudaGetErrorString(count_copy_err));
        host_ray_count = 0;
    }

    // Clean up
    cudaFree(d_image);
    cudaFree(d_ray_count);

    // // Check first few pixels to verify
    // printf("First few pixels: (%d,%d,%d) (%d,%d,%d) (%d,%d,%d)\n",
    //        image[0], image[1], image[2], image[3], image[4], image[5], image[6], image[7], image[8]);

    return host_ray_count;
}