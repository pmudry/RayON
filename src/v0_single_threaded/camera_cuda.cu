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

/** @brief Normalize a vector to unit length */
__device__ __host__ float3_simple unit_vector(const float3_simple &v)
{
    return v / v.length();
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
    float3_simple orig, dir;  ///< Ray origin and direction
    
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
    LAMBERTIAN = 0,    ///< Diffuse/matte surfaces
    MIRROR = 1,        ///< Perfect mirror surfaces
    GLASS = 2,         ///< Transparent dielectric materials
    LIGHT = 3,         ///< Emissive light sources
    ROUGH_MIRROR = 4   ///< Imperfect mirror with surface roughness
};

/**
 * @brief Hit record containing intersection information
 * Stores all data needed for shading at intersection points
 */
struct hit_record_simple
{
    float3_simple p, normal;        ///< Intersection point and surface normal
    float t;                        ///< Ray parameter at intersection
    bool front_face;               ///< Whether ray hits front face
    MaterialType material;         ///< Material type at intersection
    float3_simple color;           ///< Base color for Lambertian materials
    float refractive_index;        ///< Refractive index for glass materials
    float3_simple emission;        ///< Emission color for light materials
    float roughness;               ///< Surface roughness for rough mirrors
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
    float3_simple light_corner(-1.0f, 3.0f, -2.0f);  // Light position
    float3_simple light_u(2.0f, 0.0f, 0.0f);         // Width vector
    float3_simple light_v(0.0f, 0.0f, 1.0f);         // Height vector

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
        rec.color = float3_simple(0.34f, 0.6f, .92f); // Slight blue tint (cool metal)
        rec.material = ROUGH_MIRROR;        
        rec.roughness = 0.7f; // Higher roughness for ground surface
    }

    // Left sphere (rough mirror with golden tint)
    if (hit_sphere(float3_simple(-3.5, 0.45, -1.8), 0.8f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(1.0f, 0.85f, 0.47f); // Golden tint (brass/gold color)
        rec.roughness = 0.03f;                         // Moderate surface roughness for imperfect reflection
    }

    if (hit_sphere(float3_simple(1, 0, -2), 0.5f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.color = float3_simple(0.1f, 0.2f, 0.9f); // Blue
        rec.material = LAMBERTIAN;
    }

    if (hit_sphere(float3_simple(-0.5f, 0.18, -5), 0.7f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(0.9f, 0.1f, 0.1f); // Red
    }

    // Glass sphere (new addition) - positioned to be clearly visible
    if (hit_sphere(float3_simple(-.7f, .1, -.3f), 0.5f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = GLASS;
        rec.refractive_index = 1.49f; // Glass refractive index
    }

    // ISC Logo
    if (hit_sphere(float3_simple(-3.5f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(247/255.0f, 241/255.0f, 159/255.0f); // Yellow
    }

    if (hit_sphere(float3_simple(-3.0f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(140/255.0f, 198/255.0f, 230/255.0f); // Blue
    }

    if (hit_sphere(float3_simple(-2.5f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(168/255.0f, 144/255.0f, 192/255.0f); // Violoet
    }

    if (hit_sphere(float3_simple(-2.0f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(226/255.0f, 171/255.0f, 186/255.0f); // Rose
    }

    if (hit_sphere(float3_simple(-1.5f, -0.3, 1.2), 0.2f, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LAMBERTIAN;
        rec.color = float3_simple(152.0f/ 255.0f, 199.0f/255.0f, 191.0f/255.0f); // Green
    }


    // 12 Simple spheres in circle around golden sphere
    float3_simple golden_center = float3_simple(-3.5, 0, -2);

    // Circle of 12 spheres
    // const int num_spheres = 12;
    // for (int i = 0; i < num_spheres; i++)
    // {
    //     float angle = -M_PI / 4 + (2.0f * M_PI * i) / (float)num_spheres;
    //     float x = golden_center.x + 1.8f * cosf(angle);
    //     float z = golden_center.z + 1.8f * sinf(angle);
    //     // z = fmaxf(z, -0.5f);  // Keep visible

    //     if (hit_sphere(float3_simple(x, -0.21f, z), 0.2f, r, t_min, closest_so_far, temp_rec))
    //     {
    //         hit_anything = true;
    //         closest_so_far = temp_rec.t;
    //         rec = temp_rec;
    //         rec.material = ROUGH_MIRROR;
    //         rec.roughness = 0.5f;
    //         rec.color = float3_simple(0.31f + 0.5 * (float(i) / (float)(num_spheres - 1)), 0.2f + 0.8 * (float(i) / (float)(num_spheres - 1)), 0.6f + 0.4f * (float(i) / (float)(num_spheres - 1)));
    //     }
    // }

    // Area light - rectangular light source above the scene
    float3_simple light_corner(-1.0f, 3.0f, -2.0f); // Corner position
    float3_simple light_u(2.0f, 0.0f, 0.0f);        // Width vector (2 units wide)
    float3_simple light_v(0.0f, 0.0f, 1.0f);        // Height vector (1 unit tall)

    if (hit_rectangle(light_corner, light_u, light_v, r, t_min, closest_so_far, temp_rec))
    {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        rec.material = LIGHT;
        rec.emission = float3_simple(5.0f, 4.5f, 3.5f); // Warm white light (intensity 5)
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

    printf("CUDA rays traced: %llu\n", host_ray_count);

    // // Check first few pixels to verify
    // printf("First few pixels: (%d,%d,%d) (%d,%d,%d) (%d,%d,%d)\n",
    //        image[0], image[1], image[2], image[3], image[4], image[5], image[6], image[7], image[8]);

    return host_ray_count;
}