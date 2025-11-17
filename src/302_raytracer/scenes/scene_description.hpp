/**
 * @file scene_description.hpp
 * @brief Unified scene description format for CPU and GPU renderers
 * 
 * This file provides a common scene representation that can be:
 * 1. Built once on the host
 * 2. Converted to CPU-friendly format (polymorphic, with virtual functions)
 * 3. Converted to GPU-friendly format (flat arrays, no virtual functions)
 * 
 * Supports: primitives, triangle meshes, BVH acceleration, ray marching (SDFs)
 */

#pragma once

#include "vec3.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <functional>

namespace Scene {

//==============================================================================
// MATERIAL SYSTEM
//==============================================================================

enum class MaterialType : uint8_t {
    LAMBERTIAN,
    METAL,
    MIRROR,
    ROUGH_MIRROR,
    GLASS,
    DIELECTRIC,
    LIGHT,
    CONSTANT,
    SHOW_NORMALS,
    SDF_MATERIAL     // For ray-marched objects
};

/**
 * @brief Procedural pattern types for material coloring
 */
enum class ProceduralPattern : uint8_t {
    NONE,                    // No pattern, use solid color
    FIBONACCI_DOTS,          // Regularly spaced dots using Fibonacci grid
    CHECKERBOARD,           // Checkerboard pattern
    STRIPES                 // Striped pattern
};

/**
 * @brief Material description - unified format for all material types
 */
struct MaterialDesc {
    MaterialType type;
    
    Vec3 albedo;              // Base color / reflectance
    Vec3 emission;            // Emissive color (for lights)
    float roughness;          // Surface roughness [0-1]
    float metallic;           // Metallic factor [0-1]
    float refractive_index;   // For glass/dielectric materials
    float transmission;       // Transparency [0-1]
    int texture_id;           // Texture index (-1 = none, for future use)
    
    // Procedural pattern support
    ProceduralPattern pattern;    // Pattern type
    Vec3 pattern_color;           // Secondary color for pattern
    float pattern_param1;         // Pattern-specific parameter 1 (e.g., dot count, scale)
    float pattern_param2;         // Pattern-specific parameter 2 (e.g., dot radius)
    
    // Default constructor
    MaterialDesc() 
        : type(MaterialType::LAMBERTIAN)
        , albedo(0.7, 0.7, 0.7)
        , emission(0, 0, 0)
        , roughness(0.0f)
        , metallic(0.0f)
        , refractive_index(1.0f)
        , transmission(0.0f)
        , texture_id(-1)
        , pattern(ProceduralPattern::NONE)
        , pattern_color(0, 0, 0)
        , pattern_param1(0.0f)
        , pattern_param2(0.0f)
    {}
    
    // Factory methods for common materials
    static MaterialDesc lambertian(const Vec3& color) {
        MaterialDesc mat;
        mat.type = MaterialType::LAMBERTIAN;
        mat.albedo = color;
        return mat;
    }
    
    static MaterialDesc normal() {
        MaterialDesc mat;
        mat.type = MaterialType::SHOW_NORMALS;
        mat.albedo = Vec3(1, 1, 1);
        return mat;
    }

    static MaterialDesc metal(const Vec3& color, float roughness = 0.0f) {
        MaterialDesc mat;
        mat.type = MaterialType::METAL;
        mat.albedo = color;
        mat.roughness = roughness;
        mat.metallic = 1.0f;
        return mat;
    }
    
    static MaterialDesc mirror(const Vec3& color) {
        MaterialDesc mat;
        mat.type = MaterialType::MIRROR;
        mat.albedo = color;
        mat.metallic = 1.0f;
        return mat;
    }
    
    static MaterialDesc roughMirror(const Vec3& color, float roughness) {
        MaterialDesc mat;
        mat.type = MaterialType::ROUGH_MIRROR;
        mat.albedo = color;
        mat.roughness = roughness;
        mat.metallic = 1.0f;
        return mat;
    }
    
    static MaterialDesc glass(float ior) {
        MaterialDesc mat;
        mat.type = MaterialType::GLASS;
        mat.albedo = Vec3(1, 1, 1);
        mat.refractive_index = ior;
        mat.transmission = 1.0f;
        return mat;
    }
    
    static MaterialDesc light(const Vec3& emission) {
        MaterialDesc mat;
        mat.type = MaterialType::LIGHT;
        mat.emission = emission;
        mat.albedo = Vec3(1, 1, 1);
        return mat;
    }
    
    static MaterialDesc constant(const Vec3& color) {
        MaterialDesc mat;
        mat.type = MaterialType::CONSTANT;
        mat.albedo = color;
        return mat;
    }
    
    // Factory methods for patterned materials
    static MaterialDesc fibonacciDots(const Vec3& base_color, const Vec3& dot_color, 
                                      int dot_count = 12, float dot_radius = 0.33f) {
        MaterialDesc mat;
        mat.type = MaterialType::LAMBERTIAN;
        mat.albedo = base_color;
        mat.pattern = ProceduralPattern::FIBONACCI_DOTS;
        mat.pattern_color = dot_color;
        mat.pattern_param1 = static_cast<float>(dot_count);
        mat.pattern_param2 = dot_radius;
        return mat;
    }
};

//==============================================================================
// GEOMETRY SYSTEM
//==============================================================================

enum class GeometryType : uint8_t {
    SPHERE,
    RECTANGLE,
    CUBE,
    DISPLACED_SPHERE,    // Sphere with procedural displacement
    TRIANGLE,            // Single triangle
    TRIANGLE_MESH,       // Collection of triangles (uses BVH)
    SDF_PRIMITIVE        // Ray-marched signed distance field
};

/**
 * @brief Triangle for mesh representation
 */
struct Triangle {
    Vec3 v0, v1, v2;           // Vertices
    Vec3 n0, n1, n2;           // Vertex normals (for smooth shading)
    Vec3 uv0, uv1, uv2;        // Texture coordinates (future use)
    bool has_normals;          // True if normals are provided
    
    Triangle() : has_normals(false) {}
    
    Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2)
        : v0(v0), v1(v1), v2(v2), has_normals(false) {}
};

/**
 * @brief Triangle mesh with optional BVH
 */
struct TriangleMesh {
    std::vector<Triangle> triangles;
    int bvh_root_id;           // Index into BVH node array (-1 = not built)
    Vec3 bounds_min, bounds_max;
    
    TriangleMesh() : bvh_root_id(-1) {}
    
    // Factory methods (to be implemented)
    // static TriangleMesh fromOBJ(const std::string& filename);
    // static TriangleMesh cube(const Vec3& center, double size);
    // static TriangleMesh sphere(const Vec3& center, double radius, int subdivisions);
};

/**
 * @brief Signed Distance Field primitives for ray marching
 */
enum class SDFType : uint8_t {
    SPHERE,
    BOX,
    TORUS,
    CAPSULE,
    CYLINDER,
    PLANE,
    MANDELBULB,
    DEATH_STAR,
    CUT_HOLLOW_SPHERE,
    OCTAHEDRON,
    PYRAMID,
    CUSTOM         // User-defined distance function
};

struct SDFPrimitive {
    SDFType type;
    Vec3 position;
    Vec3 parameters;       // Size/shape parameters
    Vec3 rotation;         // Rotation in radians (pitch, yaw, roll / X, Y, Z axis)
    float blend_factor;    // For smooth blending operations
    int operation;         // Union, subtract, intersect
    
    SDFPrimitive() : type(SDFType::SPHERE), rotation(0, 0, 0), blend_factor(0.0f), operation(0) {}
};

/**
 * @brief Main geometry description - unified format for all geometry types
 */
struct GeometryDesc {
    GeometryType type;
    int material_id;       // Index into materials array
    
    // Geometry-specific data stored in union for memory efficiency
    union GeomData {
        // Sphere
        struct {
            Vec3 center;
            double radius;
        } sphere;
        
        // Rectangle (quad)
        struct {
            Vec3 corner;
            Vec3 u, v;      // Edge vectors
        } rectangle;
        
        // Cube
        struct {
            Vec3 center;
            double side_length;
            Vec3 rotation;  // Euler angles in degrees
        } cube;
        
        // Displaced sphere (golf ball, etc.)
        struct {
            Vec3 center;
            double radius;
            float displacement_scale;
            int pattern_type;
        } displaced_sphere;
        
        // Single triangle
        struct {
            Vec3 v0, v1, v2;
            Vec3 n0, n1, n2;
            bool has_normals;
        } triangle;
        
        // Mesh instance
        struct {
            int mesh_id;        // Index into mesh array
            Vec3 translation;
            Vec3 rotation;      // Euler angles
            Vec3 scale;
            bool use_bvh;       // Enable BVH acceleration for this instance
        } mesh_instance;
        
        // SDF primitive
        struct {
            SDFType sdf_type;
            Vec3 position;
            Vec3 parameters;
            Vec3 rotation;       // Rotation in radians (X, Y, Z axis)
            float max_distance;  // Ray marching limit
            float epsilon;       // Surface precision
        } sdf;
        
        GeomData() { memset(this, 0, sizeof(GeomData)); }
    } data;
    
    // Axis-aligned bounding box (for BVH)
    Vec3 bounds_min, bounds_max;
    
    GeometryDesc() : type(GeometryType::SPHERE), material_id(0) {
        bounds_min = Vec3(-1, -1, -1);
        bounds_max = Vec3(1, 1, 1);
    }
};

//==============================================================================
// BVH ACCELERATION STRUCTURE (for future implementation)
//==============================================================================

struct BVHNode {
    Vec3 bounds_min, bounds_max;  // Axis-aligned bounding box
    
    // Interior node: left/right child indices
    // Leaf node: geometry indices
    union NodeData {
        struct {
            int left_child;
            int right_child;
        } interior;
        
        struct {
            int first_geom_idx;
            int geom_count;
        } leaf;
        
        NodeData() { memset(this, 0, sizeof(NodeData)); }
    } data;
    
    bool is_leaf;
    uint8_t split_axis;  // 0=x, 1=y, 2=z
    
    BVHNode() : is_leaf(true), split_axis(0) {}
};

struct BVHTree {
    
    std::vector<BVHNode> nodes;
    int root_index;
    
    BVHTree() : root_index(-1) {}
    
    // Build methods (to be implemented in Phase 5)
    // void build(const std::vector<GeometryDesc>& geometries);
    // void buildSAH(const std::vector<GeometryDesc>& geometries);  // Surface Area Heuristic
};

//==============================================================================
// COMPLETE SCENE DESCRIPTION
//==============================================================================

/**
 * @brief Complete scene description - single source of truth for all renderers
 */
class SceneDescription {
public:
    std::vector<MaterialDesc> materials;
    std::vector<GeometryDesc> geometries;
    std::vector<TriangleMesh> meshes;
    std::vector<BVHTree> bvh_trees;            // Per-mesh BVH
    BVHTree top_level_bvh;                    // Scene-level BVH
    
    // Scene properties
    Vec3 camera_position;
    Vec3 camera_look_at;
    Vec3 camera_up;
    float camera_fov;
    
    // Rendering settings
    Vec3 background_color;
    float ambient_light;
    bool use_bvh;                             // Enable scene BVH
    
    // Constructor
    SceneDescription() 
        : camera_position(0, 0, 0)
        , camera_look_at(0, 0, -1)
        , camera_up(0, 1, 0)
        , camera_fov(90.0f)
        , background_color(0.5f, 0.7f, 1.0f)
        , ambient_light(0.1f)
        , use_bvh(false)
    {}
    
    //==========================================================================
    // BUILDER INTERFACE
    //==========================================================================
    
    /**
     * @brief Add a material to the scene
     * @return Material ID (index in materials array)
     */
    int addMaterial(const MaterialDesc& mat) {
        materials.push_back(mat);
        return static_cast<int>(materials.size() - 1);
    }
    
    /**
     * @brief Add geometry to the scene
     */
    void addGeometry(const GeometryDesc& geom) {
        geometries.push_back(geom);
    }
    
    /**
     * @brief Add a triangle mesh to the scene
     * @return Mesh ID (index in meshes array)
     */
    int addMesh(const TriangleMesh& mesh) {
        meshes.push_back(mesh);
        return static_cast<int>(meshes.size() - 1);
    }
    
    //==========================================================================
    // HIGH-LEVEL API - Convenience methods
    //==========================================================================
    
    void addSphere(const Vec3& center, double radius, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::SPHERE;
        geom.material_id = mat_id;
        geom.data.sphere.center = center;
        geom.data.sphere.radius = radius;
        
        // Compute bounding box
        Vec3 r(radius, radius, radius);
        geom.bounds_min = center - r;
        geom.bounds_max = center + r;
        
        geometries.push_back(geom);
    }
    
    void addDisplacedSphere(const Vec3& center, double radius, int mat_id, 
                           float displacement_scale = 0.2f, int pattern_type = 0) {
        GeometryDesc geom;
        geom.type = GeometryType::DISPLACED_SPHERE;
        geom.material_id = mat_id;
        geom.data.displaced_sphere.center = center;
        geom.data.displaced_sphere.radius = radius;
        geom.data.displaced_sphere.displacement_scale = displacement_scale;
        geom.data.displaced_sphere.pattern_type = pattern_type; // 0 = golf ball dimples
        
        // Compute bounding box (slightly larger due to displacement)
        Vec3 r(radius * 1.1, radius * 1.1, radius * 1.1);
        geom.bounds_min = center - r;
        geom.bounds_max = center + r;
        
        geometries.push_back(geom);
    }
    
    void addRectangle(const Vec3& corner, const Vec3& u, const Vec3& v, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::RECTANGLE;
        geom.material_id = mat_id;
        geom.data.rectangle.corner = corner;
        geom.data.rectangle.u = u;
        geom.data.rectangle.v = v;
        
        // Compute bounding box (all 4 corners)
        Vec3 corners[4] = {
            corner,
            corner + u,
            corner + v,
            corner + u + v
        };
        geom.bounds_min = corners[0];
        geom.bounds_max = corners[0];
        for (int i = 1; i < 4; i++) {
            geom.bounds_min = Vec3(
                std::min(geom.bounds_min.x(), corners[i].x()),
                std::min(geom.bounds_min.y(), corners[i].y()),
                std::min(geom.bounds_min.z(), corners[i].z())
            );
            geom.bounds_max = Vec3(
                std::max(geom.bounds_max.x(), corners[i].x()),
                std::max(geom.bounds_max.y(), corners[i].y()),
                std::max(geom.bounds_max.z(), corners[i].z())
            );
        }
        
        geometries.push_back(geom);
    }
    
    void addDisplacedSphere(const Vec3& center, double radius, int mat_id, int pattern_type = 0) {
        GeometryDesc geom;
        geom.type = GeometryType::DISPLACED_SPHERE;
        geom.material_id = mat_id;
        geom.data.displaced_sphere.center = center;
        geom.data.displaced_sphere.radius = radius;
        geom.data.displaced_sphere.displacement_scale = 0.05f;  // Default
        geom.data.displaced_sphere.pattern_type = pattern_type;
        
        // Compute bounding box (slightly larger for displacement)
        Vec3 r(radius * 1.1, radius * 1.1, radius * 1.1);
        geom.bounds_min = center - r;
        geom.bounds_max = center + r;
        
        geometries.push_back(geom);
    }
    
    void addTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::TRIANGLE;
        geom.material_id = mat_id;
        geom.data.triangle.v0 = v0;
        geom.data.triangle.v1 = v1;
        geom.data.triangle.v2 = v2;
        geom.data.triangle.has_normals = false;
        
        // Compute bounding box
        geom.bounds_min = Vec3(
            std::min(std::min(v0.x(), v1.x()), v2.x()),
            std::min(std::min(v0.y(), v1.y()), v2.y()),
            std::min(std::min(v0.z(), v1.z()), v2.z())
        );
        geom.bounds_max = Vec3(
            std::max(std::max(v0.x(), v1.x()), v2.x()),
            std::max(std::max(v0.y(), v1.y()), v2.y()),
            std::max(std::max(v0.z(), v1.z()), v2.z())
        );
        
        geometries.push_back(geom);
    }
    
    void addMeshInstance(int mesh_id, const Vec3& pos, const Vec3& rot, const Vec3& scale, int mat_id) {
        GeometryDesc geom;
        geom.type = GeometryType::TRIANGLE_MESH;
        geom.material_id = mat_id;
        geom.data.mesh_instance.mesh_id = mesh_id;
        geom.data.mesh_instance.translation = pos;
        geom.data.mesh_instance.rotation = rot;
        geom.data.mesh_instance.scale = scale;
        geom.data.mesh_instance.use_bvh = true;
        
        // TODO: Compute transformed bounding box
        geom.bounds_min = Vec3(-1, -1, -1);
        geom.bounds_max = Vec3(1, 1, 1);
        
        geometries.push_back(geom);
    }
    
    void addSDFPrimitive(SDFType type, const Vec3& pos, const Vec3& params, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        GeometryDesc geom;
        geom.type = GeometryType::SDF_PRIMITIVE;
        geom.material_id = mat_id;
        geom.data.sdf.sdf_type = type;
        geom.data.sdf.position = pos;
        geom.data.sdf.parameters = params;
        geom.data.sdf.rotation = rotation;
        geom.data.sdf.max_distance = 10.0f;  // Default ray march distance
        geom.data.sdf.epsilon = 0.001f;       // Default surface precision
        
        // TODO: Compute proper bounding box for SDF
        geom.bounds_min = pos - params;
        geom.bounds_max = pos + params;
        
        geometries.push_back(geom);
    }
    
    // Convenient SDF shape factory methods
    
    void addSDFSphere(const Vec3& center, double radius, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(radius, 0, 0);
        addSDFPrimitive(SDFType::SPHERE, center, params, mat_id, rotation);
    }
    
    void addSDFBox(const Vec3& center, const Vec3& half_extents, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        addSDFPrimitive(SDFType::BOX, center, half_extents, mat_id, rotation);
    }
    
    void addSDFTorus(const Vec3& center, double major_radius, double minor_radius, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(major_radius, minor_radius, 0);
        addSDFPrimitive(SDFType::TORUS, center, params, mat_id, rotation);
    }
    
    void addSDFCapsule(const Vec3& center, double radius, double height, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(radius, height, 0);
        addSDFPrimitive(SDFType::CAPSULE, center, params, mat_id, rotation);
    }
    
    void addSDFMandelbulb(const Vec3& center, double power = 8.0, int iterations = 15, int mat_id = 0, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(power, static_cast<double>(iterations), 0);
        addSDFPrimitive(SDFType::MANDELBULB, center, params, mat_id, rotation);
    }
    
    void addSDFDeathStar(const Vec3& center, double main_radius, double cutout_radius, double cutout_distance, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(main_radius, cutout_radius, cutout_distance);
        addSDFPrimitive(SDFType::DEATH_STAR, center, params, mat_id, rotation);
    }
    
    void addSDFCutHollowSphere(const Vec3& center, double radius, double cut_height, double thickness, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(radius, cut_height, thickness);
        addSDFPrimitive(SDFType::CUT_HOLLOW_SPHERE, center, params, mat_id, rotation);
    }
    
    void addSDFOctahedron(const Vec3& center, double size, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(size, 0, 0);
        addSDFPrimitive(SDFType::OCTAHEDRON, center, params, mat_id, rotation);
    }
    
    void addSDFPyramid(const Vec3& center, double height, int mat_id, const Vec3& rotation = Vec3(0, 0, 0)) {
        Vec3 params(height, 0, 0);
        addSDFPrimitive(SDFType::PYRAMID, center, params, mat_id, rotation);
    }
    
    //==========================================================================
    // BVH CONSTRUCTION (Phase 5)
    //==========================================================================
    
    void buildBVH() {
        if (geometries.empty()) {
            top_level_bvh.root_index = -1;
            return;
        }
        
        use_bvh = true;
        
        // Create list of geometry indices
        std::vector<int> geom_indices(geometries.size());
        for (size_t i = 0; i < geometries.size(); ++i) {
            geom_indices[i] = static_cast<int>(i);
        }
        
        // Recursively build BVH
        top_level_bvh.nodes.clear();
        top_level_bvh.root_index = buildBVHRecursive(geom_indices, 0, static_cast<int>(geom_indices.size()));
        
        // Remap geometries to match BVH leaf order for optimal memory access
        std::vector<GeometryDesc> reordered_geometries;
        reordered_geometries.reserve(geometries.size());
        
        // Traverse BVH and collect geometry in leaf order
        std::vector<int> old_to_new(geometries.size());
        int new_idx = 0;
        
        std::function<void(int)> collectGeometry = [&](int node_idx) {
            const BVHNode& node = top_level_bvh.nodes[node_idx];
            if (node.is_leaf) {
                // Store geometries and update mapping
                for (int i = 0; i < node.data.leaf.geom_count; ++i) {
                    int old_idx = geom_indices[node.data.leaf.first_geom_idx + i];
                    old_to_new[old_idx] = new_idx;
                    reordered_geometries.push_back(geometries[old_idx]);
                    new_idx++;
                }
            } else {
                collectGeometry(node.data.interior.left_child);
                collectGeometry(node.data.interior.right_child);
            }
        };
        
        collectGeometry(top_level_bvh.root_index);
        
        // Update leaf node indices to point to reordered array positions
        new_idx = 0;
        std::function<void(int)> updateLeafIndices = [&](int node_idx) {
            BVHNode& node = top_level_bvh.nodes[node_idx];
            if (node.is_leaf) {
                node.data.leaf.first_geom_idx = new_idx;
                new_idx += node.data.leaf.geom_count;
            } else {
                updateLeafIndices(node.data.interior.left_child);
                updateLeafIndices(node.data.interior.right_child);
            }
        };
        
        updateLeafIndices(top_level_bvh.root_index);
        
        // Replace geometry array with reordered version
        geometries = std::move(reordered_geometries);
    }
    
    void buildMeshBVHs() {
        // To be implemented in future for per-mesh acceleration
        // This will build BVH for each triangle mesh
    }

private:
    /**
     * @brief Recursively build BVH using Surface Area Heuristic (SAH)
     * @param geom_indices Indices of geometries to partition
     * @param start Start index in geom_indices
     * @param end End index (exclusive) in geom_indices
     * @return Index of created node in BVH tree
     */
    int buildBVHRecursive(std::vector<int>& geom_indices, int start, int end) {
        BVHNode node;
        
        // Compute bounding box for all geometries in range
        node.bounds_min = Vec3(1e30, 1e30, 1e30);
        node.bounds_max = Vec3(-1e30, -1e30, -1e30);
        
        for (int i = start; i < end; ++i) {
            const GeometryDesc& geom = geometries[geom_indices[i]];
            node.bounds_min = Vec3(
                std::min(node.bounds_min.x(), geom.bounds_min.x()),
                std::min(node.bounds_min.y(), geom.bounds_min.y()),
                std::min(node.bounds_min.z(), geom.bounds_min.z())
            );
            node.bounds_max = Vec3(
                std::max(node.bounds_max.x(), geom.bounds_max.x()),
                std::max(node.bounds_max.y(), geom.bounds_max.y()),
                std::max(node.bounds_max.z(), geom.bounds_max.z())
            );
        }
        
        int count = end - start;
        
        // Leaf node: few geometries
        if (count <= 4) {
            node.is_leaf = true;
            node.data.leaf.first_geom_idx = start;
            node.data.leaf.geom_count = count;
            
            int node_idx = static_cast<int>(top_level_bvh.nodes.size());
            top_level_bvh.nodes.push_back(node);
            return node_idx;
        }
        
        // Interior node: find best split using SAH
        float best_cost = 1e30f;
        int best_axis = 0;
        int best_split_idx = start + count / 2;
        
        Vec3 extent = node.bounds_max - node.bounds_min;
        float parent_area = 2.0f * (extent.x() * extent.y() + extent.y() * extent.z() + extent.z() * extent.x());
        
        // Try each axis
        for (int axis = 0; axis < 3; ++axis) {
            // Sort geometries along this axis by centroid
            std::sort(geom_indices.begin() + start, geom_indices.begin() + end,
                [this, axis](int a, int b) {
                    Vec3 ca = (geometries[a].bounds_min + geometries[a].bounds_max) * 0.5;
                    Vec3 cb = (geometries[b].bounds_min + geometries[b].bounds_max) * 0.5;
                    return ca[axis] < cb[axis];
                });
            
            // Try different split positions using SAH
            for (int i = start + 1; i < end; ++i) {
                // Compute left and right bounding boxes
                Vec3 left_min(1e30, 1e30, 1e30), left_max(-1e30, -1e30, -1e30);
                Vec3 right_min(1e30, 1e30, 1e30), right_max(-1e30, -1e30, -1e30);
                
                for (int j = start; j < i; ++j) {
                    const GeometryDesc& geom = geometries[geom_indices[j]];
                    left_min = Vec3(std::min(left_min.x(), geom.bounds_min.x()),
                                   std::min(left_min.y(), geom.bounds_min.y()),
                                   std::min(left_min.z(), geom.bounds_min.z()));
                    left_max = Vec3(std::max(left_max.x(), geom.bounds_max.x()),
                                   std::max(left_max.y(), geom.bounds_max.y()),
                                   std::max(left_max.z(), geom.bounds_max.z()));
                }
                
                for (int j = i; j < end; ++j) {
                    const GeometryDesc& geom = geometries[geom_indices[j]];
                    right_min = Vec3(std::min(right_min.x(), geom.bounds_min.x()),
                                    std::min(right_min.y(), geom.bounds_min.y()),
                                    std::min(right_min.z(), geom.bounds_min.z()));
                    right_max = Vec3(std::max(right_max.x(), geom.bounds_max.x()),
                                    std::max(right_max.y(), geom.bounds_max.y()),
                                    std::max(right_max.z(), geom.bounds_max.z()));
                }
                
                // Compute surface areas
                Vec3 left_extent = left_max - left_min;
                Vec3 right_extent = right_max - right_min;
                float left_area = 2.0f * (left_extent.x() * left_extent.y() + 
                                         left_extent.y() * left_extent.z() + 
                                         left_extent.z() * left_extent.x());
                float right_area = 2.0f * (right_extent.x() * right_extent.y() + 
                                          right_extent.y() * right_extent.z() + 
                                          right_extent.z() * right_extent.x());
                
                // SAH cost: C_traverse + P_left * C_left + P_right * C_right
                int left_count = i - start;
                int right_count = end - i;
                float cost = 1.0f + (left_area / parent_area) * left_count + 
                                   (right_area / parent_area) * right_count;
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_split_idx = i;
                }
            }
        }
        
        // Sort by best axis for final split
        std::sort(geom_indices.begin() + start, geom_indices.begin() + end,
            [this, best_axis](int a, int b) {
                Vec3 ca = (geometries[a].bounds_min + geometries[a].bounds_max) * 0.5;
                Vec3 cb = (geometries[b].bounds_min + geometries[b].bounds_max) * 0.5;
                return ca[best_axis] < cb[best_axis];
            });
        
        // Create interior node
        node.is_leaf = false;
        node.split_axis = static_cast<uint8_t>(best_axis);
        
        int node_idx = static_cast<int>(top_level_bvh.nodes.size());
        top_level_bvh.nodes.push_back(node);
        
        // Recursively build children
        int left_child = buildBVHRecursive(geom_indices, start, best_split_idx);
        int right_child = buildBVHRecursive(geom_indices, best_split_idx, end);
        
        // Update node with child indices
        top_level_bvh.nodes[node_idx].data.interior.left_child = left_child;
        top_level_bvh.nodes[node_idx].data.interior.right_child = right_child;
        
        return node_idx;
    }

public:
    
    //==========================================================================
    // SERIALIZATION (Future)
    //==========================================================================
    
    bool saveToFile(const std::string& filename) const {
        // To be implemented
        return false;
    }
    
    bool loadFromFile(const std::string& filename) {
        // To be implemented
        return false;
    }
    
    //==========================================================================
    // VALIDATION
    //==========================================================================
    
    bool validate() const {
        // Check material IDs are valid
        for (const auto& geom : geometries) {
            if (geom.material_id < 0 || geom.material_id >= static_cast<int>(materials.size())) {
                return false;
            }
        }
        
        // Check mesh IDs are valid
        for (const auto& geom : geometries) {
            if (geom.type == GeometryType::TRIANGLE_MESH) {
                int mesh_id = geom.data.mesh_instance.mesh_id;
                if (mesh_id < 0 || mesh_id >= static_cast<int>(meshes.size())) {
                    return false;
                }
            }
        }
        
        return true;
    }
};

} // namespace Scene
