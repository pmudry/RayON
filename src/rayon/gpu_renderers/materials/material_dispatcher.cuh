/**
 * @file material_dispatcher.cuh
 * @brief Compile-time material dispatcher using template metaprogramming
 *
 * Provides zero-overhead material dispatch by generating separate code paths
 * for each material type at compile time. The switch statement is optimized
 * to a jump table by the compiler.
 */

#pragma once
#include "cuda_raytracer.cuh"
#include "legacy/constant.cuh"
#include "legacy/glass.cuh"
#include "legacy/lambertian.cuh"
#include "legacy/light.cuh"
#include "legacy/mirror.cuh"
#include "legacy/rough_mirror.cuh"
#include "legacy/show_normals.cuh"
#include "legacy/thin_film.cuh"
#include "legacy/clear_coat.cuh"
#include "material_base.cuh"

namespace Materials
{

//==============================================================================
// MATERIAL PARAMETER UNION (POD type for GPU transfer)
//==============================================================================

/**
 * @brief Union of all material parameter types
 *
 * This is a POD (Plain Old Data) type that can be efficiently transferred
 * to GPU memory. Only one parameter struct is active at a time.
 */
union MaterialParamsUnion
{
   LambertianParams lambertian;
   MirrorParams mirror;
   RoughMirrorParams rough_mirror;
   GlassParams glass;
   LightParams light;
   ConstantParams constant;
   ShowNormalsParams show_normals;
   ThinFilmParams thin_film;
   ClearCoatParams clear_coat;

   __device__ __host__ MaterialParamsUnion() {}
};

//==============================================================================
// MATERIAL DESCRIPTOR
//==============================================================================

/**
 * @brief Complete material descriptor for GPU transfer
 *
 * Combines material type tag with parameters.
 * This is what gets stored in the scene's material array.
 */
struct MaterialDescriptor
{
   LegacyMaterialType type;    // Material type enum
   MaterialParamsUnion params; // Material-specific parameters

   __device__ __host__ MaterialDescriptor() : type(LAMBERTIAN) {}

   // Factory methods for creating material descriptors

   __device__ __host__ static MaterialDescriptor makeLambertian(const f3 &albedo)
   {
      MaterialDescriptor desc;
      desc.type = LAMBERTIAN;
      desc.params.lambertian.albedo = albedo;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeMirror(const f3 &albedo)
   {
      MaterialDescriptor desc;
      desc.type = MIRROR;
      desc.params.mirror.albedo = albedo;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeRoughMirror(const f3 &albedo, float roughness)
   {
      MaterialDescriptor desc;
      desc.type = ROUGH_MIRROR;
      desc.params.rough_mirror.albedo = albedo;
      desc.params.rough_mirror.roughness = roughness;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeGlass(float refractive_index)
   {
      MaterialDescriptor desc;
      desc.type = GLASS;
      desc.params.glass.refractive_index = refractive_index;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeLight(const f3 &emission)
   {
      MaterialDescriptor desc;
      desc.type = LIGHT;
      desc.params.light.emission = emission;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeConstant(const f3 &color)
   {
      MaterialDescriptor desc;
      desc.type = CONSTANT;
      desc.params.constant.color = color;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeShowNormals(const f3 &normal = f3(0, 0, 0))
   {
      MaterialDescriptor desc;
      desc.type = SHOW_NORMALS;
      desc.params.show_normals.normal = normal;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeThinFilm(float thickness, float film_ior, float exterior_ior)
   {
      MaterialDescriptor desc;
      desc.type = THIN_FILM;
      desc.params.thin_film.film_thickness = thickness;
      desc.params.thin_film.film_ior = film_ior;
      desc.params.thin_film.exterior_ior = exterior_ior;
      return desc;
   }

   __device__ __host__ static MaterialDescriptor makeClearCoat(const f3 &albedo, float roughness, float coat_ior)
   {
      MaterialDescriptor desc;
      desc.type = CLEAR_COAT;
      desc.params.clear_coat.albedo = albedo;
      desc.params.clear_coat.roughness = roughness;
      desc.params.clear_coat.coat_ior = coat_ior;
      return desc;
   }
};

//==============================================================================
// COMPILE-TIME MATERIAL DISPATCHER
//==============================================================================

/**
 * @brief Dispatch to appropriate material type at compile time
 *
 * This function takes a MaterialDescriptor and a callable (lambda/functor)
 * and invokes the callable with the appropriate strongly-typed material.
 *
 * The switch statement is optimized to a jump table by NVCC with -O3,
 * making this essentially zero-cost compared to manual if-else chains.
 *
 * Template magic:
 * - Func is deduced from the lambda/functor passed
 * - decltype determines return type from the callable's return type
 * - Each case instantiates the callable with a different material type
 * - Compiler generates optimized monomorphic code for each branch
 *
 * @tparam Func Callable type (deduced from lambda)
 * @param desc Material descriptor containing type and parameters
 * @param func Callable object to invoke with the material instance
 * @return Result of func(material)
 *
 * Example usage:
 * @code
 * auto result = dispatch_material(desc, [&](auto material) {
 *     using MaterialType = decltype(material);
 *     // Use material.scatter(), material.emission(), etc.
 *     return some_value;
 * });
 * @endcode
 */
template <typename Func>
__device__ __forceinline__ auto dispatch_material(const MaterialDescriptor &desc, Func &&func)
    -> decltype(func(Lambertian(desc.params.lambertian)))
{
   switch (desc.type)
   {
   case LAMBERTIAN:
      return func(Lambertian(desc.params.lambertian));

   case MIRROR:
      return func(Mirror(desc.params.mirror));

   case ROUGH_MIRROR:
      return func(RoughMirror(desc.params.rough_mirror));

   case GLASS:
      return func(Glass(desc.params.glass));

   case LIGHT:
      return func(Light(desc.params.light));

   case CONSTANT:
      return func(Constant(desc.params.constant));

   case SHOW_NORMALS:
      return func(ShowNormals(desc.params.show_normals));

   case THIN_FILM:
      return func(ThinFilm(desc.params.thin_film));

   case CLEAR_COAT:
      return func(ClearCoat(desc.params.clear_coat));

   default:
      // Fallback to Lambertian (shouldn't happen in correct usage)
      return func(Lambertian(desc.params.lambertian));
   }
}

/**
 * @brief Simplified dispatcher that returns bool (for scatter operations)
 *
 * Optimized version when you know the return type is bool.
 * Slightly more efficient than the generic version.
 */
template <typename Func>
__device__ __forceinline__ bool dispatch_material_bool(const MaterialDescriptor &desc, Func &&func)
{
   switch (desc.type)
   {
   case LAMBERTIAN:
      return func(Lambertian(desc.params.lambertian));
   case MIRROR:
      return func(Mirror(desc.params.mirror));
   case ROUGH_MIRROR:
      return func(RoughMirror(desc.params.rough_mirror));
   case GLASS:
      return func(Glass(desc.params.glass));
   case LIGHT:
      return func(Light(desc.params.light));
   case CONSTANT:
      return func(Constant(desc.params.constant));
   case SHOW_NORMALS:
      return func(ShowNormals(desc.params.show_normals));
   case THIN_FILM:
      return func(ThinFilm(desc.params.thin_film));
   case CLEAR_COAT:
      return func(ClearCoat(desc.params.clear_coat));
   default:
      return func(Lambertian(desc.params.lambertian));
   }
}

/**
 * @brief Simplified dispatcher that returns f3 (for emission operations)
 *
 * Optimized version when you know the return type is f3.
 */
template <typename Func> __device__ __forceinline__ f3 dispatch_material_f3(const MaterialDescriptor &desc, Func &&func)
{
   switch (desc.type)
   {
   case LAMBERTIAN:
      return func(Lambertian(desc.params.lambertian));
   case MIRROR:
      return func(Mirror(desc.params.mirror));
   case ROUGH_MIRROR:
      return func(RoughMirror(desc.params.rough_mirror));
   case GLASS:
      return func(Glass(desc.params.glass));
   case LIGHT:
      return func(Light(desc.params.light));
   case CONSTANT:
      return func(Constant(desc.params.constant));
   case SHOW_NORMALS:
      return func(ShowNormals(desc.params.show_normals));
   case THIN_FILM:
      return func(ThinFilm(desc.params.thin_film));
   case CLEAR_COAT:
      return func(ClearCoat(desc.params.clear_coat));
   default:
      return func(Lambertian(desc.params.lambertian));
   }
}

} // namespace Materials
