/**
 * @file volume_sdf.cu
 * @brief Signed distance grid of a closed mesh (see volume_sdf.cuh)
 */

#include "volume_sdf.cuh"
#include "volume_medium.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{

__device__ __forceinline__ float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 mul(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float  dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b)
{
   return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)
__device__ float pointTriangleDist2(float3 p, float3 a, float3 b, float3 c)
{
   const float3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
   const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
   if (d1 <= 0.0f && d2 <= 0.0f)
      return dot3(ap, ap);
   const float3 bp = sub(p, b);
   const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
   if (d3 >= 0.0f && d4 <= d3)
      return dot3(bp, bp);
   const float vc = d1 * d4 - d3 * d2;
   if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
   {
      const float  v = d1 / (d1 - d3);
      const float3 q = add(a, mul(ab, v));
      const float3 d = sub(p, q);
      return dot3(d, d);
   }
   const float3 cp = sub(p, c);
   const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
   if (d6 >= 0.0f && d5 <= d6)
      return dot3(cp, cp);
   const float vb = d5 * d2 - d1 * d6;
   if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
   {
      const float  w = d2 / (d2 - d6);
      const float3 q = add(a, mul(ac, w));
      const float3 d = sub(p, q);
      return dot3(d, d);
   }
   const float va = d3 * d6 - d5 * d4;
   if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
   {
      const float  w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      const float3 q = add(b, mul(sub(c, b), w));
      const float3 d = sub(p, q);
      return dot3(d, d);
   }
   const float denom = 1.0f / (va + vb + vc);
   const float v = vb * denom, w = vc * denom;
   const float3 q = add(a, add(mul(ab, v), mul(ac, w)));
   const float3 d = sub(p, q);
   return dot3(d, d);
}

// Does the ray p + t (1,0,0), t > 0, cross triangle abc? (Moller-Trumbore, direction = +x)
__device__ bool rayXCrossesTriangle(float3 p, float3 a, float3 b, float3 c)
{
   const float3 e1 = sub(b, a), e2 = sub(c, a);
   // pvec = dir x e2 with dir = (1,0,0): (0, -e2.z, e2.y)
   const float3 pvec = make_float3(0.0f, -e2.z, e2.y);
   const float  det  = dot3(e1, pvec);
   if (fabsf(det) < 1e-12f)
      return false;
   const float  inv  = 1.0f / det;
   const float3 tvec = sub(p, a);
   const float  u    = dot3(tvec, pvec) * inv;
   if (u < 0.0f || u > 1.0f)
      return false;
   const float3 qvec = cross3(tvec, e1);
   const float  v    = qvec.x * inv; // dot(dir, qvec)
   if (v < 0.0f || u + v > 1.0f)
      return false;
   const float t = dot3(e2, qvec) * inv;
   return t > 0.0f;
}

constexpr int TILE = 256;

__global__ void sdfKernel(const float3 *__restrict__ tris, int num_tris, float3 bmin, float3 cell, int3 dims,
                          float *__restrict__ out)
{
   __shared__ float3 s_tris[TILE * 3];
   const int n_vox = dims.x * dims.y * dims.z;
   const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
   const bool live = idx < n_vox;
   float3 p = make_float3(0.0f, 0.0f, 0.0f);
   if (live)
   {
      const int ix = idx % dims.x, iy = (idx / dims.x) % dims.y, iz = idx / (dims.x * dims.y);
      p = make_float3(bmin.x + (ix + 0.5f) * cell.x, bmin.y + (iy + 0.5f) * cell.y, bmin.z + (iz + 0.5f) * cell.z);
   }
   float min_d2 = 1e30f;
   int   crossings = 0;
   for (int base = 0; base < num_tris; base += TILE)
   {
      const int count = min(TILE, num_tris - base);
      for (int k = threadIdx.x; k < count * 3; k += blockDim.x)
         s_tris[k] = tris[(base * 3) + k];
      __syncthreads();
      if (live)
      {
         for (int t = 0; t < count; ++t)
         {
            const float3 a = s_tris[3 * t], b = s_tris[3 * t + 1], c = s_tris[3 * t + 2];
            min_d2 = fminf(min_d2, pointTriangleDist2(p, a, b, c));
            if (rayXCrossesTriangle(p, a, b, c))
               ++crossings;
         }
      }
      __syncthreads();
   }
   if (live)
      out[idx] = sqrtf(min_d2) * ((crossings & 1) ? -1.0f : 1.0f);
}

__global__ void bakeFieldKernel(VolumeParams vp, int3 dims, float *__restrict__ out)
{
   const int n_vox = dims.x * dims.y * dims.z;
   const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= n_vox)
      return;
   const int ix = idx % dims.x, iy = (idx / dims.x) % dims.y, iz = idx / (dims.x * dims.y);
   // Voxel centres span the SDF box exactly, so both textures share the same normalised coordinates
   const float x = vp.sdf_min.x + ((float)ix + 0.5f) / (dims.x * vp.sdf_inv_extent.x);
   const float y = vp.sdf_min.y + ((float)iy + 0.5f) / (dims.y * vp.sdf_inv_extent.y);
   const float z = vp.sdf_min.z + ((float)iz + 0.5f) / (dims.z * vp.sdf_inv_extent.z);
   out[idx] = volume_field_procedural(vp, x, y, z);
}

// Upload a float grid as a 3D texture (linear filtering, clamped); reuses the array when the
// dimensions did not change.
static bool uploadGridTexture(const float *d_grid, int3 dims, cudaTextureObject_t &tex, cudaArray_t &arr,
                              int3 &cur_dims)
{
   if (arr && (cur_dims.x != dims.x || cur_dims.y != dims.y || cur_dims.z != dims.z))
   {
      if (tex)
         cudaDestroyTextureObject(tex);
      cudaFreeArray(arr);
      tex = 0;
      arr = nullptr;
   }
   if (!arr)
   {
      cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
      if (cudaMalloc3DArray(&arr, &fmt, make_cudaExtent(dims.x, dims.y, dims.z)) != cudaSuccess)
      {
         fprintf(stderr, "Volume field: 3D array allocation failed\n");
         arr = nullptr;
         return false;
      }
   }
   cudaMemcpy3DParms copy = {};
   copy.srcPtr   = make_cudaPitchedPtr(const_cast<float *>(d_grid), dims.x * sizeof(float), dims.x, dims.y);
   copy.dstArray = arr;
   copy.extent   = make_cudaExtent(dims.x, dims.y, dims.z);
   copy.kind     = cudaMemcpyDeviceToDevice;
   cudaMemcpy3D(&copy);
   if (!tex)
   {
      cudaResourceDesc res = {};
      res.resType         = cudaResourceTypeArray;
      res.res.array.array = arr;
      cudaTextureDesc td = {};
      td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
      td.filterMode       = cudaFilterModeLinear;
      td.readMode         = cudaReadModeElementType;
      td.normalizedCoords = 1;
      if (cudaCreateTextureObject(&tex, &res, &td, nullptr) != cudaSuccess)
      {
         fprintf(stderr, "Volume field: texture creation failed\n");
         tex = 0;
         return false;
      }
   }
   cur_dims = dims;
   return true;
}

} // namespace

void bakeVolumeField(VolumeParams &vp, int max_cells)
{
   if (vp.sdf == 0 || max_cells < 4)
      return;
   const float3 ext = make_float3(1.0f / vp.sdf_inv_extent.x, 1.0f / vp.sdf_inv_extent.y, 1.0f / vp.sdf_inv_extent.z);
   const float  longest = std::max(ext.x, std::max(ext.y, ext.z));
   const int3   dims    = make_int3(std::max(2, (int)std::lround(ext.x / longest * max_cells)),
                                    std::max(2, (int)std::lround(ext.y / longest * max_cells)),
                                    std::max(2, (int)std::lround(ext.z / longest * max_cells)));
   const size_t n_vox = (size_t)dims.x * dims.y * dims.z;
   float *d_grid = nullptr;
   if (cudaMalloc(&d_grid, n_vox * sizeof(float)) != cudaSuccess)
      return;
   VolumeParams proc = vp;
   proc.field = 0; // bake from the procedural model, never from a previous bake
   bakeFieldKernel<<<(unsigned)((n_vox + 255) / 256), 256>>>(proc, dims, d_grid);
   if (cudaDeviceSynchronize() == cudaSuccess)
      uploadGridTexture(d_grid, dims, vp.field, vp.field_array, vp.field_dims);
   cudaFree(d_grid);
}

bool buildVolumeSDF(const std::vector<float3> &tris, int max_cells, float pad, VolumeParams &vp)
{
   vp.sdf         = 0;
   vp.sdf_array   = nullptr;
   vp.field       = 0;
   vp.field_array = nullptr;
   vp.field_dims  = make_int3(0, 0, 0);
   if (tris.size() < 3 || max_cells < 4)
      return false;
   const int num_tris = static_cast<int>(tris.size() / 3);

   float3 lo = tris[0], hi = tris[0];
   for (const float3 &v : tris)
   {
      lo = make_float3(std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z));
      hi = make_float3(std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z));
   }
   pad = std::max(pad, 0.0f);
   const float3 ext = make_float3(hi.x - lo.x + 2.0f * pad, hi.y - lo.y + 2.0f * pad, hi.z - lo.z + 2.0f * pad);
   const float  longest = std::max(ext.x, std::max(ext.y, ext.z));
   if (longest <= 0.0f)
      return false;
   const float cell_size = longest / static_cast<float>(max_cells);
   // The grid covers the mesh plus `pad` on every side, plus one cell so the border is sampled
   const float3 bmin = make_float3(lo.x - pad - cell_size, lo.y - pad - cell_size, lo.z - pad - cell_size);
   const int3   dims = make_int3(static_cast<int>(std::ceil(ext.x / cell_size)) + 3,
                                 static_cast<int>(std::ceil(ext.y / cell_size)) + 3,
                                 static_cast<int>(std::ceil(ext.z / cell_size)) + 3);
   const float3 cell = make_float3(cell_size, cell_size, cell_size);
   const float3 bmax = make_float3(bmin.x + dims.x * cell_size, bmin.y + dims.y * cell_size, bmin.z + dims.z * cell_size);
   const size_t n_vox = static_cast<size_t>(dims.x) * dims.y * dims.z;

   float3 *d_tris = nullptr;
   float  *d_out  = nullptr;
   if (cudaMalloc(&d_tris, tris.size() * sizeof(float3)) != cudaSuccess ||
       cudaMalloc(&d_out, n_vox * sizeof(float)) != cudaSuccess)
   {
      fprintf(stderr, "Volume SDF: allocation failed\n");
      cudaFree(d_tris);
      cudaFree(d_out);
      return false;
   }
   cudaMemcpy(d_tris, tris.data(), tris.size() * sizeof(float3), cudaMemcpyHostToDevice);
   sdfKernel<<<static_cast<unsigned>((n_vox + TILE - 1) / TILE), TILE>>>(d_tris, num_tris, bmin, cell, dims, d_out);
   cudaError_t err = cudaDeviceSynchronize();
   cudaFree(d_tris);
   if (err != cudaSuccess)
   {
      fprintf(stderr, "Volume SDF: kernel failed: %s\n", cudaGetErrorString(err));
      cudaFree(d_out);
      return false;
   }

   // 3D texture: linear filtering, clamped (outside the box the distance stays positive)
   cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
   cudaArray_t arr = nullptr;
   if (cudaMalloc3DArray(&arr, &fmt, make_cudaExtent(dims.x, dims.y, dims.z)) != cudaSuccess)
   {
      fprintf(stderr, "Volume SDF: 3D array allocation failed\n");
      cudaFree(d_out);
      return false;
   }
   cudaMemcpy3DParms copy = {};
   copy.srcPtr   = make_cudaPitchedPtr(d_out, dims.x * sizeof(float), dims.x, dims.y);
   copy.dstArray = arr;
   copy.extent   = make_cudaExtent(dims.x, dims.y, dims.z);
   copy.kind     = cudaMemcpyDeviceToDevice;
   cudaMemcpy3D(&copy);
   cudaFree(d_out);

   cudaResourceDesc res = {};
   res.resType         = cudaResourceTypeArray;
   res.res.array.array = arr;
   cudaTextureDesc tex = {};
   tex.addressMode[0] = tex.addressMode[1] = tex.addressMode[2] = cudaAddressModeClamp;
   tex.filterMode       = cudaFilterModeLinear;
   tex.readMode         = cudaReadModeElementType;
   tex.normalizedCoords = 1;
   cudaTextureObject_t tex_obj = 0;
   if (cudaCreateTextureObject(&tex_obj, &res, &tex, nullptr) != cudaSuccess)
   {
      fprintf(stderr, "Volume SDF: texture creation failed\n");
      cudaFreeArray(arr);
      return false;
   }
   vp.sdf            = tex_obj;
   vp.sdf_array      = arr;
   vp.sdf_min        = bmin;
   vp.sdf_inv_extent = make_float3(1.0f / (bmax.x - bmin.x), 1.0f / (bmax.y - bmin.y), 1.0f / (bmax.z - bmin.z));
   printf("Volume SDF: %d triangles -> %dx%dx%d grid (cell %.3f)\n", num_tris, dims.x, dims.y, dims.z, cell_size);
   return true;
}

void freeVolumeSDF(VolumeParams &vp)
{
   if (vp.sdf)
      cudaDestroyTextureObject(vp.sdf);
   if (vp.sdf_array)
      cudaFreeArray(vp.sdf_array);
   vp.sdf       = 0;
   vp.sdf_array = nullptr;
   if (vp.field)
      cudaDestroyTextureObject(vp.field);
   if (vp.field_array)
      cudaFreeArray(vp.field_array);
   vp.field       = 0;
   vp.field_array = nullptr;
}
