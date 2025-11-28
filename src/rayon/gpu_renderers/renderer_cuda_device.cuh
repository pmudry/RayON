// Forward declaration for CudaScene::Scene
namespace CudaScene
{
struct Scene;
}

// Forward declarations of CUDA functions
extern "C"
{
   // Host function for accumulative CUDA rendering (used for both one-shot and progressive rendering)
   // For one-shot rendering: call once with samples_to_add = total samples, total_samples_so_far = 0
   // For progressive rendering: call multiple times, incrementing total_samples_so_far each time
   unsigned long long renderPixelsCUDAAccumulative(
       unsigned char *image, float *accum_buffer, CudaScene::Scene *scene, int width, int height, double cam_center_x,
       double cam_center_y, double cam_center_z, double pixel00_x, double pixel00_y, double pixel00_z, double delta_u_x,
       double delta_u_y, double delta_u_z, double delta_v_x, double delta_v_y, double delta_v_z, int samples_to_add,
       int total_samples_so_far, int max_depth, void **d_rand_states_ptr, void **d_accum_buffer_ptr, double cam_u_x,
       double cam_u_y, double cam_u_z, double cam_v_x, double cam_v_y, double cam_v_z);

   // Helper to free device random states
   void freeDeviceRandomStates(void *d_rand_states);

   // Helper to free device accumulation buffer
   void freeDeviceAccumBuffer(void *d_accum_buffer);

   // Set global light intensity (affects area light emission)
   void setLightIntensity(float intensity);

   // Set background gradient intensity (sky brightness)
   void setBackgroundIntensity(float intensity);

   // Set metal roughness/fuzziness multiplier
   void setMetalFuzziness(float fuzziness);

   // Set glass refraction index
   void setGlassRefractionIndex(float index);

   // Depth of Field controls
   void setDOFEnabled(bool enabled);
   void setDOFAperture(float aperture);
   void setDOFFocusDistance(float distance);
}