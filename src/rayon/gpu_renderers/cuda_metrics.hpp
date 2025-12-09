#pragma once

#include <string>
#include <cstddef>

// Helper function to get CUDA device metrics without exposing CUDA headers to the host compiler
void getCudaDeviceMetrics(std::string& name, size_t& vram_used);
