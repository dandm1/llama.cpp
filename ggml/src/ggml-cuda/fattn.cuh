#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// bytes of pool memory one launch of this op needs on this device, see ggml_backend_cuda_device_get_op_scratch_size
size_t ggml_cuda_flash_attn_ext_scratch_size(int device, const ggml_tensor * dst);
