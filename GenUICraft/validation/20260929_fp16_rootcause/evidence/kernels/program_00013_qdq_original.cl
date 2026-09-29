#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global uint2* weights_buffer,
  __write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  __read_only image2d_t weights_scale_image2d,
  __read_only image2d_t weights_zero_point_image2d,
  int4 shared_int4_0,
  half4 shared_half4_0) {
  int dst_s = get_global_id(0);
  int dst_end_slice = shared_int4_0.x;
  int dst_s_wg_offset = get_group_id(0) * get_local_size(0);
  if (dst_s_wg_offset >= dst_end_slice) return;
  half4 r_sp0_s0 = (half4)(0.0f);
  int2 tid;
  tid.x = get_local_id(0);
  tid.y = get_local_id(1);
  if (dst_s < shared_int4_0.x) {
  half4 w_scale_s0 = read_imageh(weights_scale_image2d, smp_zero, (int2)((dst_s + 0), 0));
  half4 w_zp_s0 = read_imageh(weights_zero_point_image2d, smp_zero, (int2)((dst_s + 0), 0));
  half4 w_bias_s0 = -w_scale_s0 * ((half4)(8) + w_zp_s0);
  for (int src_s = tid.y; src_s < shared_int4_0.y; src_s += 16) {
    half4 v0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_0.y + (src_s))));
    half4 w0, w1, w2, w3;
    int linear_i4o4 = src_s * shared_int4_0.x + dst_s;
    uint2 w = weights_buffer[linear_i4o4];
    
  w0.x = convert_half((w.x) & 15u);
  w0.y = convert_half((w.x >>  4u) & 15u);
  w0.z = convert_half((w.x >>  8u) & 15u);
  w0.w = convert_half((w.x >> 12u) & 15u);
  w1.x = convert_half((w.x >> 16u) & 15u);
  w1.y = convert_half((w.x >> 20u) & 15u);
  w1.z = convert_half((w.x >> 24u) & 15u);
  w1.w = convert_half((w.x >> 28u) & 15u);
  w2.x = convert_half((w.y) & 15u);
  w2.y = convert_half((w.y >>  4u) & 15u);
  w2.z = convert_half((w.y >>  8u) & 15u);
  w2.w = convert_half((w.y >> 12u) & 15u);
  w3.x = convert_half((w.y >> 16u) & 15u);
  w3.y = convert_half((w.y >> 20u) & 15u);
  w3.z = convert_half((w.y >> 24u) & 15u);
  w3.w = convert_half((w.y >> 28u) & 15u);
;
    w0 = w0 * w_scale_s0 + w_bias_s0;
    w1 = w1 * w_scale_s0 + w_bias_s0;
    w2 = w2 * w_scale_s0 + w_bias_s0;
    w3 = w3 * w_scale_s0 + w_bias_s0;
    r_sp0_s0 += v0.x * w0;
    r_sp0_s0 += v0.y * w1;
    r_sp0_s0 += v0.z * w2;
    r_sp0_s0 += v0.w * w3;
  } 
  } 
  __local half4 temp[256];
  temp[tid.x * 16 + tid.y] = r_sp0_s0;
  for (int ystride = 16 / 2; ystride > 0; ystride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid.y < ystride) {
      r_sp0_s0 += temp[tid.x * 16 + tid.y + ystride];
      temp[tid.x * 16 + tid.y] = r_sp0_s0;
    }
  }
  if (dst_s >= shared_int4_0.x) return;
  if (tid.y != 0) return;
  {
  half4 res_value = convert_half4(r_sp0_s0);
  {

   half4 res_value_final;
  {  
  {
  half4 clamped_value = min((half4)(shared_half4_0.y), max((half4)(shared_half4_0.z), res_value));
  half4 quantized_value = round((clamped_value - (half4)(shared_half4_0.z)) * (half4)(shared_half4_0.x));
  
  half4 dequantized_value = quantized_value * (half4)(shared_half4_0.w) + (half4)(shared_half4_0.z);
  res_value_final = dequantized_value;}
  }
  write_imageh(dst_tensor_image2d, (int2)((0), ((0) * shared_int4_0.x + (dst_s + 0))), res_value_final);
};
  }
}
