#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  int4 shared_int4_0,
  half4 shared_half4_0) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int S = get_global_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x || S >= shared_int4_0.y) { 
    return; 
  } 
  half4 src = read_imageh(src_tensor_image2d, smp_zero, (int2)((X), ((Y) * shared_int4_0.w + (S))));
  {

   half4 src_final;
  {  
  {
  half4 clamped_value = min((half4)(shared_half4_0.y), max((half4)(shared_half4_0.z), src));
  half4 quantized_value = round((clamped_value - (half4)(shared_half4_0.z)) * (half4)(shared_half4_0.x));
  
  half4 dequantized_value = quantized_value * (half4)(shared_half4_0.w) + (half4)(shared_half4_0.z);
  src_final = dequantized_value;}
  }
  write_imageh(dst_tensor_image2d, (int2)((X), ((Y) * shared_int4_0.y + (S))), src_final);
};
} 
