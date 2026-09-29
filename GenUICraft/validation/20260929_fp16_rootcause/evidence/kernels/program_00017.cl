#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t second_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  int4 shared_int4_0) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int S = get_global_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x || S >= shared_int4_0.y) { 
    return; 
  } 
  half4 result;
    half4 first_value = read_imageh(src_tensor_image2d, smp_zero, (int2)((X), (Y)));
  first_value.y = first_value.x;
  first_value.z = first_value.x;
  first_value.w = first_value.x;
  half4 second_val = read_imageh(second_tensor_image2d, smp_zero, (int2)((S), 0));
result = first_value * second_val;
  write_imageh(dst_tensor_image2d, (int2)((X), ((Y) * shared_int4_0.y + (S))), result);
} 
