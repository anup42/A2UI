#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t indices_image2d,
  __read_only image2d_t src_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1) {
  int linear_id = get_global_id(0);
  int X = linear_id / 1;
  int B = linear_id % 1;
  int Y = get_global_id(1);
  int S = get_global_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x || S >= shared_int4_0.y) { 
    return; 
  } 
  int gather_index;
    {
  int slice_coord_TMP = (B) / 4;
  int sub_ch_coord_TMP = (B) % 4;
  int4 src_TMP = read_imagei(indices_image2d, smp_zero, (int2)((0), (0)));
  gather_index = (int[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];
  };
  half4 result = read_imageh(src_tensor_image2d, smp_zero, (int2)(((X) * shared_int4_0.w + (gather_index)), ((Y) * shared_int4_1.x + (S))));
  write_imageh(dst_tensor_image2d, (int2)((X), ((Y) * shared_int4_0.y + (S))), result);
}
