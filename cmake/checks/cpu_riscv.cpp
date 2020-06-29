#include <stdio.h>

#if defined(__riscv)
#  include <riscv_vector.h>
#  define CV_RISCV 1
#endif

#if defined CV_RISCV
int test()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    vfloat32m1_t val = vle32_v_f32m1((const float*)(src));
    return (int)vfmv_f_s_f32m1_f32(val);
}
#else
#error "RISCV is not supported"
#endif

int main()
{
  printf("%d\n", test());
  return 0;
}
