/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_HAL_INTRIN_RISCV_HPP
#define OPENCV_HAL_INTRIN_RISCV_HPP

#include <algorithm>

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD128 1
#define CV_SIMD128_64F 0


//////////// Types ////////////

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16() {}
    explicit v_uint8x16(vuint8m1_t v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = vle8_v_u8m1(v);
    }
    uchar get0() const
    {
        return vmv_x_s_u8m1_u8(val);
    }

    vuint8m1_t val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(vint8m1_t v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
               schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = vle8_v_i8m1(v);
    }
    schar get0() const
    {
        return vmv_x_s_i8m1_i8(val);
    }

    vint8m1_t val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(vuint16m1_t v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = vle16_v_u16m1(v);
    }
    ushort get0() const
    {
        return vmv_x_s_u16m1_u16(val);
    }

    vuint16m1_t val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(vint16m1_t v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = vle16_v_i16m1(v);
    }
    short get0() const
    {
        return vmv_x_s_i16m1_i16(val);
    }

    vint16m1_t val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(vuint32m1_t v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        unsigned v[] = {v0, v1, v2, v3};
        val = vle32_v_u32m1(v);
    }
    unsigned get0() const
    {
        return vmv_x_s_u32m1_u32(val);
    }

    vuint32m1_t val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(vint32m1_t v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        val = vle32_v_i32m1(v);
    }
    int get0() const
    {
        return vmv_x_s_i32m1_i32(val);
    }
    vint32m1_t val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() {}
    explicit v_float32x4(vfloat32m1_t v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        val = vle32_v_f32m1(v);
    }
    float get0() const
    {
        return vfmv_f_s_f32m1_f32(val);
    }
    vfloat32m1_t val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    explicit v_uint64x2(vuint64m1_t v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        val = vle64_v_u64m1(v);
    }
    uint64 get0() const
    {
        return vmv_x_s_u64m1_u64(val);
    }
    vuint64m1_t val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    explicit v_int64x2(vint64m1_t v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        val = vle64_v_i64m1(v);
    }
    int64 get0() const
    {
        return vmv_x_s_i64m1_i64(val);
    }
    vint64m1_t val;
};

#if CV_SIMD128_64F
struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    explicit v_float64x2(vfloat64m1_t v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        val = vle64_v_f64m1(v);
    }
    double get0() const
    {
        return vfmv_f_s_f64m1_f64(val);
    }
    vfloat64m1_t val;
};
#endif


//////////// Load and store operations ////////////

#define OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(_Tpvec, _Tp, _Tpv, suffix1, suffix2) \
inline v_##_Tpvec v_setzero_##suffix1() { return v_##_Tpvec(vzero_##suffix2##m1()); } \
inline v_##_Tpvec v_setall_##suffix1(_Tp v) { return v_##_Tpvec(vmv_v_x_##suffix2##m1(v)); } \
inline v##_Tpv##m1_t vreinterpret_##suffix2##_##suffix2(v##_Tpv##m1_t v) { return v; } \
inline v_uint8x16 v_reinterpret_as_u8(const v_##_Tpvec& v) { return v_uint8x16(vreinterpret_u8_##suffix2##_u8m1(v.val)); } \
inline v_int8x16 v_reinterpret_as_s8(const v_##_Tpvec& v) { return v_int8x16(vreinterpret_i8_##suffix2##_i8m1(v.val)); } \
inline v_uint16x8 v_reinterpret_as_u16(const v_##_Tpvec& v) { return v_uint16x8(vreinterpret_u16_##suffix2##_u16m1(v.val)); } \
inline v_int16x8 v_reinterpret_as_s16(const v_##_Tpvec& v) { return v_int16x8(vreinterpret_i16_##suffix2##_i16m1(v.val)); } \
inline v_uint32x4 v_reinterpret_as_u32(const v_##_Tpvec& v) { return v_uint32x4(vreinterpret_u32_##suffix2##_u32m1(v.val)); } \
inline v_int32x4 v_reinterpret_as_s32(const v_##_Tpvec& v) { return v_int32x4(vreinterpret_i32_##suffix2##_i32m1(v.val)); } \
inline v_uint64x2 v_reinterpret_as_u64(const v_##_Tpvec& v) { return v_uint64x2(vreinterpret_u64_##suffix2##_u64m1(v.val)); } \
inline v_int64x2 v_reinterpret_as_s64(const v_##_Tpvec& v) { return v_int64x2(vreinterpret_i64_##suffix2##_i64m1(v.val)); } \
inline v_float32x4 v_reinterpret_as_f32(const v_##_Tpvec& v) { return v_float32x4(vreinterpret_f32_##suffix2##_f32m1(v.val)); }

OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(uint8x16, uchar, uint8, u8, u8)
OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(int8x16, schar, int8, s8, i8)
OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(uint16x8, ushort, uint16, u16, u16)
OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(int16x8, short, int16, s16, i16)
OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(uint32x4, unsigned, uint32, u32, u32)
OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(int32x4, int, s32, int32, i32)
OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(uint64x2, uint64, uint64, u64, u64)
OPENCV_HAL_IMPL_RISCV_INIT_INTEGER(int64x2, int64, int64, s64, i64)

#define OPENCV_HAL_IMPL_RISCV_INIT_FP(_Tpv, _Tp, suffix) \
inline v_##_Tpv v_setzero_##suffix() { return v_##_Tpv(vfmv_v_f_##suffix##m1((_Tp)0)); } \
inline v_##_Tpv v_setall_##suffix(_Tp v) { return v_##_Tpv(vfmv_v_f_##suffix##m1(v)); }

OPENCV_HAL_IMPL_RISCV_INIT_FP(float32x4, float, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RISCV_INIT_FP(float64x2, double, f64)
#endif

#define OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(_Tpvec, _Tp, _Tpv, width, suffix, vl, halfvl) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(vle##width##_v_##suffix##m1(ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(vle##width##_v_##suffix##m1(ptr)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ \
    vsetvl_e##width##m1(halfvl); \
    v##_Tpv##m1_t v = vle##width##_v_##suffix##m1(ptr); \
    vsetvl_e##width##m1(vl); \
    return _Tpvec(v); \
} \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    vsetvl_e##width##m1(halfvl); \
    v##_Tpv##m1_t v1 = vle##width##_v_##suffix##m1(ptr0); \
    v##_Tpv##m1_t v2 = vle##width##_v_##suffix##m1(ptr1); \
    vsetvl_e##width##m1(vl); \
    return _Tpvec(vslideup_vx_i8m1_m(vmset_m_b##width() , v1, v2, 64)); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ vse##width##_v_##suffix##m1(ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ vse##width##_v_##suffix##m1(ptr, a.val); }

OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_uint8x16, uchar, uint8, 8, u8, 16, 8)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_int8x16, schar, int8, 8, i8, 16, 8)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_uint16x8, ushort, uint16, 16, u16, 8, 4)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_int16x8, short, int16, 16, i16, 8, 4)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_uint32x4, unsigned, uint32, 32, u32, 4, 2)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_int32x4, int, int32, 32, i32, 4, 2)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_uint64x2, uint64, uint64, 64, u64, 2, 1)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_int64x2, int64, int64, 64, i64, 2, 1)
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_float32x4, float, float32, 32, f32, 4, 2)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RISCV_LOADSTORE_OP(v_float64x2, double, float64, 64, f64, 2, 1)
#endif


//////////// Value reordering ////////////

#define OPENCV_HAL_IMPL_RISCV_EXPAND(_Tpvec, _Tpwvec, _Tp, suffix, wsuffix) \
inline _Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    return _Tpwvec(vreinterpret_##wsuffix##_##suffix##_##wsuffix##m1(vle##width##_v_##suffix##m1(ptr))); \
}

OPENCV_HAL_IMPL_RISCV_EXPAND(v_uint8x16, v_uint16x8, uchar, 8, u8, u16)
OPENCV_HAL_IMPL_RISCV_EXPAND(v_int8x16, v_int16x8, schar, 8, i8, i16)
OPENCV_HAL_IMPL_RISCV_EXPAND(v_uint16x8, v_uint32x4, ushort, 16, u16, u32)
OPENCV_HAL_IMPL_RISCV_EXPAND(v_int16x8, v_int32x4, short, 16, i16, i32)
OPENCV_HAL_IMPL_RISCV_EXPAND(v_uint32x4, v_uint64x2, uint, 32, u32, u64)
OPENCV_HAL_IMPL_RISCV_EXPAND(v_int32x4, v_int64x2, int, 32, i32, i64)

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    return v_uint32x4(vreinterpret_u32_u8_u32m1(ptr));
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    return v_int32x4(vreinterpret_i32_i8_i32m1(ptr));
}


////////////// Lookup table access ////////////////////

inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]],
        tab[idx[ 1]],
        tab[idx[ 2]],
        tab[idx[ 3]],
        tab[idx[ 4]],
        tab[idx[ 5]],
        tab[idx[ 6]],
        tab[idx[ 7]],
        tab[idx[ 8]],
        tab[idx[ 9]],
        tab[idx[10]],
        tab[idx[11]],
        tab[idx[12]],
        tab[idx[13]],
        tab[idx[14]],
        tab[idx[15]]
    };
    return v_int8x16(vle8_v_i8m1(elems));
}
inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[4]],
        tab[idx[4] + 1],
        tab[idx[5]],
        tab[idx[5] + 1],
        tab[idx[6]],
        tab[idx[6] + 1],
        tab[idx[7]],
        tab[idx[7] + 1]
    };
    return v_int8x16(vle8_v_i8m1(elems));
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[0] + 2],
        tab[idx[0] + 3],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[1] + 2],
        tab[idx[1] + 3],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[2] + 2],
        tab[idx[2] + 3],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[3] + 2],
        tab[idx[3] + 3]
    };
    return v_int8x16(vle8_v_i8m1(elems));
}
inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((schar*)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((schar*)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((schar*)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]],
        tab[idx[4]],
        tab[idx[5]],
        tab[idx[6]],
        tab[idx[7]]
    };
    return v_int16x8(vle16_v_i16m1(elems));
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1]
    };
    return v_int16x8(vle16_v_i16m1(elems));
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[0] + 2],
        tab[idx[0] + 3],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[1] + 2],
        tab[idx[1] + 3]
    };
    return v_int16x8(vle16_v_i16m1(elems));
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((short*)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((short*)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((short*)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_int32x4(vle32_v_i32m1(elems));
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1]
    };
    return v_int32x4(vle32_v_i32m1(elems));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(vle32_v_i32m1(tab + idx[0]));
}

inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[2] =
    {
        tab[idx[0]],
        tab[idx[1]]
    };
    return v_int32x4(vle64_v_i64m1(elems));
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(vle64_v_i64m1(tab + idx[0]));
}
inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_float32x4(vle32_v_f32m1(elems));
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1]
    };
    return v_float32x4(vle32_v_f32m1(elems));
}
inline v_float32x4 v_lut_quads(const float* tab, const int* idx)
{
    return v_float32x4(vle32_v_f32m1(tab + idx[0]));
}

#if CV_SIMD128_64F
inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        tab[idx[0]],
        tab[idx[1]]
    };
    return v_float64x2(vle64_v_f64m1(elems));
}

inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{
    return v_float64x2(vle64_v_f64m1(tab + idx[0]));
}


CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}

#endif
