// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_INTRIN_RVV_HPP
#define OPENCV_HAL_INTRIN_RVV_HPP

#include <algorithm>

namespace cv
{

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD128 1
#define CV_SIMD128_64F 1

//////////// Unsupported native intrinsics in C++ ////////////

/*
class vuint8mf2_t
{
    uchar v[8] = {0};
};
class vint8mf2_t
{
    schar v[8] = {0};
};
class vuint16mf2_t
{
    ushort v[4] = {0};
};
class vint16mf2_t
{
    short v[4] = {0};
};
class vuint32mf2_t
{
    unsigned v[2] = {0};
};
class vint32mf2_t
{
    int v[2] = {0};
};
class vfloat32mf2_t
{
    float v[2] = {0};
};
class vuint64mf2_t
{
    uint64 v[1] = {0};
};
class vint64mf2_t
{
    int64 v[1] = {0};
};
class vfloat64mf2_t
{
    double v[1] = {0};
};
class vuint8mf4_t
{
    uchar v[4] = {0};
};
class vint8mf4_t
{
    schar v[4] = {0};
};

#define OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2) \
inline _Tpvec1 vreinterpret_v_##suffix2##m1_##suffix1##m1(const _Tpvec2& v) \
{ \
    CV_UNUSED(v); \
    return _Tpvec1(vzero_##suffix1##m1()); \
} \
inline _Tpvec2 vreinterpret_v_##suffix1##m1_##suffix2##m1(const _Tpvec1& v) \
{ \
    CV_UNUSED(v); \
    return _Tpvec2(vzero_##suffix2##m1()); \
}

OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vuint8m1_t, vuint16m1_t, u8, u16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vint8m1_t, vint16m1_t, i8, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vuint8m1_t, vuint32m1_t, u8, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vint8m1_t, vint32m1_t, i8, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vuint8m1_t, vuint64m1_t, u8, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vint8m1_t, vint64m1_t, i8, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vuint16m1_t, vuint32m1_t, u16, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vint16m1_t, vint32m1_t, i16, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vuint16m1_t, vuint64m1_t, u16, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vint16m1_t, vint64m1_t, i16, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vuint32m1_t, vuint64m1_t, u32, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(vint32m1_t, vint64m1_t, i32, i64)

#define OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(_Tpvec, _Tp, suffix, width) \
inline _Tpvec vle##width##_v_##suffix##mf2(const _Tp* ptr) \
{ \
    CV_UNUSED(ptr); \
    _Tpvec tmp; \
    return tmp; \
} \
inline void vse##width##_v_##suffix##mf2(_Tp* ptr, _Tpvec v) \
{ \
    CV_UNUSED(ptr); \
    CV_UNUSED(v); \
    return; \
}

OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vuint8mf2_t, uint8_t, u8, 8)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vint8mf2_t, int8_t, i8, 8)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vuint16mf2_t, uint16_t, u16, 16)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vint16mf2_t, int16_t, i16, 16)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vuint32mf2_t, uint32_t, u32, 32)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vint32mf2_t, int32_t, i32, 32)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vfloat32mf2_t, float32_t, f32, 32)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vuint64mf2_t, uint64_t, u64, 64)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vint64mf2_t, int64_t, i64, 64)
OPENCV_HAL_IMPL_RVV_NATIVE_LOADSTORE_MF2(vfloat64mf2_t, float64_t, f64, 64)

inline vfloat32mf2_t vfncvt_f_f_w_f32mf2 (vfloat64m1_t v)
{
    CV_UNUSED(v);
    vfloat32mf2_t tmp;
    return tmp;
}

#define OPENCV_HAL_IMPL_RVV_NATIVE_WCVT(_Tpwvec, _Tpvec, wcvt, suffix) \
inline _Tpwvec wcvt (_Tpvec v) \
{ \
    CV_UNUSED(v); \
    return _Tpwvec(vzero_##suffix##m1()); \
}

OPENCV_HAL_IMPL_RVV_NATIVE_WCVT(vuint16m1_t, vuint8mf2_t, vwcvtu_x_x_v_u16m1, u16)
OPENCV_HAL_IMPL_RVV_NATIVE_WCVT(vint16m1_t, vint8mf2_t, vwcvt_x_x_v_i16m1, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_WCVT(vuint32m1_t, vuint16mf2_t, vwcvtu_x_x_v_u32m1, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_WCVT(vint32m1_t, vint16mf2_t, vwcvt_x_x_v_i32m1, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_WCVT(vuint64m1_t, vuint32mf2_t, vwcvtu_x_x_v_u64m1, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_WCVT(vint64m1_t, vint32mf2_t, vwcvt_x_x_v_i64m1, i64)

inline vuint8mf4_t vle8_v_u8mf4 (const uint8_t *base)
{
    CV_UNUSED(base);
    vuint8mf4_t tmp;
    return tmp;
}
inline vint8mf4_t vle8_v_i8mf4 (const int8_t *base)
{
    CV_UNUSED(base);
    vint8mf4_t tmp;
    return tmp;
}

inline vuint16mf2_t vwcvtu_x_x_v_u16mf2 (vuint8mf4_t src)
{
    CV_UNUSED(src);
    vuint16mf2_t tmp;
    return tmp;
}
inline vint16mf2_t vwcvt_x_x_v_i16mf2 (vint8mf4_t src)
{
    CV_UNUSED(src);
    vint16mf2_t tmp;
    return tmp;
}

#define OPENCV_HAL_IMPL_RVV_NATIVE_NCLIP(_Tpnvec, _Tpvec, nclip) \
inline _Tpnvec nclip (_Tpvec op1, uint8_t op2) \
{ \
    CV_UNUSED(op1); \
    CV_UNUSED(op2); \
    _Tpnvec tmp; \
    return tmp; \
}

OPENCV_HAL_IMPL_RVV_NATIVE_NCLIP(vuint8mf2_t, vuint16m1_t, vnclipu_wx_u8mf2)
OPENCV_HAL_IMPL_RVV_NATIVE_NCLIP(vint8mf2_t, vint16m1_t, vnclip_wx_i8mf2)
OPENCV_HAL_IMPL_RVV_NATIVE_NCLIP(vuint16mf2_t, vuint32m1_t, vnclipu_wx_u16mf2)
OPENCV_HAL_IMPL_RVV_NATIVE_NCLIP(vint16mf2_t, vint32m1_t, vnclip_wx_i16mf2)
OPENCV_HAL_IMPL_RVV_NATIVE_NCLIP(vuint32mf2_t, vuint64m1_t, vnclipu_wx_u32mf2)
OPENCV_HAL_IMPL_RVV_NATIVE_NCLIP(vint32mf2_t, vint64m1_t, vnclip_wx_i32mf2)

inline vuint8mf2_t vreinterpret_v_i8mf2_u8mf2 (vint8mf2_t src)
{
    CV_UNUSED(src);
    vuint8mf2_t tmp;
    return tmp;
}
inline vuint16mf2_t vreinterpret_v_i16mf2_u16mf2 (vint16mf2_t src)
{
    CV_UNUSED(src);
    vuint16mf2_t tmp;
    return tmp;
}

*/

//////////// Types ////////////

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16() {}
    explicit v_uint8x16(vuint8m1_t v)
    {
        vse8_v_u8m1(val, v);
    }
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vuint8m1_t() const
    {
        return vle8_v_u8m1(val);
    }
    uchar get0() const
    {
        return val[0];
    }

    uchar val[16];
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(vint8m1_t v)
    {
        vse8_v_i8m1(val, v);
    }    
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
               schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vint8m1_t() const
    {
        return vle8_v_i8m1(val);
    }
    schar get0() const
    {
        return val[0];
    }

    schar val[16];
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(vuint16m1_t v)
    {
        vse16_v_u16m1(val, v);
    }    
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vuint16m1_t() const
    {
        return vle16_v_u16m1(val);
    }
    ushort get0() const
    {
        return val[0];
    }

    ushort val[8];
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(vint16m1_t v)
    {
        vse16_v_i16m1(val, v);
    }    
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vint16m1_t() const
    {
        return vle16_v_i16m1(val);
    }
    short get0() const
    {
        return val[0];
    }

    short val[8];
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(vuint32m1_t v)
    {
        vse32_v_u32m1(val, v);
    }
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        unsigned v[] = {v0, v1, v2, v3};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vuint32m1_t() const
    {
        return vle32_v_u32m1(val);
    }
    unsigned get0() const
    {
        return val[0];
    }

    unsigned val[4];
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(vint32m1_t v)
    {
        vse32_v_i32m1(val, v);
    }    
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vint32m1_t() const
    {
        return vle32_v_i32m1(val);
    }    
    int get0() const
    {
        return val[0];
    }
    int val[4];
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() {}
    explicit v_float32x4(vfloat32m1_t v)
    {
        vse32_v_f32m1(val, v);
    }    
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vfloat32m1_t() const
    {
        return vle32_v_f32m1(val);
    } 
    float get0() const
    {
        return val[0];
    }
    float val[4];
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    explicit v_uint64x2(vuint64m1_t v)
    {
        vse64_v_u64m1(val, v);
    }    
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vuint64m1_t() const
    {
        return vle64_v_u64m1(val);
    }
    uint64 get0() const
    {
        return val[0];
    }

    uint64 val[2];
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    explicit v_int64x2(vint64m1_t v)
    {
        vse64_v_i64m1(val, v);
    }
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vint64m1_t() const
    {
        return vle64_v_i64m1(val);
    }
    int64 get0() const
    {
        return val[0];
    }

    int64 val[2];
};

#if CV_SIMD128_64F
struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    explicit v_float64x2(vfloat64m1_t v)
    {
        vse64_v_f64m1(val, v);
    }
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        for (int i = 0; i < nlanes; ++i)
        {
            val[i] = v[i];
        }
    }
    operator vfloat64m1_t() const
    {
        return vle64_v_f64m1(val);
    }
    double get0() const
    {
        return val[0];
    }

    double val[2];
};
#endif


//////////// Initial ////////////

#define OPENCV_HAL_IMPL_RVV_INIT_INTEGER(_Tpvec, _Tp, suffix1, suffix2) \
inline v_##_Tpvec v_setzero_##suffix1() { return v_##_Tpvec(vzero_##suffix2##m1()); } \
inline v_##_Tpvec v_setall_##suffix1(_Tp v) { return v_##_Tpvec(vmv_v_x_##suffix2##m1(v)); }

OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint8x16, uchar, u8, u8)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int8x16, schar, s8, i8)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint16x8, ushort, u16, u16)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int16x8, short, s16, i16)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint32x4, unsigned, u32, u32)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int32x4, int, s32, i32)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint64x2, uint64, u64, u64)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int64x2, int64, s64, i64)

#define OPENCV_HAL_IMPL_RVV_INIT_FP(_Tpv, _Tp, suffix) \
inline v_##_Tpv v_setzero_##suffix() { return v_##_Tpv(vzero_##suffix##m1()); } \
inline v_##_Tpv v_setall_##suffix(_Tp v) { return v_##_Tpv(vfmv_v_f_##suffix##m1(v)); }

OPENCV_HAL_IMPL_RVV_INIT_FP(float32x4, float, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_INIT_FP(float64x2, double, f64)
#endif

//////////// Reinterpret ////////////

#define OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(_Tpvec, suffix) \
inline v_##_Tpvec v_reinterpret_as_##suffix(const v_##_Tpvec& v) { return v; }

OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(uint8x16, u8)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(int8x16, s8)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(uint16x8, u16)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(int16x8, s16)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(uint32x4, u32)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(int32x4, s32)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(float32x4, f32)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(uint64x2, u64)
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(int64x2, s64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_SELF_REINTERPRET(float64x2, f64)
#endif

#define OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, intrin_suffix1, intrin_suffix2) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) { return v_##_Tpvec1(vreinterpret_v_##intrin_suffix2##m1_##intrin_suffix1##m1(v)); } \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) { return v_##_Tpvec2(vreinterpret_v_##intrin_suffix1##m1_##intrin_suffix2##m1(v)); }

OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint8x16, int8x16, u8, s8, u8, i8);
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint16x8, int16x8, u16, s16, u16, i16);
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint32x4, int32x4, u32, s32, u32, i32);
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint32x4, float32x4, u32, f32, u32, f32);
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int32x4, float32x4, s32, f32, i32, f32);
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint64x2, int64x2, u64, s64, u64, i64);
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint64x2, float64x2, u64, f64, u64, f64);
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int64x2, float64x2, s64, f64, i64, f64);
#endif
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint8x16, uint16x8, u8, u16, u8, u16)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint8x16, uint32x4, u8, u32, u8, u32)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint8x16, uint64x2, u8, u64, u8, u64)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint16x8, uint32x4, u16, u32, u16, u32)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint16x8, uint64x2, u16, u64, u16, u64)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(uint32x4, uint64x2, u32, u64, u32, u64)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int8x16, int16x8, s8, s16, i8, i16)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int8x16, int32x4, s8, s32, i8, i32)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int8x16, int64x2, s8, s64, i8, i64)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int16x8, int32x4, s16, s32, i16, i32)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int16x8, int64x2, s16, s64, i16, i64)
OPENCV_HAL_IMPL_RVV_ONE_TIME_REINTERPRET(int32x4, int64x2, s32, s64, i32, i64)

#define OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, intrin_suffix1, intrin_suffix2, inter_suffix) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) { return v_##_Tpvec1(vreinterpret_v_##inter_suffix##m1_##intrin_suffix1##m1(vreinterpret_v_##intrin_suffix2##m1_##inter_suffix##m1(v))); } \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) { return v_##_Tpvec2(vreinterpret_v_##inter_suffix##m1_##intrin_suffix2##m1(vreinterpret_v_##intrin_suffix1##m1_##inter_suffix##m1(v))); }

OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8x16, int16x8, u8, s16, u8, i16, u16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8x16, int32x4, u8, s32, u8, i32, u32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8x16, int64x2, u8, s64, u8, i64, u64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16x8, int8x16, u16, s8, u16, i8, u8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16x8, int32x4, u16, s32, u16, i32, u32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16x8, int64x2, u16, s64, u16, i64, u64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32x4, int8x16, u32, s8, u32, i8, u8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32x4, int16x8, u32, s16, u32, i16, u16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32x4, int64x2, u32, s64, u32, i64, u64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64x2, int8x16, u64, s8, u64, i8, u8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64x2, int16x8, u64, s16, u64, i16, u16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64x2, int32x4, u64, s32, u64, i32, u32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8x16, float32x4, u8, f32, u8, f32, u32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16x8, float32x4, u16, f32, u16, f32, u32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64x2, float32x4, u64, f32, u64, f32, u32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8x16, float32x4, s8, f32, i8, f32, i32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16x8, float32x4, s16, f32, i16, f32, i32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int64x2, float32x4, s64, f32, i64, f32, i32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8x16, float64x2, u8, f64, u8, f64, u64);
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16x8, float64x2, u16, f64, u16, f64, u64);
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32x4, float64x2, u32, f64, u32, f64, u64);
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8x16, float64x2, s8, f64, i8, f64, i64);
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16x8, float64x2, s16, f64, i16, f64, i64);
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int32x4, float64x2, s32, f64, i32, f64, i64);
#endif

inline v_float32x4 v_reinterpret_as_f32(const v_float64x2& v)
{
    return v_float32x4(vreinterpret_v_i32m1_f32m1(vreinterpret_v_i64m1_i32m1(vreinterpret_v_f64m1_i64m1(v))));
}
inline v_float64x2 v_reinterpret_as_f64(const v_float32x4& v)
{
    return v_float64x2(vreinterpret_v_i64m1_f64m1(vreinterpret_v_i32m1_i64m1(vreinterpret_v_f32m1_i32m1(v))));
}

////////////// Extract //////////////

#define OPENCV_HAL_IMPL_RVV_EXTRACT(_Tpvec, _Tp, suffix, vmv) \
template <int s> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(vslidedown_vx_##suffix##m1(b, a, s)); \
} \
template<int i> inline _Tp v_extract_n(_Tpvec v) \
{ \
    return _Tp(vmv(vslidedown_vx_##suffix##m1(vzero_##suffix##m1(), v, i))); \
}


OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint8x16, uchar, u8, vmv_x_s_u8m1_u8)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int8x16, schar, i8, vmv_x_s_i8m1_i8)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint16x8, ushort, u16, vmv_x_s_u16m1_u16)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int16x8, short, i16, vmv_x_s_i16m1_i16)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint32x4, uint, u32, vmv_x_s_u32m1_u32)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int32x4, int, i32, vmv_x_s_i32m1_i32)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint64x2, uint64, u64, vmv_x_s_u64m1_u64)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int64x2, int64, i64, vmv_x_s_i64m1_i64)
OPENCV_HAL_IMPL_RVV_EXTRACT(v_float32x4, float, f32, vfmv_f_s_f32m1_f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_EXTRACT(v_float64x2, double, f64, vfmv_f_s_f64m1_f64)
#endif

////////////// Load/Store //////////////

#define OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(_Tpvec, _Tp, width, suffix) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(vle##width##_v_##suffix##m1(ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(vle##width##_v_##suffix##m1(ptr)); } \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ vse##width##_v_##suffix##m1(ptr, a); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ vse##width##_v_##suffix##m1(ptr, a); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ vse##width##_v_##suffix##m1(ptr, a); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ vse##width##_v_##suffix##m1(ptr, a); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    _Tp CV_DECL_ALIGNED(32) tmp_ptr[_Tpvec::nlanes] = {0}; \
    vse##width##_v_##suffix##m1(tmp_ptr, a); \
    for(int i = 0; i < _Tpvec::nlanes/2; ++i) \
    { \
        ptr[i] = tmp_ptr[i]; \
    } \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    _Tp CV_DECL_ALIGNED(32) tmp_ptr[_Tpvec::nlanes] = {0}; \
    vse##width##_v_##suffix##m1(tmp_ptr, a); \
    for(int i = 0; i < _Tpvec::nlanes/2; ++i) \
    { \
        ptr[i] = tmp_ptr[i+_Tpvec::nlanes/2]; \
    } \
}

OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint8x16, uchar, 8, u8)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int8x16, schar, 8, i8)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint16x8, ushort, 16, u16)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int16x8, short, 16, i16)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint32x4, unsigned, 32, u32)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int32x4, int, 32, i32)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint64x2, uint64, 64, u64)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int64x2, int64, 64, i64)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float32x4, float, 32, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float64x2, double, 64, f64)
#endif

inline v_int8x16 v_load_low(const schar* ptr)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7],
        0, 0, 0, 0, 0, 0, 0, 0
    };
    return v_int8x16(vle8_v_i8m1(elems));
}
inline v_int8x16 v_load_halves(const schar* ptr0, const schar* ptr1)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        ptr0[0], ptr0[1], ptr0[2], ptr0[3], ptr0[4], ptr0[5], ptr0[6], ptr0[7],
        ptr1[0], ptr1[1], ptr1[2], ptr1[3], ptr1[4], ptr1[5], ptr1[6], ptr1[7]
    };
    return v_int8x16(vle8_v_i8m1(elems));
}
inline v_uint8x16 v_load_low(const uchar* ptr) { return v_reinterpret_as_u8(v_load_low((schar*)ptr)); }
inline v_uint8x16 v_load_halves(const uchar* ptr0, const uchar* ptr1) { return v_reinterpret_as_u8(v_load_halves((schar*)ptr0, (schar*)ptr1)); }

inline v_int16x8 v_load_low(const short* ptr)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        ptr[0], ptr[1], ptr[2], ptr[3], 0, 0, 0, 0
    };
    return v_int16x8(vle16_v_i16m1(elems));
}
inline v_int16x8 v_load_halves(const short* ptr0, const short* ptr1)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        ptr0[0], ptr0[1], ptr0[2], ptr0[3], ptr1[0], ptr1[1], ptr1[2], ptr1[3]
    };
    return v_int16x8(vle16_v_i16m1(elems));
}
inline v_uint16x8 v_load_low(const ushort* ptr) { return v_reinterpret_as_u16(v_load_low((short*)ptr)); }
inline v_uint16x8 v_load_halves(const ushort* ptr0, const ushort* ptr1) { return v_reinterpret_as_u16(v_load_halves((short*)ptr0, (short*)ptr1)); }

inline v_int32x4 v_load_low(const int* ptr)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        ptr[0], ptr[1], 0, 0
    };
    return v_int32x4(vle32_v_i32m1(elems));
}
inline v_int32x4 v_load_halves(const int* ptr0, const int* ptr1)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        ptr0[0], ptr0[1], ptr1[0], ptr1[1]
    };
    return v_int32x4(vle32_v_i32m1(elems));
}
inline v_float32x4 v_load_low(const float* ptr)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        ptr[0], ptr[1], 0, 0
    };
    return v_float32x4(vle32_v_f32m1(elems));
}
inline v_float32x4 v_load_halves(const float* ptr0, const float* ptr1)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        ptr0[0], ptr0[1], ptr1[0], ptr1[1]
    };
    return v_float32x4(vle32_v_f32m1(elems));
}
inline v_uint32x4 v_load_low(const unsigned* ptr) { return v_reinterpret_as_u32(v_load_low((int*)ptr)); }
inline v_uint32x4 v_load_halves(const unsigned* ptr0, const unsigned* ptr1) { return v_reinterpret_as_u32(v_load_halves((int*)ptr0, (int*)ptr1)); }

inline v_int64x2 v_load_low(const int64* ptr)
{
    int64 CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr[0], 0
    };
    return v_int64x2(vle64_v_i64m1(elems));
}
inline v_int64x2 v_load_halves(const int64* ptr0, const int64* ptr1)
{
    int64 CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr0[0], ptr1[0]
    };
    return v_int64x2(vle64_v_i64m1(elems));
}
inline v_uint64x2 v_load_low(const uint64* ptr) { return v_reinterpret_as_u64(v_load_low((int64*)ptr)); }
inline v_uint64x2 v_load_halves(const uint64* ptr0, const uint64* ptr1) { return v_reinterpret_as_u64(v_load_halves((int64*)ptr0, (int64*)ptr1)); }

#if CV_SIMD128_64F
inline v_float64x2 v_load_low(const double* ptr)
{
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr[0], 0
    };
    return v_float64x2(vle64_v_f64m1(elems));
}
inline v_float64x2 v_load_halves(const double* ptr0, const double* ptr1)
{
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr0[0], ptr1[0]
    };
    return v_float64x2(vle64_v_f64m1(elems));
}
#endif


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
    int64_t CV_DECL_ALIGNED(32) elems[2] =
    {
        tab[idx[0]],
        tab[idx[1]]
    };
    return v_int64x2(vle64_v_i64m1(elems));
}
inline v_int64x2 v_lut_pairs(const int64* tab, const int* idx)
{
    return v_int64x2(vle64_v_i64m1(tab + idx[0]));
}
inline v_uint64x2 v_lut(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

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

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[v_extract_n<0>(idxvec)],
        tab[v_extract_n<1>(idxvec)],
        tab[v_extract_n<2>(idxvec)],
        tab[v_extract_n<3>(idxvec)]
    };
    return v_int32x4(vle32_v_i32m1(elems));
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    unsigned CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[v_extract_n<0>(idxvec)],
        tab[v_extract_n<1>(idxvec)],
        tab[v_extract_n<2>(idxvec)],
        tab[v_extract_n<3>(idxvec)]
    };
    return v_uint32x4(vle32_v_u32m1(elems));
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[v_extract_n<0>(idxvec)],
        tab[v_extract_n<1>(idxvec)],
        tab[v_extract_n<2>(idxvec)],
        tab[v_extract_n<3>(idxvec)]
    };
    return v_float32x4(vle32_v_f32m1(elems));
}

inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    x = v_float32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
    y = v_float32x4(tab[idx[0]+1], tab[idx[1]+1], tab[idx[2]+1], tab[idx[3]+1]);
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

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        tab[v_extract_n<0>(idxvec)],
        tab[v_extract_n<1>(idxvec)]
    };
    return v_float64x2(vle64_v_f64m1(elems));
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    int CV_DECL_ALIGNED(32) idx[4] = {0};
    v_store_aligned(idx, idxvec);

    x = v_float64x2(tab[idx[0]], tab[idx[1]]);
    y = v_float64x2(tab[idx[0]+1], tab[idx[1]+1]);
}
#endif

////////////// Pack boolean ////////////////////

inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    ushort CV_DECL_ALIGNED(32) ptra[8] = {0};
    ushort CV_DECL_ALIGNED(32) ptrb[8] = {0};
    v_store(ptra, a);
    v_store(ptrb, b);
    ushort CV_DECL_ALIGNED(32) elems[16] =
    {
        ptra[0], ptra[1], ptra[2], ptra[3], ptra[4], ptra[5], ptra[6], ptra[7],
        ptrb[0], ptrb[1], ptrb[2], ptrb[3], ptrb[4], ptrb[5], ptrb[6], ptrb[7]
    };
    return v_uint8x16(vle8_v_u8m1((uchar*)elems));
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    unsigned CV_DECL_ALIGNED(32) ptra[4] = {0};
    unsigned CV_DECL_ALIGNED(32) ptrb[4] = {0};
    unsigned CV_DECL_ALIGNED(32) ptrc[4] = {0};
    unsigned CV_DECL_ALIGNED(32) ptrd[4] = {0};
    v_store(ptra, a);
    v_store(ptrb, b);
    v_store(ptrc, c);
    v_store(ptrd, d);
    unsigned CV_DECL_ALIGNED(32) elems[16] =
    {
        ptra[0], ptra[1], ptra[2], ptra[3],
        ptrb[0], ptrb[1], ptrb[2], ptrb[3],
        ptrc[0], ptrc[1], ptrc[2], ptrc[3],
        ptrd[0], ptrd[1], ptrd[2], ptrd[3]
    };
    return v_uint8x16(vle8_v_u8m1((uchar*)elems));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    uint64 CV_DECL_ALIGNED(32) ptra[2] = {0};
    uint64 CV_DECL_ALIGNED(32) ptrb[2] = {0};
    uint64 CV_DECL_ALIGNED(32) ptrc[2] = {0};
    uint64 CV_DECL_ALIGNED(32) ptrd[2] = {0};
    uint64 CV_DECL_ALIGNED(32) ptre[2] = {0};
    uint64 CV_DECL_ALIGNED(32) ptrf[2] = {0};
    uint64 CV_DECL_ALIGNED(32) ptrg[2] = {0};
    uint64 CV_DECL_ALIGNED(32) ptrh[2] = {0};
    v_store(ptra, a);
    v_store(ptrb, b);
    v_store(ptrc, c);
    v_store(ptrd, d);
    v_store(ptre, e);
    v_store(ptrf, f);
    v_store(ptrg, g);
    v_store(ptrh, h);
    uint64 CV_DECL_ALIGNED(32) elems[16] =
    {
        ptra[0], ptra[1],
        ptrb[0], ptrb[1],
        ptrc[0], ptrc[1],
        ptrd[0], ptrd[1],
        ptre[0], ptre[1],
        ptrf[0], ptrf[1],
        ptrg[0], ptrg[1],
        ptrh[0], ptrh[1],
    };
    return v_uint8x16(vle8_v_u8m1((uchar*)elems));
}

////////////// Arithmetics //////////////
#define OPENCV_HAL_IMPL_RVV_BIN_OP(bin_op, _Tpvec, intrin) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a, b)); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    a = _Tpvec(intrin(a, b)); \
    return a; \
}

OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_uint8x16, vsaddu_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_uint8x16, vssubu_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_uint8x16, vmul_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_uint8x16, vdivu_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_int8x16, vsadd_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_int8x16, vssub_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_int8x16, vmul_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_int8x16, vdiv_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_uint16x8, vsaddu_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_uint16x8, vssubu_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_uint16x8, vmul_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_uint16x8, vdivu_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_int16x8, vsadd_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_int16x8, vssub_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_int16x8, vmul_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_int16x8, vdiv_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_uint32x4, vadd_vv_u32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_uint32x4, vsub_vv_u32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_uint32x4, vmul_vv_u32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_uint32x4, vdivu_vv_u32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_int32x4, vadd_vv_i32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_int32x4, vsub_vv_i32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_int32x4, vmul_vv_i32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_int32x4, vdiv_vv_i32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_float32x4, vfadd_vv_f32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_float32x4, vfsub_vv_f32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_float32x4, vfmul_vv_f32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_float32x4, vfdiv_vv_f32m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_uint64x2, vadd_vv_u64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_uint64x2, vsub_vv_u64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_uint64x2, vmul_vv_u64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_uint64x2, vdivu_vv_u64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_int64x2, vadd_vv_i64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_int64x2, vsub_vv_i64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_int64x2, vmul_vv_i64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_int64x2, vdiv_vv_i64m1)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_BIN_OP(+, v_float64x2, vfadd_vv_f64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(-, v_float64x2, vfsub_vv_f64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(*, v_float64x2, vfmul_vv_f64m1)
OPENCV_HAL_IMPL_RVV_BIN_OP(/, v_float64x2, vfdiv_vv_f64m1)
#endif


////////////// Bitwise logic //////////////

#define OPENCV_HAL_IMPL_RVV_LOGIC_OP(_Tpvec, suffix) \
OPENCV_HAL_IMPL_RVV_BIN_OP(&, _Tpvec, vand_vv_##suffix##m1) \
OPENCV_HAL_IMPL_RVV_BIN_OP(|, _Tpvec, vor_vv_##suffix##m1) \
OPENCV_HAL_IMPL_RVV_BIN_OP(^, _Tpvec, vxor_vv_##suffix##m1) \
inline _Tpvec operator ~ (const _Tpvec& a) \
{ \
    return _Tpvec(vnot_v_##suffix##m1(a)); \
}

OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint8x16, u8)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int8x16, i8)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint16x8, u16)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int16x8, i16)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint32x4, u32)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int32x4, i32)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint64x2, u64)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int64x2, i64)

#define OPENCV_HAL_IMPL_RVV_FLT_BIT_OP(bin_op, intrin) \
inline v_float32x4 operator bin_op (const v_float32x4& a, const v_float32x4& b) \
{ \
    return v_float32x4(vreinterpret_v_i32m1_f32m1(intrin(vreinterpret_v_f32m1_i32m1(a), vreinterpret_v_f32m1_i32m1(b)))); \
} \
inline v_float32x4& operator bin_op##= (v_float32x4& a, const v_float32x4& b) \
{ \
    a = v_float32x4(vreinterpret_v_i32m1_f32m1(intrin(vreinterpret_v_f32m1_i32m1(a), vreinterpret_v_f32m1_i32m1(b)))); \
    return a; \
}

OPENCV_HAL_IMPL_RVV_FLT_BIT_OP(&, vadd_vv_i32m1)
OPENCV_HAL_IMPL_RVV_FLT_BIT_OP(|, vor_vv_i32m1)
OPENCV_HAL_IMPL_RVV_FLT_BIT_OP(^, vxor_vv_i32m1)

inline v_float32x4 operator ~ (const v_float32x4& a)
{
    return v_float32x4(vreinterpret_v_i32m1_f32m1(vnot_v_i32m1(vreinterpret_v_f32m1_i32m1(a))));
}

#if CV_SIMD128_64F
#define OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(bin_op, intrin) \
inline v_float64x2 operator bin_op (const v_float64x2& a, const v_float64x2& b) \
{ \
    return v_float64x2(vreinterpret_v_i64m1_f64m1(intrin(vreinterpret_v_f64m1_i64m1(a), vreinterpret_v_f64m1_i64m1(b)))); \
} \
inline v_float64x2& operator bin_op##= (v_float64x2& a, const v_float64x2& b) \
{ \
    a = v_float64x2(vreinterpret_v_i64m1_f64m1(intrin(vreinterpret_v_f64m1_i64m1(a), vreinterpret_v_f64m1_i64m1(b)))); \
    return a; \
}

OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(&, vadd_vv_i64m1)
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(|, vor_vv_i64m1)
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(^, vxor_vv_i64m1)

inline v_float64x2 operator ~ (const v_float64x2& a)
{
    return v_float64x2(vreinterpret_v_i64m1_f64m1(vnot_v_i64m1(vreinterpret_v_f64m1_i64m1(a))));
}
#endif

////////////// Bitwise shifts //////////////

#define OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(_Tpvec, suffix) \
inline _Tpvec operator << (const _Tpvec& a, int n) \
{ return _Tpvec(vsll_vx_##suffix##m1(a, uint8_t(n))); } \
inline _Tpvec operator >> (const _Tpvec& a, int n) \
{ return _Tpvec(vsrl_vx_##suffix##m1(a, uint8_t(n))); } \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return _Tpvec(vsll_vx_##suffix##m1(a, uint8_t(n))); } \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return _Tpvec(vsrl_vx_##suffix##m1(a, uint8_t(n))); }

#define OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(_Tpvec, suffix) \
inline _Tpvec operator << (const _Tpvec& a, int n) \
{ return _Tpvec(vsll_vx_##suffix##m1(a, uint8_t(n))); } \
inline _Tpvec operator >> (const _Tpvec& a, int n) \
{ return _Tpvec(vsra_vx_##suffix##m1(a, uint8_t(n))); } \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return _Tpvec(vsll_vx_##suffix##m1(a, uint8_t(n))); } \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return _Tpvec(vsra_vx_##suffix##m1(a, uint8_t(n))); }

OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint8x16, u8)
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint16x8, u16)
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint32x4, u32)
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint64x2, u64)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int8x16, i8)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int16x8, i16)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int32x4, i32)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int64x2, i64)


////////////// Comparison //////////////

#define OPENCV_HAL_IMPL_RVV_UNSIGNED_INT_CMP_OP(_Tpvec, suffix, width) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmseq_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsne_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsltu_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsgtu_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsleu_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsgeu_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); }

#define OPENCV_HAL_IMPL_RVV_SIGNED_INT_CMP_OP(_Tpvec, suffix, width) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmseq_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsne_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmslt_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsgt_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsle_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vmerge_vxm_##suffix##m1(vmsge_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); }

#define OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, suffix, width) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vfmerge_vfm_##suffix##m1(vmfeq_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vfmerge_vfm_##suffix##m1(vmfne_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vfmerge_vfm_##suffix##m1(vmflt_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vfmerge_vfm_##suffix##m1(vmfgt_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vfmerge_vfm_##suffix##m1(vmfle_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); } \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vfmerge_vfm_##suffix##m1(vmfge_vv_##suffix##m1_b##width(a, b), vzero_##suffix##m1(), 1)); }

OPENCV_HAL_IMPL_RVV_UNSIGNED_INT_CMP_OP(v_uint8x16, u8, 8)
OPENCV_HAL_IMPL_RVV_UNSIGNED_INT_CMP_OP(v_uint16x8, u16, 16)
OPENCV_HAL_IMPL_RVV_UNSIGNED_INT_CMP_OP(v_uint32x4, u32, 32)
OPENCV_HAL_IMPL_RVV_UNSIGNED_INT_CMP_OP(v_uint64x2, u64, 64)
OPENCV_HAL_IMPL_RVV_SIGNED_INT_CMP_OP(v_int8x16, i8, 8)
OPENCV_HAL_IMPL_RVV_SIGNED_INT_CMP_OP(v_int16x8, i16, 16)
OPENCV_HAL_IMPL_RVV_SIGNED_INT_CMP_OP(v_int32x4, i32, 32)
OPENCV_HAL_IMPL_RVV_SIGNED_INT_CMP_OP(v_int64x2, i64, 64)
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(v_float32x4, f32, 32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(v_float64x2, f64, 64)
#endif

inline v_float32x4 v_not_nan(const v_float32x4& a)
{ return a == a; }

#if CV_SIMD128_64F
inline v_float64x2 v_not_nan(const v_float64x2& a)
{ return a == a; }
#endif

////////////// Min/Max //////////////

#define OPENCV_HAL_IMPL_RVV_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a, b)); \
}

OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8x16, v_min, vminu_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8x16, v_max, vmaxu_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8x16, v_min, vmin_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8x16, v_max, vmax_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16x8, v_min, vminu_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16x8, v_max, vmaxu_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16x8, v_min, vmin_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16x8, v_max, vmax_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32x4, v_min, vminu_vv_u32m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32x4, v_max, vmaxu_vv_u32m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32x4, v_min, vmin_vv_i32m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32x4, v_max, vmax_vv_i32m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32x4, v_min, vfmin_vv_f32m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32x4, v_max, vfmax_vv_f32m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint64x2, v_min, vminu_vv_u64m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint64x2, v_max, vmaxu_vv_u64m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int64x2, v_min, vmin_vv_i64m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int64x2, v_max, vmax_vv_i64m1)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64x2, v_min, vfmin_vv_f64m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64x2, v_max, vfmax_vv_f64m1)
#endif

////////////// Arithmetics wrap //////////////

OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8x16, v_add_wrap, vadd_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8x16, v_add_wrap, vadd_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16x8, v_add_wrap, vadd_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16x8, v_add_wrap, vadd_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8x16, v_sub_wrap, vsub_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8x16, v_sub_wrap, vsub_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16x8, v_sub_wrap, vsub_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16x8, v_sub_wrap, vsub_vv_i16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8x16, v_mul_wrap, vmul_vv_u8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8x16, v_mul_wrap, vmul_vv_i8m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16x8, v_mul_wrap, vmul_vv_u16m1)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16x8, v_mul_wrap, vmul_vv_i16m1)

////////////// Reduce //////////////

#define OPENCV_HAL_IMPL_RVV_REDUCE(_Tpvec, func, scalartype, suffix, red) \
inline scalartype v_reduce_##func(const _Tpvec& a)  \
{ \
    _Tpvec dst = _Tpvec(vzero_##suffix##m1()); \
    v##red##_vs_##suffix##m1_##suffix##m1(dst, a, vzero_##suffix##m1()); \
    return scalartype(dst.get0()); \
}

OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8x16, sum, unsigned, u8, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8x16, sum, int, i8, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16x8, sum, unsigned, u16, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16x8, sum, int, i16, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32x4, sum, unsigned, u32, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32x4, sum, int, i32, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_float32x4, sum, float, f32, fredsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint64x2, sum, uint64, u64, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int64x2, sum, int64, i64, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8x16, min, uchar, u8, redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8x16, min, schar, i8, redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16x8, min, ushort, u16, redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16x8, min, short, i16, redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32x4, min, unsigned, u32, redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32x4, min, int, i32, redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_float32x4, min, float, f32, fredmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8x16, max, uchar, u8, redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8x16, max, schar, i8, redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16x8, max, ushort, u16, redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16x8, max, short, i16, redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32x4, max, unsigned, u32, redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32x4, max, int, i32, redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_float32x4, max, float, f32, fredmax)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_REDUCE(v_float64x2, sum, double, f64, fredsum)
#endif

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        v_reduce_sum(a),
        v_reduce_sum(b),
        v_reduce_sum(c),
        v_reduce_sum(d)
    };
    return v_float32x4(vle32_v_f32m1(elems));
}

////////////// Square-Root //////////////

inline v_float32x4 v_sqrt(const v_float32x4& x)
{
    return v_float32x4(vfsqrt_v_f32m1(x));
}

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{
    v_float32x4 one = v_setall_f32(1.0f);
    return one / v_sqrt(x);
}

#if CV_SIMD128_64F
inline v_float64x2 v_sqrt(const v_float64x2& x)
{
    return v_float64x2(vfsqrt_v_f64m1(x));
}

inline v_float64x2 v_invsqrt(const v_float64x2& x)
{
    v_float64x2 one = v_setall_f64(1.0f);
    return one / v_sqrt(x);
}
#endif

inline v_float32x4 v_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 x(vfmacc_vv_f32m1(vfmul_vv_f32m1(a, a), b, b));
    return v_sqrt(x);
}

inline v_float32x4 v_sqr_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4(vfmacc_vv_f32m1(vfmul_vv_f32m1(a, a), b, b));
}

#if CV_SIMD128_64F
inline v_float64x2 v_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    v_float64x2 x(vfmacc_vv_f64m1(vfmul_vv_f64m1(a, a), b, b));
    return v_sqrt(x);
}

inline v_float64x2 v_sqr_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2(vfmacc_vv_f64m1(vfmul_vv_f64m1(a, a), b, b));
}
#endif

////////////// Multiply-Add //////////////

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return v_float32x4(vfmacc_vv_f32m1(c, a, b));
}
inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_int32x4(vmacc_vv_i32m1(c, a, b));
}

inline v_float32x4 v_muladd(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return v_fma(a, b, c);
}

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_fma(a, b, c);
}

#if CV_SIMD128_64F
inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_float64x2(vfmacc_vv_f64m1(c, a, b));
}

inline v_float64x2 v_muladd(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_fma(a, b, c);
}
#endif

////////////// Check all/any //////////////

#define OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(_Tpvec, suffix) \
inline bool v_check_all(const _Tpvec& a) \
{ \
    return ((v_reduce_max(a >= v_setzero_##suffix())) == 0); \
} \
inline bool v_check_any(const _Tpvec& a) \
{ \
    return ((v_reduce_max(a < v_setzero_##suffix())) == 1); \
}

OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_uint8x16, u8)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int8x16, s8)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_uint16x8, u16)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int16x8, s16)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_uint32x4, u32)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int32x4, s32)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_float32x4, f32)

inline bool v_check_all(const v_uint64x2& a)
{
    v_uint64x2 dst = v_uint64x2(vzero_u64m1());
    vredmaxu_vs_u64m1_u64m1(dst, (a >= v_setzero_u64()), vzero_u64m1());
    return bool(dst.get0());
}
inline bool v_check_any(const v_uint64x2& a)
{
    v_uint64x2 dst = v_uint64x2(vzero_u64m1());
    vredmaxu_vs_u64m1_u64m1(dst, (a < v_setzero_u64()), vzero_u64m1());
    return bool(dst.get0());
}

inline bool v_check_all(const v_int64x2& a)
{ return v_check_all(v_reinterpret_as_u64(a)); }
inline bool v_check_any(const v_int64x2& a)
{ return v_check_any(v_reinterpret_as_u64(a)); }

#if CV_SIMD128_64F
inline bool v_check_all(const v_float64x2& a)
{ return v_check_all(v_reinterpret_as_u64(a)); }
inline bool v_check_any(const v_float64x2& a)
{ return v_check_any(v_reinterpret_as_u64(a)); }
#endif

////////////// abs //////////////

#define OPENCV_HAL_IMPL_RVV_ABSDIFF(_Tpvec, _Tprvec, abs, cast) \
inline _Tprvec v_##abs(const _Tpvec& a, const _Tpvec& b) \
{ \
    return cast(v_max(a, b) - v_min(a, b)); \
}

OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int8x16, v_uint8x16, absdiff, v_reinterpret_as_u8)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int16x8, v_uint16x8, absdiff, v_reinterpret_as_u16)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int32x4, v_uint32x4, absdiff, v_reinterpret_as_u32)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int64x2, v_uint64x2, absdiff, v_reinterpret_as_u64)

OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint8x16, v_uint8x16, absdiff, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint16x8, v_uint16x8, absdiff, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint32x4, v_uint32x4, absdiff, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint64x2, v_uint64x2, absdiff, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float32x4, v_float32x4, absdiff, OPENCV_HAL_NOP)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float64x2, v_float64x2, absdiff, OPENCV_HAL_NOP)
#endif

OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int8x16, v_int8x16, absdiffs, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int16x8, v_int16x8, absdiffs, OPENCV_HAL_NOP)

#define OPENCV_HAL_IMPL_RVV_ABS(_Tprvec, _Tpvec, suffix) \
inline _Tprvec v_abs(const _Tpvec& a) \
{ \
    return v_absdiff(a, v_setzero_##suffix()); \
}

OPENCV_HAL_IMPL_RVV_ABS(v_uint8x16, v_int8x16, s8)
OPENCV_HAL_IMPL_RVV_ABS(v_uint16x8, v_int16x8, s16)
OPENCV_HAL_IMPL_RVV_ABS(v_uint32x4, v_int32x4, s32)
OPENCV_HAL_IMPL_RVV_ABS(v_float32x4, v_float32x4, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_ABS(v_float64x2, v_float64x2, f64)
#endif


#define OPENCV_HAL_IMPL_RVV_REDUCE_SAD(_Tpvec, scalartype) \
inline scalartype v_reduce_sad(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_reduce_sum(v_absdiff(a, b)); \
}

OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint8x16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int8x16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint16x8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int16x8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int32x4, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_float32x4, float)

////////////// Select //////////////

#define OPENCV_HAL_IMPL_RVV_SELECT(_Tpvec, suffix, ne) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(vmerge_vvm_##suffix##m1(ne(mask, vzero_##suffix##m1()), a, b)); \
}

OPENCV_HAL_IMPL_RVV_SELECT(v_uint8x16, u8, vmsne_vv_u8m1_b8)
OPENCV_HAL_IMPL_RVV_SELECT(v_int8x16, i8, vmsne_vv_i8m1_b8)
OPENCV_HAL_IMPL_RVV_SELECT(v_uint16x8, u16, vmsne_vv_u16m1_b16)
OPENCV_HAL_IMPL_RVV_SELECT(v_int16x8, i16, vmsne_vv_i16m1_b16)
OPENCV_HAL_IMPL_RVV_SELECT(v_uint32x4, u32, vmsne_vv_u32m1_b32)
OPENCV_HAL_IMPL_RVV_SELECT(v_int32x4, i32, vmsne_vv_i32m1_b32)
OPENCV_HAL_IMPL_RVV_SELECT(v_float32x4, f32, vmfne_vv_f32m1_b32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_SELECT(v_float64x2, f64, vmfne_vv_f64m1_b64)
#endif

////////////// Rotate shift //////////////

#define OPENCV_HAL_IMPL_RVV_ROTATE_OP(_Tpvec, suffix) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return _Tpvec(vslidedown_vx_##suffix##m1(vzero_##suffix##m1(), a, n)); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return _Tpvec(vslideup_vx_##suffix##m1(vzero_##suffix##m1(), a, n)); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(vslidedown_vx_##suffix##m1(b, a, n)); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(vslideup_vx_##suffix##m1(b, a, n)); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ CV_UNUSED(b); return a; }


OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_uint8x16, u8)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_int8x16, i8)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_uint16x8, u16)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_int16x8, i16)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_uint32x4, u32)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_int32x4, i32)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_float32x4, f32)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_uint64x2, u64)
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_int64x2, i64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_ROTATE_OP(v_float64x2, f64)
#endif

////////////// Convert to float //////////////

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(vfcvt_f_x_v_f32m1(a));
}

#if CV_SIMD128_64F
inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    float CV_DECL_ALIGNED(32) ptr[2] = {0};
    vse32_v_f32mf2(ptr, vfncvt_f_f_w_f32mf2(a));
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        ptr[0], ptr[1], 0, 0
    };
    return v_float32x4(vle32_v_f32m1(elems));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{
    float CV_DECL_ALIGNED(32) ptra[2] = {0};
    float CV_DECL_ALIGNED(32) ptrb[2] = {0};
    vse32_v_f32mf2(ptra, vfncvt_f_f_w_f32mf2(a));
    vse32_v_f32mf2(ptrb, vfncvt_f_f_w_f32mf2(b));
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        ptra[0], ptra[1], ptrb[0], ptrb[1]
    };
    return v_float32x4(vle32_v_f32m1(elems));
}

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
    double CV_DECL_ALIGNED(32) ptr[4] = {0};
    vse64_v_f64m2(ptr, vfwcvt_f_x_v_f64m2(a));
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr[0], ptr[1]
    };
    return v_float64x2(vle64_v_f64m1(elems));
}

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{
    double CV_DECL_ALIGNED(32) ptr[4] = {0};
    vse64_v_f64m2(ptr, vfwcvt_f_x_v_f64m2(a));
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr[2], ptr[3]
    };
    return v_float64x2(vle64_v_f64m1(elems));
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    double CV_DECL_ALIGNED(32) ptr[4] = {0};
    vse64_v_f64m2(ptr, vfwcvt_f_f_v_f64m2(a));
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr[0], ptr[1]
    };
    return v_float64x2(vle64_v_f64m1(elems));
}

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{
    double CV_DECL_ALIGNED(32) ptr[4] = {0};
    vse64_v_f64m2(ptr, vfwcvt_f_f_v_f64m2(a));
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        ptr[2], ptr[3]
    };
    return v_float64x2(vle64_v_f64m1(elems));
}

inline v_float64x2 v_cvt_f64(const v_int64x2& a)
{
    return v_float64x2(vfcvt_f_x_v_f64m1(a));
}
#endif

////////////// Broadcast //////////////

#define OPENCV_HAL_IMPL_RVV_BROADCAST(_Tpvec, suffix) \
template<int i> inline _Tpvec v_broadcast_element(_Tpvec v) \
{ \
    return v_setall_##suffix(v_extract_n<i>(v)); \
}

OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint8x16, u8)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int8x16, s8)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint16x8, u16)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int16x8, s16)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint32x4, u32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int32x4, s32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint64x2, u64)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int64x2, s64)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_float32x4, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_BROADCAST(v_float64x2, f64)
#endif

////////////// Transpose4x4 //////////////

#define OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(_Tpvec, _Tp, suffix) \
inline void v_transpose4x4(const v_##_Tpvec& a0, const v_##_Tpvec& a1, \
                         const v_##_Tpvec& a2, const v_##_Tpvec& a3, \
                         v_##_Tpvec& b0, v_##_Tpvec& b1, \
                         v_##_Tpvec& b2, v_##_Tpvec& b3) \
{ \
    _Tp CV_DECL_ALIGNED(32) elems0[4] = \
    { \
        v_extract_n<0>(a0), \
        v_extract_n<0>(a1), \
        v_extract_n<0>(a2), \
        v_extract_n<0>(a3) \
    }; \
    b0 = v_load(elems0); \
    _Tp CV_DECL_ALIGNED(32) elems1[4] = \
    { \
        v_extract_n<1>(a0), \
        v_extract_n<1>(a1), \
        v_extract_n<1>(a2), \
        v_extract_n<1>(a3) \
    }; \
    b1 = v_load(elems1); \
    _Tp CV_DECL_ALIGNED(32) elems2[4] = \
    { \
        v_extract_n<2>(a0), \
        v_extract_n<2>(a1), \
        v_extract_n<2>(a2), \
        v_extract_n<2>(a3) \
    }; \
    b2 = v_load(elems2); \
    _Tp CV_DECL_ALIGNED(32) elems3[4] = \
    { \
        v_extract_n<3>(a0), \
        v_extract_n<3>(a1), \
        v_extract_n<3>(a2), \
        v_extract_n<3>(a3) \
    }; \
    b3 = v_load(elems3); \
}

OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(int32x4, int, i32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(float32x4, float, f32)

////////////// Reverse //////////////

#define OPENCV_HAL_IMPL_RVV_REVERSE(_Tpvec, _Tp, width, suffix) \
inline _Tpvec v_reverse(const _Tpvec& a)  \
{ \
    _Tp CV_DECL_ALIGNED(32) ptr[_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptra[_Tpvec::nlanes] = {0}; \
    v_store(ptra, a); \
    for (int i = 0; i < _Tpvec::nlanes; i++) \
    { \
        ptr[i] = ptra[_Tpvec::nlanes-i-1]; \
    } \
    return v_load(ptr); \
}

OPENCV_HAL_IMPL_RVV_REVERSE(v_uint8x16, uchar, 8, u8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int8x16, schar, 8, i8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint16x8, ushort, 16, u16)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int16x8, short, 16, i16)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint32x4, unsigned, 32, u32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int32x4, int, 32, i32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_float32x4, float, 32, f32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint64x2, uint64, 64, u64)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int64x2, int64, 64, i64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_REVERSE(v_float64x2, double, 64, f64)
#endif

//////////// Value reordering ////////////

#define OPENCV_HAL_IMPL_RVV_EXPAND(_Tpwvec, _Tp, _Tpvec, width, suffix, wcvt) \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
{ \
    _Tp CV_DECL_ALIGNED(32) lptr[_Tpvec::nlanes/2] = {0}; \
    _Tp CV_DECL_ALIGNED(32) hptr[_Tpvec::nlanes/2] = {0}; \
    v_store_low(lptr, a); \
    v_store_high(hptr, a); \
    b0 = _Tpwvec(wcvt(vle##width##_v_##suffix##mf2(lptr))); \
    b1 = _Tpwvec(wcvt(vle##width##_v_##suffix##mf2(hptr))); \
} \
inline _Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _Tp CV_DECL_ALIGNED(32) lptr[_Tpvec::nlanes/2] = {0}; \
    v_store_low(lptr, a); \
    return _Tpwvec(wcvt(vle##width##_v_##suffix##mf2(lptr))); \
} \
inline _Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _Tp CV_DECL_ALIGNED(32) hptr[_Tpvec::nlanes/2] = {0}; \
    v_store_high(hptr, a); \
    return _Tpwvec(wcvt(vle##width##_v_##suffix##mf2(hptr))); \
} \
inline _Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    return _Tpwvec(wcvt(vle##width##_v_##suffix##mf2(ptr))); \
}

OPENCV_HAL_IMPL_RVV_EXPAND(v_uint16x8, uchar, v_uint8x16, 8, u8, vwcvtu_x_x_v_u16m1)
OPENCV_HAL_IMPL_RVV_EXPAND(v_int16x8, schar, v_int8x16, 8, i8, vwcvt_x_x_v_i16m1)
OPENCV_HAL_IMPL_RVV_EXPAND(v_uint32x4, ushort, v_uint16x8, 16, u16, vwcvtu_x_x_v_u32m1)
OPENCV_HAL_IMPL_RVV_EXPAND(v_int32x4, short, v_int16x8, 16, i16, vwcvt_x_x_v_i32m1)
OPENCV_HAL_IMPL_RVV_EXPAND(v_uint64x2, uint, v_uint32x4, 32, u32, vwcvtu_x_x_v_u64m1)
OPENCV_HAL_IMPL_RVV_EXPAND(v_int64x2, int, v_int32x4, 32, i32, vwcvt_x_x_v_i64m1)

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    return v_uint32x4(vwcvtu_x_x_v_u32m1(vwcvtu_x_x_v_u16mf2(vle8_v_u8mf4(ptr))));
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    return v_int32x4(vwcvt_x_x_v_i32m1(vwcvt_x_x_v_i16mf2(vle8_v_i8mf4(ptr))));
}


#define OPENCV_HAL_IMPL_RVV_PACK(_Tpvec, _Tp, _Tpwvec, pack, width, suffix, rshr, cast) \
inline _Tpvec v_##pack(const _Tpwvec& a, const _Tpwvec& b) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra[_Tpvec::nlanes/2] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[_Tpvec::nlanes/2] = {0}; \
    vse##width##_v_##suffix##mf2(ptra, cast(rshr(a, 0))); \
    vse##width##_v_##suffix##mf2(ptrb, cast(rshr(b, 0))); \
    return v_load_halves(ptra, ptrb); \
} \
inline void v_##pack##_store(_Tp* ptr, const _Tpwvec& a) \
{ \
    vse##width##_v_##suffix##mf2(ptr, cast(rshr(a, 0))); \
} \
template<int n> inline \
_Tpvec v_rshr_##pack(const _Tpwvec& a, const _Tpwvec& b) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra[_Tpvec::nlanes/2] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[_Tpvec::nlanes/2] = {0}; \
    vse##width##_v_##suffix##mf2(ptra, cast(rshr(a, n))); \
    vse##width##_v_##suffix##mf2(ptrb, cast(rshr(b, n))); \
    return v_load_halves(ptra, ptrb); \
} \
template<int n> inline \
void v_rshr_##pack##_store(_Tp* ptr, const _Tpwvec& a) \
{ \
    vse##width##_v_##suffix##mf2(ptr, cast(rshr(a, n))); \
}

OPENCV_HAL_IMPL_RVV_PACK(v_uint8x16, uchar, v_uint16x8, pack, 8, u8, vnclipu_wx_u8mf2, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_PACK(v_int8x16, schar, v_int16x8, pack, 8, i8, vnclip_wx_i8mf2, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_PACK(v_uint16x8, ushort, v_uint32x4, pack, 16, u16, vnclipu_wx_u16mf2, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_PACK(v_int16x8, short, v_int32x4, pack, 16, i16, vnclip_wx_i16mf2, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_PACK(v_uint32x4, unsigned, v_uint64x2, pack, 32, u32, vnclipu_wx_u32mf2, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_PACK(v_int32x4, int, v_int64x2, pack, 32, i32, vnclip_wx_i32mf2, OPENCV_HAL_NOP)

OPENCV_HAL_IMPL_RVV_PACK(v_uint8x16, uchar, v_int16x8, pack_u, 8, u8, vnclip_wx_i8mf2, vreinterpret_v_i8mf2_u8mf2)
OPENCV_HAL_IMPL_RVV_PACK(v_uint16x8, ushort, v_int32x4, pack_u, 16, u16, vnclip_wx_i16mf2, vreinterpret_v_i16mf2_u16mf2)


#define OPENCV_HAL_IMPL_RVV_UNPACKS(_Tpvec, _Tp, width, suffix) \
inline void v_zip(const v_##_Tpvec& a0, const v_##_Tpvec& a1, v_##_Tpvec& b0, v_##_Tpvec& b1) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra0[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptra1[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb0[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb1[v_##_Tpvec::nlanes] = {0}; \
    v_store(ptra0, a0); \
    v_store(ptra1, a1); \
    int i; \
    for( i = 0; i < v_##_Tpvec::nlanes/2; i++ ) \
    { \
        ptrb0[i*2] = ptra0[i]; \
        ptrb0[i*2+1] = ptra1[i]; \
    } \
    for( ; i < v_##_Tpvec::nlanes; i++ ) \
    { \
        ptrb1[i*2-v_##_Tpvec::nlanes] = ptra0[i]; \
        ptrb1[i*2-v_##_Tpvec::nlanes+1] = ptra1[i]; \
    } \
    b0 = v_load(ptrb0); \
    b1 = v_load(ptrb1); \
} \
inline v_##_Tpvec v_combine_low(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes/2] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes/2] = {0}; \
    v_store_low(ptra, a); \
    v_store_low(ptrb, b); \
    return v_load_halves(ptra, ptrb); \
} \
inline v_##_Tpvec v_combine_high(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes/2] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes/2] = {0}; \
    v_store_high(ptra, a); \
    v_store_high(ptrb, b); \
    return v_load_halves(ptra, ptrb); \
} \
inline void v_recombine(const v_##_Tpvec& a, const v_##_Tpvec& b, v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    c = v_combine_low(a, b); \
    d = v_combine_high(a, b); \
}

OPENCV_HAL_IMPL_RVV_UNPACKS(uint8x16, uchar, 8, u8)
OPENCV_HAL_IMPL_RVV_UNPACKS(int8x16, schar, 8, i8)
OPENCV_HAL_IMPL_RVV_UNPACKS(uint16x8, ushort, 16, u16)
OPENCV_HAL_IMPL_RVV_UNPACKS(int16x8, short, 16, i16)
OPENCV_HAL_IMPL_RVV_UNPACKS(uint32x4, unsigned, 32, u32)
OPENCV_HAL_IMPL_RVV_UNPACKS(int32x4, int, 32, i32)
OPENCV_HAL_IMPL_RVV_UNPACKS(float32x4, float, 32, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_UNPACKS(float64x2, double, 64, f64)
#endif


#define OPENCV_HAL_IMPL_RVV_INTERLEAVED(_Tpvec, _Tp, suffix, width) \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes] = {0}; \
    int i, i2; \
    for( i = i2 = 0; i < v_##_Tpvec::nlanes; i++, i2 += 2 ) \
    { \
        ptra[i] = ptr[i2]; \
        ptrb[i] = ptr[i2+1]; \
    } \
    a = v_load(ptra); \
    b = v_load(ptrb); \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, v_##_Tpvec& c) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrc[v_##_Tpvec::nlanes] = {0}; \
    int i, i3; \
    for( i = i3 = 0; i < v_##_Tpvec::nlanes; i++, i3 += 3 ) \
    { \
        ptra[i] = ptr[i3]; \
        ptrb[i] = ptr[i3+1]; \
        ptrc[i] = ptr[i3+2]; \
    } \
    a = v_load(ptra); \
    b = v_load(ptrb); \
    c = v_load(ptrc); \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, \
                                v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrc[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrd[v_##_Tpvec::nlanes] = {0}; \
    int i, i4; \
    for( i = i4 = 0; i < v_##_Tpvec::nlanes; i++, i4 += 4 ) \
    { \
        ptra[i] = ptr[i4]; \
        ptrb[i] = ptr[i4+1]; \
        ptrc[i] = ptr[i4+2]; \
        ptrd[i] = ptr[i4+3]; \
    } \
    a = v_load(ptra); \
    b = v_load(ptrb); \
    c = v_load(ptrc); \
    d = v_load(ptrd); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    int i, i2; \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes] = {0}; \
    v_store(ptra, a); \
    v_store(ptrb, b); \
    for( i = i2 = 0; i < v_##_Tpvec::nlanes; i++, i2 += 2 ) \
    { \
        ptr[i2] = ptra[i]; \
        ptr[i2+1] = ptrb[i]; \
    } \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    int i, i3; \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrc[v_##_Tpvec::nlanes] = {0}; \
    v_store(ptra, a); \
    v_store(ptrb, b); \
    v_store(ptrc, c); \
    for( i = i3 = 0; i < v_##_Tpvec::nlanes; i++, i3 += 3 ) \
    { \
        ptr[i3] = ptra[i]; \
        ptr[i3+1] = ptrb[i]; \
        ptr[i3+2] = ptrc[i]; \
    } \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, const v_##_Tpvec& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    int i, i4; \
    _Tp CV_DECL_ALIGNED(32) ptra[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrb[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrc[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrd[v_##_Tpvec::nlanes] = {0}; \
    v_store(ptra, a); \
    v_store(ptrb, b); \
    v_store(ptrc, c); \
    v_store(ptrd, d); \
    for( i = i4 = 0; i < v_##_Tpvec::nlanes; i++, i4 += 4 ) \
    { \
        ptr[i4] = ptra[i]; \
        ptr[i4+1] = ptrb[i]; \
        ptr[i4+2] = ptrc[i]; \
        ptr[i4+3] = ptrd[i]; \
    } \
} \
inline v_##_Tpvec v_interleave_pairs(const v_##_Tpvec& vec) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptr[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrvec[v_##_Tpvec::nlanes] = {0}; \
    v_store(ptrvec, vec); \
    for (int i = 0; i < v_##_Tpvec::nlanes/4; i++) \
    { \
        ptr[4*i  ] = ptrvec[4*i  ]; \
        ptr[4*i+1] = ptrvec[4*i+2]; \
        ptr[4*i+2] = ptrvec[4*i+1]; \
        ptr[4*i+3] = ptrvec[4*i+3]; \
    } \
    return v_load(ptr); \
} \
inline v_##_Tpvec v_interleave_quads(const v_##_Tpvec& vec) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptr[v_##_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrvec[v_##_Tpvec::nlanes] = {0}; \
    v_store(ptrvec, vec); \
    for (int i = 0; i < v_##_Tpvec::nlanes/8; i++) \
    { \
        ptr[8*i  ] = ptrvec[4*i  ]; \
        ptr[8*i+1] = ptrvec[4*i+4]; \
        ptr[8*i+2] = ptrvec[4*i+1]; \
        ptr[8*i+3] = ptrvec[4*i+5]; \
        ptr[8*i+4] = ptrvec[4*i+2]; \
        ptr[8*i+5] = ptrvec[4*i+6]; \
        ptr[8*i+6] = ptrvec[4*i+3]; \
        ptr[8*i+7] = ptrvec[4*i+7]; \
    } \
    return v_load(ptr); \
}

OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint8x16, uchar, u8, 8)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int8x16, schar, i8, 8)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint16x8, ushort, u16, 16)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int16x8, short, i16, 16)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint32x4, unsigned, u32, 32)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int32x4, int, i32, 32)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(float32x4, float, f32, 32)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint64x2, uint64, u64, 64)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int64x2, int64, i64, 64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_INTERLEAVED(float64x2, double, f64, 64)
#endif

//////////// PopCount ////////////

static const unsigned char popCountTable[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

#define OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(_rTpvec, _Tpvec, _rTp, _Tp, suffix) \
inline _rTpvec v_popcount(const _Tpvec& a) \
{ \
    uchar CV_DECL_ALIGNED(32) ptra[16] = {0}; \
    v_store(ptra, v_reinterpret_as_u8(a)); \
    _rTp CV_DECL_ALIGNED(32) ptr[_Tpvec::nlanes] = {0}; \
    v_store(ptr, v_setzero_##suffix()); \
    for (int i = 0; i < _Tpvec::nlanes*(int)sizeof(_Tp); i++) \
        ptr[i/sizeof(_Tp)] += popCountTable[ptra[i]]; \
    return v_load(ptr); \
}

OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint8x16, v_uint8x16, uchar, uchar, u8)
OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint8x16, v_int8x16, uchar, schar, u8)
OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint16x8, v_uint16x8, ushort, ushort, u16)
OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint16x8, v_int16x8, ushort, short, u16)
OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint32x4, v_uint32x4, unsigned, unsigned, u32)
OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint32x4, v_int32x4, unsigned, int, u32)
OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint64x2, v_uint64x2, uint64, uint64, u64)
OPENCV_HAL_IMPL_RVV_POPCOUNT_OP(v_uint64x2, v_int64x2, uint64, int64, u64)

//////////// SignMask ////////////

#define OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(_Tpvec, _Tp, suffix) \
inline int v_signmask(const _Tpvec& a) \
{ \
    int mask = 0; \
    _Tp CV_DECL_ALIGNED(32) ptr[_Tpvec::nlanes] = {0}; \
    v_store(ptr, v_reinterpret_as_##suffix(a)); \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        mask |= (int(ptr[i]) < 0) << i; \
    return mask; \
}

OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int16x8, short, s16)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int32x4, int, s32)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_float32x4, float, f32)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int64x2, int64, s64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_float64x2, double, f64)
#endif

//////////// Scan forward ////////////

#define OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(_Tpvec, _Tp, suffix) \
inline int v_scan_forward(const _Tpvec& a) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptr[_Tpvec::nlanes] = {0}; \
    v_store(ptr, v_reinterpret_as_##suffix(a)); \
    for (int i = 0; i < _Tpvec::nlanes; i++) \
        if(int(ptr[i]) < 0) \
            return i; \
    return 0; \
}

OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_int16x8, short, s16)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_int32x4, int, s32)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_float32x4, float, f32)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_int64x2, int64, s64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_SCAN_FORWOARD_OP(v_float64x2, double, f64)
#endif

//////////// Pack triplets ////////////

#define OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(_Tpvec, _Tp) \
inline _Tpvec v_pack_triplets(const _Tpvec& vec) \
{ \
    _Tp CV_DECL_ALIGNED(32) ptr[_Tpvec::nlanes] = {0}; \
    _Tp CV_DECL_ALIGNED(32) ptrvec[_Tpvec::nlanes] = {0}; \
    v_store(ptrvec, vec); \
    for (int i = 0; i < _Tpvec::nlanes/4; i++) \
    { \
        ptr[3*i  ] = ptrvec[4*i  ]; \
        ptr[3*i+1] = ptrvec[4*i+2]; \
        ptr[3*i+2] = ptrvec[4*i+2]; \
    } \
    return v_load(ptr); \
}

OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint8x16, uchar)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int8x16, schar)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint16x8, ushort)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int16x8, short)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int32x4, int)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_float32x4, float)


////// FP16 support ///////

#if CV_FP16
inline v_float32x4 v_load_expand(const float16_t* ptr)
{
    return v_float32x4(vfwcvt_f_f_v_f32m1(vle16_v_f16mf2(ptr)));
}

inline void v_pack_store(float16_t* ptr, const v_float32x4& v)
{
    vse16_v_f16mf2(ptr, vfncvt_f_f_w_f16mf2(v));
}
#else
inline v_float32x4 v_load_expand(const float16_t* ptr)
{
    const int N = 4;
    float buf[N];
    for( int i = 0; i < N; i++ ) buf[i] = (float)ptr[i];
    return v_load(buf);
}

inline void v_pack_store(float16_t* ptr, const v_float32x4& v)
{
    const int N = 4;
    float buf[N];
    v_store(buf, v);
    for( int i = 0; i < N; i++ ) ptr[i] = float16_t(buf[i]);
}
#endif

////////////// Rounding //////////////

inline v_int32x4 v_round(const v_float32x4& a)
{
    return v_int32x4(vfcvt_x_f_v_i32m1(a));
}

inline v_int32x4 v_floor(const v_float32x4& a)
{
    v_float32x4 ZP5 = v_setall_f32(0.5f);
    v_float32x4 t = a - ZP5;
    return v_int32x4(vfcvt_x_f_v_i32m1(t));
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    v_float32x4 ZP5 = v_setall_f32(0.5f);
    v_float32x4 t = a + ZP5;
    return v_int32x4(vfcvt_x_f_v_i32m1(t));
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{
    return v_int32x4(vfcvt_rtz_x_f_v_i32m1(a));
}
#if CV_SIMD128_64F
inline v_int32x4 v_round(const v_float64x2& a)
{
    CV_UNUSED(a);
    return v_setzero_s32();
}

inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{
    CV_UNUSED(a);
    CV_UNUSED(b);
    return v_setzero_s32();
}

inline v_int32x4 v_floor(const v_float64x2& a)
{
    CV_UNUSED(a);
    return v_setzero_s32();
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
    CV_UNUSED(a);
    return v_setzero_s32();
}

inline v_int32x4 v_trunc(const v_float64x2& a)
{
    CV_UNUSED(a);
    return v_setzero_s32();
}
#endif


//////// Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{
    int CV_DECL_ALIGNED(32) ptr[8] = {0};
    v_int32x4 t1, t2;
    vse32_v_i32m2(ptr, vwmul_vv_i32m2(a, b));
    v_load_deinterleave(ptr, t1, t2);
    return t1 + t2;
}
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    int CV_DECL_ALIGNED(32) ptr[8] = {0};
    v_int32x4 t1, t2;
    vse32_v_i32m2(ptr, vwmul_vv_i32m2(a, b));
    v_load_deinterleave(ptr, t1, t2);
    return t1 + t2 + c;
}

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
    int64 CV_DECL_ALIGNED(32) ptr[4] = {0};
    v_int64x2 t1, t2;
    vse64_v_i64m2(ptr, vwmul_vv_i64m2(a, b));
    v_load_deinterleave(ptr, t1, t2);
    return t1 + t2;
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    int64 CV_DECL_ALIGNED(32) ptr[4] = {0};
    v_int64x2 t1, t2;
    vse64_v_i64m2(ptr, vwmul_vv_i64m2(a, b));
    v_load_deinterleave(ptr, t1, t2);
    return t1 + t2 + c;
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    unsigned CV_DECL_ALIGNED(32) ptr[16] = {0};
    v_uint32x4 t1, t2, t3, t4;
    vse32_v_u32m4(ptr, vqmaccu_vv_u32m4(vzero_u32m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4;
}
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b,
                                   const v_uint32x4& c)
{
    unsigned CV_DECL_ALIGNED(32) ptr[16] = {0};
    v_uint32x4 t1, t2, t3, t4;
    vse32_v_u32m4(ptr, vqmaccu_vv_u32m4(vzero_u32m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4 + c;
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    int CV_DECL_ALIGNED(32) ptr[16] = {0};
    v_int32x4 t1, t2, t3, t4;
    vse32_v_i32m4(ptr, vqmacc_vv_i32m4(vzero_i32m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4;
}
inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b,
                                  const v_int32x4& c)
{
    int CV_DECL_ALIGNED(32) ptr[16] = {0};
    v_int32x4 t1, t2, t3, t4;
    vse32_v_i32m4(ptr, vqmacc_vv_i32m4(vzero_i32m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4 + c;
}

// 16 >> 64
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    uint64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    v_uint64x2 t1, t2, t3, t4;
    vse64_v_u64m4(ptr, vqmaccu_vv_u64m4(vzero_u64m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4;
}
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{
    uint64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    v_uint64x2 t1, t2, t3, t4;
    vse64_v_u64m4(ptr, vqmaccu_vv_u64m4(vzero_u64m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4 + c;
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    int64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    v_int64x2 t1, t2, t3, t4;
    vse64_v_i64m4(ptr, vqmacc_vv_i64m4(vzero_i64m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4;
}
inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b,
                                  const v_int64x2& c)
{
    int64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    v_int64x2 t1, t2, t3, t4;
    vse64_v_i64m4(ptr, vqmacc_vv_i64m4(vzero_i64m4(), a, b));
    v_load_deinterleave(ptr, t1, t2, t3, t4);
    return t1 + t2 + t3 + t4 + c;
}

// 32 >> 64f
#if CV_SIMD128_64F
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a,   const v_int32x4& b,
                                    const v_float64x2& c)
{ return v_dotprod_expand(a, b) + c; }
#endif

//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{
    int CV_DECL_ALIGNED(32) ptr[8] = {0};
    vse32_v_i32m2(ptr, vwmul_vv_i32m2(a, b));
    v_int32x4 t1 = v_load(ptr);
    v_int32x4 t2 = v_load(ptr+4);
    return t1 + t2;
}
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    int CV_DECL_ALIGNED(32) ptr[8] = {0};
    vse32_v_i32m2(ptr, vwmul_vv_i32m2(a, b));
    v_int32x4 t1 = v_load(ptr);
    v_int32x4 t2 = v_load(ptr+4);
    return t1 + t2 + c;
}

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{
    int64 CV_DECL_ALIGNED(32) ptr[4] = {0};
    vse64_v_i64m2(ptr, vwmul_vv_i64m2(a, b));
    v_int64x2 t1 = v_load(ptr);
    v_int64x2 t2 = v_load(ptr+2);
    return t1 + t2;
}
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    int64 CV_DECL_ALIGNED(32) ptr[4] = {0};
    vse64_v_i64m2(ptr, vwmul_vv_i64m2(a, b));
    v_int64x2 t1 = v_load(ptr);
    v_int64x2 t2 = v_load(ptr+2);
    return t1 + t2 + c;
}


// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{
    unsigned CV_DECL_ALIGNED(32) ptr[16] = {0};
    vse32_v_u32m4(ptr, vqmaccu_vv_u32m4(vzero_u32m4(), a, b));
    v_uint32x4 t1 = v_load(ptr);
    v_uint32x4 t2 = v_load(ptr+4);
    v_uint32x4 t3 = v_load(ptr+8);
    v_uint32x4 t4 = v_load(ptr+12);
    return t1 + t2 + t3 + t4;
}
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{
    unsigned CV_DECL_ALIGNED(32) ptr[16] = {0};
    vse32_v_u32m4(ptr, vqmaccu_vv_u32m4(vzero_u32m4(), a, b));
    v_uint32x4 t1 = v_load(ptr);
    v_uint32x4 t2 = v_load(ptr+4);
    v_uint32x4 t3 = v_load(ptr+8);
    v_uint32x4 t4 = v_load(ptr+12);
    return t1 + t2 + t3 + t4 + c;
}
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{
    int CV_DECL_ALIGNED(32) ptr[16] = {0};
    vse32_v_i32m4(ptr, vqmacc_vv_i32m4(vzero_i32m4(), a, b));
    v_int32x4 t1 = v_load(ptr);
    v_int32x4 t2 = v_load(ptr+4);
    v_int32x4 t3 = v_load(ptr+8);
    v_int32x4 t4 = v_load(ptr+12);
    return t1 + t2 + t3 + t4;
}
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{
    int CV_DECL_ALIGNED(32) ptr[16] = {0};
    vse32_v_i32m4(ptr, vqmacc_vv_i32m4(vzero_i32m4(), a, b));
    v_int32x4 t1 = v_load(ptr);
    v_int32x4 t2 = v_load(ptr+4);
    v_int32x4 t3 = v_load(ptr+8);
    v_int32x4 t4 = v_load(ptr+12);
    return t1 + t2 + t3 + t4 + c;
}

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{
    uint64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    vse64_v_u64m4(ptr, vqmaccu_vv_u64m4(vzero_u64m4(), a, b));
    v_uint64x2 t1 = v_load(ptr);
    v_uint64x2 t2 = v_load(ptr+2);
    v_uint64x2 t3 = v_load(ptr+4);
    v_uint64x2 t4 = v_load(ptr+6);
    return t1 + t2 + t3 + t4;
}
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{
    uint64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    vse64_v_u64m4(ptr, vqmaccu_vv_u64m4(vzero_u64m4(), a, b));
    v_uint64x2 t1 = v_load(ptr);
    v_uint64x2 t2 = v_load(ptr+2);
    v_uint64x2 t3 = v_load(ptr+4);
    v_uint64x2 t4 = v_load(ptr+6);
    return t1 + t2 + t3 + t4 + c;
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    int64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    vse64_v_i64m4(ptr, vqmacc_vv_i64m4(vzero_i64m4(), a, b));
    v_int64x2 t1 = v_load(ptr);
    v_int64x2 t2 = v_load(ptr+2);
    v_int64x2 t3 = v_load(ptr+4);
    v_int64x2 t4 = v_load(ptr+6);
    return t1 + t2 + t3 + t4;
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{
    int64 CV_DECL_ALIGNED(32) ptr[8] = {0};
    vse64_v_i64m4(ptr, vqmacc_vv_i64m4(vzero_i64m4(), a, b));
    v_int64x2 t1 = v_load(ptr);
    v_int64x2 t2 = v_load(ptr+2);
    v_int64x2 t3 = v_load(ptr+4);
    v_int64x2 t4 = v_load(ptr+6);
    return t1 + t2 + t3 + t4 + c;
}

// 32 >> 64f
#if CV_SIMD128_64F
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod_fast(a, b)); }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_dotprod_expand_fast(a, b) + c; }
#endif


inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    vfloat32m1_t res = vfmul_vf_f32m1(m0, v_extract_n<0>(v));
    res = vfmacc_vf_f32m1(res, v_extract_n<1>(v), m1);
    res = vfmacc_vf_f32m1(res, v_extract_n<2>(v), m2);
    res = vfmacc_vf_f32m1(res, v_extract_n<3>(v), m3);
    return v_float32x4(res);
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    vfloat32m1_t res = vfmul_vf_f32m1(m0, v_extract_n<0>(v));
    res = vfmacc_vf_f32m1(res, v_extract_n<1>(v), m1);
    res = vfmacc_vf_f32m1(res, v_extract_n<2>(v), m2);
    return v_float32x4(res) + a;
}

#define OPENCV_HAL_IMPL_RVV_MUL_EXPAND(_Tpvec, _Tpwvec, _Tpw, suffix, wmul, width) \
inline void v_mul_expand(const _Tpvec& a, const _Tpvec& b, _Tpwvec& c, _Tpwvec& d) \
{ \
    _Tpw CV_DECL_ALIGNED(32) ptr[_Tpwvec::nlanes*2] = {0}; \
    vse##width##_v_##suffix##m2(ptr, wmul(a, b)); \
    c = _Tpwvec(vle##width##_v_##suffix##m1(ptr)); \
    d = _Tpwvec(vle##width##_v_##suffix##m1(ptr+_Tpwvec::nlanes)); \
}

OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint8x16, v_uint16x8, ushort, u16, vwmulu_vv_u16m2, 16)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int8x16, v_int16x8, short, i16, vwmul_vv_i16m2, 16)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint16x8, v_uint32x4, unsigned, u32, vwmulu_vv_u32m2, 32)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int16x8, v_int32x4, int, i32, vwmul_vv_i32m2, 32)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint32x4, v_uint64x2, uint64, u64, vwmulu_vv_u64m2, 64)


inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{
    return v_int16x8(vnsra_wx_i16m1(vwmul_vv_i32m2(a, b), 16));
}
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(vnsrl_wx_u16m1(vwmulu_vv_u32m2(a, b), 16));
}



inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END


}

#endif
