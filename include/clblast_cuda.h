
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the special CUDA interface to the CLBlast BLAS routines. It also contains the
// definitions of the returned status codes and the layout and transpose types. This is the header
// users of the CUDA API of CLBlast should include and use.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_CUDA_H_
#define CLBLAST_CLBLAST_CUDA_H_

#include <cstdlib> // For size_t
#include <string> // For OverrideParameters function
#include <unordered_map> // For OverrideParameters function

// CUDA
#include <cuda.h> // CUDA driver API
#include <nvrtc.h> // NVIDIA runtime compilation API

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#if defined(_WIN32) && defined(CLBLAST_DLL)
  #if defined(COMPILING_DLL)
    #define PUBLIC_API __declspec(dllexport)
  #else
    #define PUBLIC_API __declspec(dllimport)
  #endif
#else
  #define PUBLIC_API
#endif

namespace clblast {
// =================================================================================================

// Status codes. These codes can be returned by functions declared in this header file. The error
// codes match either the standard CUDA driver API error codes or the regular CLBlast error codes.
enum class StatusCode {

  // Status codes in common with the OpenCL standard
  kSuccess                   =   0, // CUDA_SUCCESS
  kInvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
  kInvalidLocalThreadsTotal  = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
  kInvalidLocalThreadsDim    = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension

  // Status codes in common with the clBLAS library
  kNotImplemented            = -1024, // Routine or functionality not implemented yet
  kInvalidMatrixA            = -1022, // Matrix A is not a valid OpenCL buffer
  kInvalidMatrixB            = -1021, // Matrix B is not a valid OpenCL buffer
  kInvalidMatrixC            = -1020, // Matrix C is not a valid OpenCL buffer
  kInvalidVectorX            = -1019, // Vector X is not a valid OpenCL buffer
  kInvalidVectorY            = -1018, // Vector Y is not a valid OpenCL buffer
  kInvalidDimension          = -1017, // Dimensions M, N, and K have to be larger than zero
  kInvalidLeadDimA           = -1016, // LD of A is smaller than the matrix's first dimension
  kInvalidLeadDimB           = -1015, // LD of B is smaller than the matrix's first dimension
  kInvalidLeadDimC           = -1014, // LD of C is smaller than the matrix's first dimension
  kInvalidIncrementX         = -1013, // Increment of vector X cannot be zero
  kInvalidIncrementY         = -1012, // Increment of vector Y cannot be zero
  kInsufficientMemoryA       = -1011, // Matrix A's OpenCL buffer is too small
  kInsufficientMemoryB       = -1010, // Matrix B's OpenCL buffer is too small
  kInsufficientMemoryC       = -1009, // Matrix C's OpenCL buffer is too small
  kInsufficientMemoryX       = -1008, // Vector X's OpenCL buffer is too small
  kInsufficientMemoryY       = -1007, // Vector Y's OpenCL buffer is too small

  // Custom additional status codes for CLBlast
  kInsufficientMemoryTemp    = -2050, // Temporary buffer provided to GEMM routine is too small
  kInvalidBatchCount         = -2049, // The batch count needs to be positive
  kInvalidOverrideKernel     = -2048, // Trying to override parameters for an invalid kernel
  kMissingOverrideParameter  = -2047, // Missing override parameter(s) for the target kernel
  kInvalidLocalMemUsage      = -2046, // Not enough local memory available on this device
  kNoHalfPrecision           = -2045, // Half precision (16-bits) not supported by the device
  kNoDoublePrecision         = -2044, // Double precision (64-bits) not supported by the device
  kInvalidVectorScalar       = -2043, // The unit-sized vector is not a valid OpenCL buffer
  kInsufficientMemoryScalar  = -2042, // The unit-sized vector's OpenCL buffer is too small
  kDatabaseError             = -2041, // Entry for the device was not found in the database
  kUnknownError              = -2040, // A catch-all error code representing an unspecified error
  kUnexpectedError           = -2039, // A catch-all error code representing an unexpected exception
};

// Matrix layout and transpose types
enum class Layout { kRowMajor = 101, kColMajor = 102 };
enum class Transpose { kNo = 111, kYes = 112, kConjugate = 113 };
enum class Triangle { kUpper = 121, kLower = 122 };
enum class Diagonal { kNonUnit = 131, kUnit = 132 };
enum class Side { kLeft = 141, kRight = 142 };
enum class KernelMode { kCrossCorrelation = 151, kConvolution = 152 };

// Precision scoped enum (values in bits)
enum class Precision { kHalf = 16, kSingle = 32, kDouble = 64,
                       kComplexSingle = 3232, kComplexDouble = 6464, kAny = -1 };

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
template <typename T>
StatusCode Rotg(CUdeviceptr sa_buffer, const size_t sa_offset,
                CUdeviceptr sb_buffer, const size_t sb_offset,
                CUdeviceptr sc_buffer, const size_t sc_offset,
                CUdeviceptr ss_buffer, const size_t ss_offset,
                const CUcontext context, const CUdevice device);

// Generate modified givens plane rotation: SROTMG/DROTMG
template <typename T>
StatusCode Rotmg(CUdeviceptr sd1_buffer, const size_t sd1_offset,
                 CUdeviceptr sd2_buffer, const size_t sd2_offset,
                 CUdeviceptr sx1_buffer, const size_t sx1_offset,
                 const CUdeviceptr sy1_buffer, const size_t sy1_offset,
                 CUdeviceptr sparam_buffer, const size_t sparam_offset,
                 const CUcontext context, const CUdevice device);

// Apply givens plane rotation: SROT/DROT
template <typename T>
StatusCode Rot(const size_t n,
               CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
               const T cos,
               const T sin,
               const CUcontext context, const CUdevice device);

// Apply modified givens plane rotation: SROTM/DROTM
template <typename T>
StatusCode Rotm(const size_t n,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                CUdeviceptr sparam_buffer, const size_t sparam_offset,
                const CUcontext context, const CUdevice device);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
template <typename T>
StatusCode Swap(const size_t n,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
template <typename T>
StatusCode Scal(const size_t n,
                const T alpha,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
template <typename T>
StatusCode Copy(const size_t n,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
template <typename T>
StatusCode Axpy(const size_t n,
                const T alpha,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Dot product of two vectors: SDOT/DDOT/HDOT
template <typename T>
StatusCode Dot(const size_t n,
               CUdeviceptr dot_buffer, const size_t dot_offset,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
               const CUcontext context, const CUdevice device);

// Dot product of two complex vectors: CDOTU/ZDOTU
template <typename T>
StatusCode Dotu(const size_t n,
                CUdeviceptr dot_buffer, const size_t dot_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
template <typename T>
StatusCode Dotc(const size_t n,
                CUdeviceptr dot_buffer, const size_t dot_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
template <typename T>
StatusCode Nrm2(const size_t n,
                CUdeviceptr nrm2_buffer, const size_t nrm2_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
template <typename T>
StatusCode Asum(const size_t n,
                CUdeviceptr asum_buffer, const size_t asum_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
template <typename T>
StatusCode Sum(const size_t n,
               CUdeviceptr sum_buffer, const size_t sum_offset,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               const CUcontext context, const CUdevice device);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
StatusCode Amax(const size_t n,
                CUdeviceptr imax_buffer, const size_t imax_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
template <typename T>
StatusCode Amin(const size_t n,
                CUdeviceptr imin_buffer, const size_t imin_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
template <typename T>
StatusCode Max(const size_t n,
               CUdeviceptr imax_buffer, const size_t imax_offset,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               const CUcontext context, const CUdevice device);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
template <typename T>
StatusCode Min(const size_t n,
               CUdeviceptr imin_buffer, const size_t imin_offset,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               const CUcontext context, const CUdevice device);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
template <typename T>
StatusCode Gemv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
template <typename T>
StatusCode Gbmv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n, const size_t kl, const size_t ku,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
template <typename T>
StatusCode Hemv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
template <typename T>
StatusCode Hbmv(const Layout layout, const Triangle triangle,
                const size_t n, const size_t k,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
template <typename T>
StatusCode Hpmv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr ap_buffer, const size_t ap_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
template <typename T>
StatusCode Symv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
template <typename T>
StatusCode Sbmv(const Layout layout, const Triangle triangle,
                const size_t n, const size_t k,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
template <typename T>
StatusCode Spmv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr ap_buffer, const size_t ap_offset,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                const CUcontext context, const CUdevice device);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
template <typename T>
StatusCode Trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
template <typename T>
StatusCode Tbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n, const size_t k,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
template <typename T>
StatusCode Tpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const CUdeviceptr ap_buffer, const size_t ap_offset,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
StatusCode Trsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
StatusCode Tbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n, const size_t k,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
StatusCode Tpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const CUdeviceptr ap_buffer, const size_t ap_offset,
                CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUcontext context, const CUdevice device);

// General rank-1 matrix update: SGER/DGER/HGER
template <typename T>
StatusCode Ger(const Layout layout,
               const size_t m, const size_t n,
               const T alpha,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
               CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
               const CUcontext context, const CUdevice device);

// General rank-1 complex matrix update: CGERU/ZGERU
template <typename T>
StatusCode Geru(const Layout layout,
                const size_t m, const size_t n,
                const T alpha,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUcontext context, const CUdevice device);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
template <typename T>
StatusCode Gerc(const Layout layout,
                const size_t m, const size_t n,
                const T alpha,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUcontext context, const CUdevice device);

// Hermitian rank-1 matrix update: CHER/ZHER
template <typename T>
StatusCode Her(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
               const CUcontext context, const CUdevice device);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
template <typename T>
StatusCode Hpr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               CUdeviceptr ap_buffer, const size_t ap_offset,
               const CUcontext context, const CUdevice device);

// Hermitian rank-2 matrix update: CHER2/ZHER2
template <typename T>
StatusCode Her2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUcontext context, const CUdevice device);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
template <typename T>
StatusCode Hpr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                CUdeviceptr ap_buffer, const size_t ap_offset,
                const CUcontext context, const CUdevice device);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
template <typename T>
StatusCode Syr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
               const CUcontext context, const CUdevice device);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
template <typename T>
StatusCode Spr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               CUdeviceptr ap_buffer, const size_t ap_offset,
               const CUcontext context, const CUdevice device);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
template <typename T>
StatusCode Syr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUcontext context, const CUdevice device);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
template <typename T>
StatusCode Spr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
                const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
                CUdeviceptr ap_buffer, const size_t ap_offset,
                const CUcontext context, const CUdevice device);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
template <typename T>
StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                const size_t m, const size_t n, const size_t k,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld,
                const CUcontext context, const CUdevice device,
                CUdeviceptr temp_buffer = 0);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
template <typename T>
StatusCode Symm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld,
                const CUcontext context, const CUdevice device);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
template <typename T>
StatusCode Hemm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld,
                const CUcontext context, const CUdevice device);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
template <typename T>
StatusCode Syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const T beta,
                CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld,
                const CUcontext context, const CUdevice device);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
template <typename T>
StatusCode Herk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                const T beta,
                CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld,
                const CUcontext context, const CUdevice device);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
template <typename T>
StatusCode Syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k,
                 const T alpha,
                 const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                 const CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                 const T beta,
                 CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld,
                 const CUcontext context, const CUdevice device);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
template <typename T, typename U>
StatusCode Her2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k,
                 const T alpha,
                 const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                 const CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                 const U beta,
                 CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld,
                 const CUcontext context, const CUdevice device);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
template <typename T>
StatusCode Trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                const CUcontext context, const CUdevice device);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
template <typename T>
StatusCode Trsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                const CUcontext context, const CUdevice device);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
template <typename T>
StatusCode Had(const size_t n,
               const T alpha,
               const CUdeviceptr x_buffer, const size_t x_offset, const size_t x_inc,
               const CUdeviceptr y_buffer, const size_t y_offset, const size_t y_inc,
               const T beta,
               CUdeviceptr z_buffer, const size_t z_offset, const size_t z_inc,
               const CUcontext context, const CUdevice device);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
template <typename T>
StatusCode Omatcopy(const Layout layout, const Transpose a_transpose,
                    const size_t m, const size_t n,
                    const T alpha,
                    const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld,
                    CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld,
                    const CUcontext context, const CUdevice device);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
template <typename T>
StatusCode Im2col(const KernelMode kernel_mode,
                  const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                  const CUdeviceptr im_buffer, const size_t im_offset,
                  CUdeviceptr col_buffer, const size_t col_offset,
                  const CUcontext context, const CUdevice device);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
template <typename T>
StatusCode Col2im(const KernelMode kernel_mode,
                  const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                  const CUdeviceptr col_buffer, const size_t col_offset,
                  CUdeviceptr im_buffer, const size_t im_offset,
                  const CUcontext context, const CUdevice device);

// Batched convolution as GEMM (non-BLAS function): SCONVGEMM/DCONVGEMM/HCONVGEMM
template <typename T>
StatusCode Convgemm(const KernelMode kernel_mode,
                    const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                    const CUdeviceptr im_buffer, const size_t im_offset,
                    const CUdeviceptr kernel_buffer, const size_t kernel_offset,
                    CUdeviceptr result_buffer, const size_t result_offset,
                    const CUcontext context, const CUdevice device);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
template <typename T>
StatusCode AxpyBatched(const size_t n,
                       const T *alphas,
                       const CUdeviceptr x_buffer, const size_t *x_offsets, const size_t x_inc,
                       CUdeviceptr y_buffer, const size_t *y_offsets, const size_t y_inc,
                       const size_t batch_count,
                       const CUcontext context, const CUdevice device);

// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
template <typename T>
StatusCode GemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                       const size_t m, const size_t n, const size_t k,
                       const T *alphas,
                       const CUdeviceptr a_buffer, const size_t *a_offsets, const size_t a_ld,
                       const CUdeviceptr b_buffer, const size_t *b_offsets, const size_t b_ld,
                       const T *betas,
                       CUdeviceptr c_buffer, const size_t *c_offsets, const size_t c_ld,
                       const size_t batch_count,
                       const CUcontext context, const CUdevice device);

// StridedBatched version of GEMM: SGEMMSTRIDEDBATCHED/DGEMMSTRIDEDBATCHED/CGEMMSTRIDEDBATCHED/ZGEMMSTRIDEDBATCHED/HGEMMSTRIDEDBATCHED
template <typename T>
StatusCode GemmStridedBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                              const size_t m, const size_t n, const size_t k,
                              const T alpha,
                              const CUdeviceptr a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                              const CUdeviceptr b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                              const T beta,
                              CUdeviceptr c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                              const size_t batch_count,
                              const CUcontext context, const CUdevice device);

// =================================================================================================

// Retrieves the required size of the temporary buffer for the GEMM kernel (optional)
template <typename T>
StatusCode GemmTempBufferSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                              const size_t m, const size_t n, const size_t k,
                              const size_t a_offset, const size_t a_ld,
                              const size_t b_offset, const size_t b_ld,
                              const size_t c_offset, const size_t c_ld,
                              const CUdevice device, size_t& temp_buffer_size);

// =================================================================================================

// CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on
// for the same device. This cache can be cleared to free up system memory or in case of debugging.
StatusCode PUBLIC_API ClearCache();

// The cache can also be pre-initialized for a specific device with all possible CLBlast kernels.
// Further CLBlast routine calls will then run at maximum speed.
StatusCode PUBLIC_API FillCache(const CUdevice device);

// =================================================================================================

// Retrieves current tuning parameters for a specific device-precision-kernel combination
StatusCode PUBLIC_API RetrieveParameters(const CUdevice device, const std::string &kernel_name,
                                         const Precision precision,
                                         std::unordered_map<std::string,size_t> &parameters);

// Overrides tuning parameters for a specific device-precision-kernel combination. The next time
// the target routine is called it will re-compile and use the new parameters from then on.
StatusCode PUBLIC_API OverrideParameters(const CUdevice device, const std::string &kernel_name,
                                         const Precision precision,
                                         const std::unordered_map<std::string,size_t> &parameters);

// =================================================================================================

} // namespace clblast

// CLBLAST_CLBLAST_CUDA_H_
#endif
