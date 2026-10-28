/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef JXL_THREADS_EXPORT_H
#define JXL_THREADS_EXPORT_H

#ifdef JXL_THREADS_STATIC_DEFINE
#  define JXL_THREADS_EXPORT
#  define JXL_THREADS_NO_EXPORT
#else
#  ifndef JXL_THREADS_EXPORT
#    if defined(__GNUC__) || defined(__clang__)
#      define JXL_THREADS_EXPORT __attribute__((visibility("default")))
#    else
#      define JXL_THREADS_EXPORT
#    endif
#  endif

#  ifndef JXL_THREADS_NO_EXPORT
#    if defined(__GNUC__) || defined(__clang__)
#      define JXL_THREADS_NO_EXPORT __attribute__((visibility("hidden")))
#    else
#      define JXL_THREADS_NO_EXPORT
#    endif
#  endif
#endif

#ifndef JXL_THREADS_DEPRECATED
#  if defined(_MSC_VER)
#    define JXL_THREADS_DEPRECATED __declspec(deprecated)
#  elif defined(__GNUC__) || defined(__clang__)
#    define JXL_THREADS_DEPRECATED __attribute__((__deprecated__))
#  else
#    define JXL_THREADS_DEPRECATED
#  endif
#endif

#ifndef JXL_THREADS_DEPRECATED_EXPORT
#  define JXL_THREADS_DEPRECATED_EXPORT JXL_THREADS_EXPORT JXL_THREADS_DEPRECATED
#endif

#ifndef JXL_THREADS_DEPRECATED_NO_EXPORT
#  define JXL_THREADS_DEPRECATED_NO_EXPORT JXL_THREADS_NO_EXPORT JXL_THREADS_DEPRECATED
#endif

#endif /* JXL_THREADS_EXPORT_H */
