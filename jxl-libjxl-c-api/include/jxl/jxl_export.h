/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef JXL_EXPORT_H
#define JXL_EXPORT_H

#ifdef JXL_STATIC_DEFINE
#  define JXL_EXPORT
#  define JXL_NO_EXPORT
#else
#  ifndef JXL_EXPORT
#    if defined(__GNUC__) || defined(__clang__)
#      define JXL_EXPORT __attribute__((visibility("default")))
#    else
#      define JXL_EXPORT
#    endif
#  endif

#  ifndef JXL_NO_EXPORT
#    if defined(__GNUC__) || defined(__clang__)
#      define JXL_NO_EXPORT __attribute__((visibility("hidden")))
#    else
#      define JXL_NO_EXPORT
#    endif
#  endif
#endif

#ifndef JXL_DEPRECATED
#  if defined(_MSC_VER)
#    define JXL_DEPRECATED __declspec(deprecated)
#  elif defined(__GNUC__) || defined(__clang__)
#    define JXL_DEPRECATED __attribute__((__deprecated__))
#  else
#    define JXL_DEPRECATED
#  endif
#endif

#ifndef JXL_DEPRECATED_EXPORT
#  define JXL_DEPRECATED_EXPORT JXL_EXPORT JXL_DEPRECATED
#endif

#ifndef JXL_DEPRECATED_NO_EXPORT
#  define JXL_DEPRECATED_NO_EXPORT JXL_NO_EXPORT JXL_DEPRECATED
#endif

#endif /* JXL_EXPORT_H */
