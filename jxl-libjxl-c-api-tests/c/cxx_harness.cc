// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/decode_cxx.h>
#include <jxl/encode_cxx.h>
#include <jxl/resizable_parallel_runner_cxx.h>
#include <jxl/thread_parallel_runner_cxx.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

struct CxxCapture {
  size_t width = 0;
  size_t bpp = 0;
  uint8_t* dst = nullptr;
};

void cxx_scanline_cb(void* opaque, size_t x, size_t y, size_t num_pixels,
                     const void* pixels) {
  auto* cap = static_cast<CxxCapture*>(opaque);
  std::memcpy(cap->dst + (y * cap->width + x) * cap->bpp, pixels,
              num_pixels * cap->bpp);
}

}  // namespace

extern "C" int cxx_harness_decode_rgba8(const uint8_t* data, size_t data_size,
                                        uint32_t* out_w, uint32_t* out_h,
                                        uint8_t** out_pixels,
                                        size_t* out_size) {
  if (!data || !out_w || !out_h || !out_pixels || !out_size) {
    return -1;
  }
  *out_pixels = nullptr;
  *out_size = 0;

  // Also instantiate JxlEncoderMake and JxlThreadParallelRunnerMake to verify
  // all four *_cxx.h RAII wrappers compile and link cleanly.
  JxlEncoderPtr unused_enc = JxlEncoderMake(nullptr);
  (void)unused_enc;
  JxlThreadParallelRunnerPtr unused_tpr =
      JxlThreadParallelRunnerMake(nullptr, 2);
  if (!unused_tpr) {
    return -2;
  }

  JxlDecoderPtr dec = JxlDecoderMake(nullptr);
  JxlResizableParallelRunnerPtr runner =
      JxlResizableParallelRunnerMake(nullptr);
  if (!dec || !runner) {
    return -3;
  }

  if (JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner,
                                  runner.get()) != JXL_DEC_SUCCESS ||
      JxlDecoderSubscribeEvents(dec.get(),
                                JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
          JXL_DEC_SUCCESS ||
      JxlDecoderSetInput(dec.get(), data, data_size) != JXL_DEC_SUCCESS) {
    return -4;
  }
  JxlDecoderCloseInput(dec.get());

  if (JxlDecoderProcessInput(dec.get()) != JXL_DEC_BASIC_INFO) {
    return -5;
  }
  JxlBasicInfo info{};
  if (JxlDecoderGetBasicInfo(dec.get(), &info) != JXL_DEC_SUCCESS) {
    return -6;
  }
  JxlResizableParallelRunnerSetThreads(
      runner.get(),
      JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));

  const size_t total = static_cast<size_t>(info.xsize) * info.ysize * 4;
  auto* buf = static_cast<uint8_t*>(std::malloc(total > 0 ? total : 1));
  if (!buf) {
    return -7;
  }
  CxxCapture cap{info.xsize, 4, buf};
  JxlPixelFormat fmt{4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  if (JxlDecoderProcessInput(dec.get()) != JXL_DEC_NEED_IMAGE_OUT_BUFFER ||
      JxlDecoderSetImageOutCallback(dec.get(), &fmt, cxx_scanline_cb, &cap) !=
          JXL_DEC_SUCCESS ||
      JxlDecoderProcessInput(dec.get()) != JXL_DEC_FULL_IMAGE ||
      JxlDecoderProcessInput(dec.get()) != JXL_DEC_SUCCESS) {
    std::free(buf);
    return -8;
  }

  *out_w = info.xsize;
  *out_h = info.ysize;
  *out_pixels = buf;
  *out_size = total;
  return 0;
}

extern "C" void cxx_harness_free(uint8_t* ptr) { std::free(ptr); }
