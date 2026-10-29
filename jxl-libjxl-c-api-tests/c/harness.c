/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#include <jxl/cms.h>
#include <jxl/cms_interface.h>
#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/compressed_icc.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/gain_map.h>
#include <jxl/memory_manager.h>
#include <jxl/parallel_runner.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/stats.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/types.h>
#include <jxl/version.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  C_OUT_MODE_BUFFER = 0,
  C_OUT_MODE_SIMPLE_CALLBACK = 1,
  C_OUT_MODE_MT_CALLBACK = 2,
};

enum {
  C_RUNNER_NONE = 0,
  C_RUNNER_THREAD = 1,
  C_RUNNER_RESIZABLE = 2,
};

typedef struct {
  size_t chunk_size;
  JxlPixelFormat format;
  uint32_t output_mode;
  uint32_t runner_mode;
  size_t num_threads;
  bool keep_orientation;
  bool unpremultiply_alpha;
  bool decode_ec0;
} CHarnessOptions;

typedef struct {
  JxlBasicInfo basic_info;
  size_t frame_count;
  size_t frame_bytes;
  uint8_t* pixels;
  size_t pixels_size;
  uint8_t* ec0_pixels;
  size_t ec0_size;
} CHarnessResult;

void c_harness_abi_sizes(size_t* opts_size, size_t* result_size) {
  if (opts_size) {
    *opts_size = sizeof(CHarnessOptions);
  }
  if (result_size) {
    *result_size = sizeof(CHarnessResult);
  }
}

void c_harness_free_result(CHarnessResult* res) {
  if (!res) {
    return;
  }
  free(res->pixels);
  free(res->ec0_pixels);
  memset(res, 0, sizeof(*res));
}

static size_t bytes_per_sample_for(JxlDataType dt) {
  switch (dt) {
    case JXL_TYPE_FLOAT:
      return 4;
    case JXL_TYPE_UINT8:
      return 1;
    case JXL_TYPE_UINT16:
    case JXL_TYPE_FLOAT16:
      return 2;
    default:
      return 0;
  }
}

typedef struct {
  size_t width;
  size_t height;
  size_t bpp;
  size_t sample_align;
  uint8_t* frame_buf;
  uint8_t* visited;
  bool error;
  int init_calls;
  int destroy_calls;
} CallbackCapture;

typedef struct {
  CallbackCapture* cap;
  size_t num_threads;
  size_t num_pixels_per_thread;
} MtRunState;

static void write_stripe(CallbackCapture* cap, size_t x, size_t y,
                         size_t num_pixels, const void* pixels) {
  if (y >= cap->height || x + num_pixels > cap->width ||
      ((uintptr_t)pixels % cap->sample_align) != 0) {
    cap->error = true;
    return;
  }
  const size_t offset_px = y * cap->width + x;
  memcpy(cap->frame_buf + offset_px * cap->bpp, pixels, num_pixels * cap->bpp);
  for (size_t i = 0; i < num_pixels; ++i) {
    cap->visited[offset_px + i] += 1;
  }
}

static void simple_out_cb(void* opaque, size_t x, size_t y, size_t num_pixels,
                          const void* pixels) {
  write_stripe((CallbackCapture*)opaque, x, y, num_pixels, pixels);
}

static void* mt_init_cb(void* init_opaque, size_t num_threads,
                        size_t num_pixels_per_thread) {
  CallbackCapture* cap = (CallbackCapture*)init_opaque;
  cap->init_calls += 1;
  if (num_threads == 0 || num_pixels_per_thread == 0) {
    cap->error = true;
    return NULL;
  }
  MtRunState* state = (MtRunState*)malloc(sizeof(MtRunState));
  if (!state) {
    return NULL;
  }
  state->cap = cap;
  state->num_threads = num_threads;
  state->num_pixels_per_thread = num_pixels_per_thread;
  return state;
}

static void mt_run_cb(void* run_opaque, size_t thread_id, size_t x, size_t y,
                      size_t num_pixels, const void* pixels) {
  MtRunState* state = (MtRunState*)run_opaque;
  if (thread_id >= state->num_threads ||
      num_pixels > state->num_pixels_per_thread) {
    state->cap->error = true;
    return;
  }
  write_stripe(state->cap, x, y, num_pixels, pixels);
}

static void mt_destroy_cb(void* run_opaque) {
  MtRunState* state = (MtRunState*)run_opaque;
  if (state) {
    state->cap->destroy_calls += 1;
    free(state);
  }
}

static int append_bytes(uint8_t** dst, size_t* dst_len, const uint8_t* src,
                        size_t src_len) {
  if (src_len == 0) {
    return 0;
  }
  uint8_t* next = (uint8_t*)realloc(*dst, *dst_len + src_len);
  if (!next) {
    return -1;
  }
  memcpy(next + *dst_len, src, src_len);
  *dst = next;
  *dst_len += src_len;
  return 0;
}

typedef struct {
  const uint8_t* data;
  size_t data_size;
  size_t pos;
  const CHarnessOptions* opts;
  CHarnessResult* out;

  JxlDecoder* dec;
  void* thread_runner;
  void* resizable_runner;

  size_t bps;
  size_t bpp;
  bool got_basic_info;
  size_t bytes_per_row;
  size_t stride;
  size_t strided_size;
  size_t ec0_frame_size;

  uint8_t* strided_buf;
  uint8_t* packed_frame;
  uint8_t* visited;
  uint8_t* ec0_frame;
  CallbackCapture cap;
} DecodeSession;

static void session_destroy(DecodeSession* s) {
  free(s->strided_buf);
  free(s->packed_frame);
  free(s->visited);
  free(s->ec0_frame);
  if (s->thread_runner) {
    JxlThreadParallelRunnerDestroy(s->thread_runner);
  }
  if (s->resizable_runner) {
    JxlResizableParallelRunnerDestroy(s->resizable_runner);
  }
  JxlDecoderDestroy(s->dec);
}

static int configure_decoder(DecodeSession* s) {
  const CHarnessOptions* opts = s->opts;
  if (opts->runner_mode == C_RUNNER_THREAD) {
    s->thread_runner = JxlThreadParallelRunnerCreate(NULL, opts->num_threads);
    if (!s->thread_runner ||
        JxlDecoderSetParallelRunner(s->dec, JxlThreadParallelRunner,
                                    s->thread_runner) != JXL_DEC_SUCCESS) {
      return -4;
    }
  } else if (opts->runner_mode == C_RUNNER_RESIZABLE) {
    s->resizable_runner = JxlResizableParallelRunnerCreate(NULL);
    if (!s->resizable_runner ||
        JxlDecoderSetParallelRunner(s->dec, JxlResizableParallelRunner,
                                    s->resizable_runner) != JXL_DEC_SUCCESS) {
      return -5;
    }
  }

  if (JxlDecoderSetKeepOrientation(
          s->dec, opts->keep_orientation ? JXL_TRUE : JXL_FALSE) !=
          JXL_DEC_SUCCESS ||
      JxlDecoderSetUnpremultiplyAlpha(
          s->dec, opts->unpremultiply_alpha ? JXL_TRUE : JXL_FALSE) !=
          JXL_DEC_SUCCESS) {
    return -6;
  }

  if (JxlDecoderSubscribeEvents(s->dec, JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                                            JXL_DEC_FULL_IMAGE) !=
      JXL_DEC_SUCCESS) {
    return -7;
  }

  if (opts->chunk_size == 0) {
    if (JxlDecoderSetInput(s->dec, s->data, s->data_size) != JXL_DEC_SUCCESS) {
      return -8;
    }
    s->pos = s->data_size;
    JxlDecoderCloseInput(s->dec);
  }
  return 0;
}

static int feed_next_chunk(DecodeSession* s) {
  if (s->opts->chunk_size == 0) {
    return -9;
  }
  s->pos -= JxlDecoderReleaseInput(s->dec);
  if (s->pos >= s->data_size) {
    return -10;
  }
  size_t next_len = s->data_size - s->pos;
  if (next_len > s->opts->chunk_size) {
    next_len = s->opts->chunk_size;
  }
  if (JxlDecoderSetInput(s->dec, s->data + s->pos, next_len) !=
      JXL_DEC_SUCCESS) {
    return -11;
  }
  s->pos += next_len;
  if (s->pos == s->data_size) {
    JxlDecoderCloseInput(s->dec);
  }
  return 0;
}

static int handle_basic_info(DecodeSession* s) {
  JxlBasicInfo* info = &s->out->basic_info;
  if (JxlDecoderGetBasicInfo(s->dec, info) != JXL_DEC_SUCCESS) {
    return -12;
  }
  s->got_basic_info = true;

  if (s->resizable_runner) {
    uint32_t suggested =
        JxlResizableParallelRunnerSuggestThreads(info->xsize, info->ysize);
    JxlResizableParallelRunnerSetThreads(
        s->resizable_runner,
        s->opts->num_threads > 0 ? s->opts->num_threads : (size_t)suggested);
  }

  const JxlPixelFormat* fmt = &s->opts->format;
  s->bytes_per_row = (size_t)info->xsize * s->bpp;
  s->stride = s->bytes_per_row;
  if (fmt->align > 1) {
    size_t rem = s->bytes_per_row % fmt->align;
    if (rem != 0) {
      s->stride += fmt->align - rem;
    }
  }
  s->out->frame_bytes = s->bytes_per_row * (size_t)info->ysize;

  size_t queried_size = 0;
  if (JxlDecoderImageOutBufferSize(s->dec, fmt, &queried_size) !=
      JXL_DEC_SUCCESS) {
    return -13;
  }
  size_t expected_strided = info->ysize == 0
                                ? 0
                                : (size_t)(info->ysize - 1) * s->stride +
                                      s->bytes_per_row;
  if (queried_size != expected_strided) {
    return -14;
  }
  s->strided_size = queried_size;

  size_t num_px = (size_t)info->xsize * (size_t)info->ysize;
  s->strided_buf = (uint8_t*)malloc(s->strided_size > 0 ? s->strided_size : 1);
  s->packed_frame =
      (uint8_t*)malloc(s->out->frame_bytes > 0 ? s->out->frame_bytes : 1);
  s->visited = (uint8_t*)malloc(num_px > 0 ? num_px : 1);
  if (!s->strided_buf || !s->packed_frame || !s->visited) {
    return -15;
  }

  if (s->opts->decode_ec0) {
    JxlPixelFormat ec_fmt = {1, fmt->data_type, fmt->endianness, 0};
    if (JxlDecoderExtraChannelBufferSize(s->dec, &ec_fmt, &s->ec0_frame_size,
                                         0) != JXL_DEC_SUCCESS) {
      return -16;
    }
    s->ec0_frame =
        (uint8_t*)malloc(s->ec0_frame_size > 0 ? s->ec0_frame_size : 1);
    if (!s->ec0_frame) {
      return -17;
    }
  }
  return 0;
}

static int prepare_frame_outputs(DecodeSession* s) {
  if (!s->got_basic_info) {
    return -19;
  }
  const JxlPixelFormat* fmt = &s->opts->format;
  if (s->opts->decode_ec0) {
    JxlPixelFormat ec_fmt = {1, fmt->data_type, fmt->endianness, 0};
    memset(s->ec0_frame, 0, s->ec0_frame_size);
    if (JxlDecoderSetExtraChannelBuffer(s->dec, &ec_fmt, s->ec0_frame,
                                        s->ec0_frame_size,
                                        0) != JXL_DEC_SUCCESS) {
      return -20;
    }
  }

  if (s->opts->output_mode == C_OUT_MODE_BUFFER) {
    memset(s->strided_buf, 0, s->strided_size);
    if (JxlDecoderSetImageOutBuffer(s->dec, fmt, s->strided_buf,
                                    s->strided_size) != JXL_DEC_SUCCESS) {
      return -21;
    }
    return 0;
  }

  const JxlBasicInfo* info = &s->out->basic_info;
  memset(s->packed_frame, 0, s->out->frame_bytes);
  memset(s->visited, 0, (size_t)info->xsize * (size_t)info->ysize);
  s->cap = (CallbackCapture){
      .width = info->xsize,
      .height = info->ysize,
      .bpp = s->bpp,
      .sample_align = s->bps,
      .frame_buf = s->packed_frame,
      .visited = s->visited,
  };

  if (s->opts->output_mode == C_OUT_MODE_SIMPLE_CALLBACK) {
    return JxlDecoderSetImageOutCallback(s->dec, fmt, simple_out_cb, &s->cap) ==
                   JXL_DEC_SUCCESS
               ? 0
               : -22;
  }
  if (s->opts->output_mode == C_OUT_MODE_MT_CALLBACK) {
    return JxlDecoderSetMultithreadedImageOutCallback(
               s->dec, fmt, mt_init_cb, mt_run_cb, mt_destroy_cb, &s->cap) ==
                   JXL_DEC_SUCCESS
               ? 0
               : -23;
  }
  return -24;
}

static int collect_frame_outputs(DecodeSession* s) {
  const JxlBasicInfo* info = &s->out->basic_info;
  if (s->opts->output_mode == C_OUT_MODE_BUFFER) {
    for (size_t y = 0; y < (size_t)info->ysize; ++y) {
      memcpy(s->packed_frame + y * s->bytes_per_row,
             s->strided_buf + y * s->stride, s->bytes_per_row);
    }
  } else {
    if (s->cap.error) {
      return -25;
    }
    size_t total_px = (size_t)info->xsize * (size_t)info->ysize;
    for (size_t i = 0; i < total_px; ++i) {
      if (s->visited[i] != 1) {
        return -26;
      }
    }
    if (s->opts->output_mode == C_OUT_MODE_MT_CALLBACK &&
        (s->cap.init_calls != 1 || s->cap.destroy_calls != 1)) {
      return -27;
    }
  }

  if (append_bytes(&s->out->pixels, &s->out->pixels_size, s->packed_frame,
                   s->out->frame_bytes) != 0) {
    return -28;
  }
  if (s->opts->decode_ec0 &&
      append_bytes(&s->out->ec0_pixels, &s->out->ec0_size, s->ec0_frame,
                   s->ec0_frame_size) != 0) {
    return -29;
  }
  s->out->frame_count += 1;
  return 0;
}

int c_harness_decode(const uint8_t* data, size_t data_size,
                     const CHarnessOptions* opts, CHarnessResult* out) {
  if (out) {
    memset(out, 0, sizeof(*out));
  }
  if (!data || !opts || !out) {
    return -1;
  }

  const size_t bps = bytes_per_sample_for(opts->format.data_type);
  if (bps == 0) {
    return -2;
  }

  DecodeSession s;
  memset(&s, 0, sizeof(s));
  s.data = data;
  s.data_size = data_size;
  s.opts = opts;
  s.out = out;
  s.bps = bps;
  s.bpp = (size_t)opts->format.num_channels * bps;
  s.dec = JxlDecoderCreate(NULL);
  if (!s.dec) {
    return -3;
  }

  int rc = configure_decoder(&s);
  while (rc == 0) {
    JxlDecoderStatus status = JxlDecoderProcessInput(s.dec);
    if (status == JXL_DEC_SUCCESS) {
      break;
    }
    switch (status) {
      case JXL_DEC_NEED_MORE_INPUT:
        rc = feed_next_chunk(&s);
        break;
      case JXL_DEC_BASIC_INFO:
        rc = handle_basic_info(&s);
        break;
      case JXL_DEC_FRAME: {
        JxlFrameHeader fh;
        if (!s.got_basic_info ||
            JxlDecoderGetFrameHeader(s.dec, &fh) != JXL_DEC_SUCCESS) {
          rc = -18;
        }
        break;
      }
      case JXL_DEC_NEED_IMAGE_OUT_BUFFER:
        rc = prepare_frame_outputs(&s);
        break;
      case JXL_DEC_FULL_IMAGE:
        rc = collect_frame_outputs(&s);
        break;
      default:
        rc = -30;
        break;
    }
  }

  if (rc == 0 && (!s.got_basic_info || out->frame_count == 0)) {
    rc = -31;
  }

  session_destroy(&s);
  if (rc != 0) {
    c_harness_free_result(out);
  }
  return rc;
}

int c_harness_exercise_misc_apis(void) {
#if JPEGXL_NUMERIC_VERSION < JPEGXL_COMPUTE_NUMERIC_VERSION(0, 11, 0)
  return -1;
#endif
  if (JxlDecoderVersion() == 0 || JxlDecoderVersion() != JxlEncoderVersion()) {
    return -2;
  }

  const uint8_t codestream[4] = {0xff, 0x0a, 0x00, 0x00};
  if (JxlSignatureCheck(codestream, sizeof(codestream)) != JXL_SIG_CODESTREAM) {
    return -3;
  }

  JxlColorEncoding srgb;
  JxlColorEncoding lin;
  JxlColorEncodingSetToSRGB(&srgb, JXL_FALSE);
  JxlColorEncodingSetToLinearSRGB(&lin, JXL_TRUE);
  if (srgb.transfer_function != JXL_TRANSFER_FUNCTION_SRGB ||
      lin.transfer_function != JXL_TRANSFER_FUNCTION_LINEAR ||
      lin.color_space != JXL_COLOR_SPACE_GRAY) {
    return -4;
  }

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  JxlFrameHeader fh;
  JxlEncoderInitFrameHeader(&fh);
  JxlBlendInfo bi;
  JxlEncoderInitBlendInfo(&bi);
  JxlExtraChannelInfo eci;
  JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_ALPHA, &eci);
  if (info.bits_per_sample != 8 || info.num_color_channels != 3 ||
      bi.blendmode != JXL_BLEND_REPLACE || eci.type != JXL_CHANNEL_ALPHA) {
    return -5;
  }

  const JxlCmsInterface* cms = JxlGetDefaultCms();
  if (!cms) {
    return -6;
  }
  JxlDecoder* dec = JxlDecoderCreate(NULL);
  if (!dec || JxlDecoderSetCms(dec, *cms) != JXL_DEC_ERROR) {
    JxlDecoderDestroy(dec);
    return -7;
  }
  JxlDecoderDestroy(dec);

  if (JxlICCProfileEncode(NULL, NULL, 0, NULL, NULL) != JXL_FALSE ||
      JxlICCProfileDecode(NULL, NULL, 0, NULL, NULL) != JXL_FALSE ||
      JxlGainMapReadBundle(NULL, NULL, 0, NULL) != JXL_FALSE ||
      JxlGainMapWriteBundle(NULL, NULL, 0, NULL) != JXL_FALSE ||
      JxlGainMapGetBundleSize(NULL, NULL) != JXL_FALSE) {
    return -8;
  }

  if (JxlEncoderCreate(NULL) != NULL || JxlEncoderStatsCreate() != NULL ||
      JxlThreadParallelRunnerDefaultNumWorkerThreads() == 0) {
    return -9;
  }

  return 0;
}
