// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/platform/image-decoders/jpeg/jpeg_rs_image_decoder.h"

#include "base/containers/span.h"
#include "third_party/blink/renderer/platform/image-decoders/fast_shared_buffer_reader.h"
#include "third_party/skia/include/core/SkColorSpace.h"
#include "third_party/skia/include/core/SkTypes.h"

// Include the generated CXX bridge header
#include "third_party/jxl-rs/jxl_chromium/src/jpeg.rs.h"

namespace blink {

using jpeg_rs::jpeg_rs_decoder_create;
using jpeg_rs::jpeg_rs_signature_check;
using jpeg_rs::JpegRsBasicInfo;
using jpeg_rs::JpegRsDecoder;
using jpeg_rs::JpegRsPixelFormat;
using jpeg_rs::JpegRsProcessResult;
using jpeg_rs::JpegRsStatus;

namespace {

// Maximum decoded pixels (same as JXL decoder for consistency)
constexpr uint64_t kMaxDecodedPixels = 1024ULL * 1024 * 1024;

}  // namespace

JPEGRsImageDecoder::JPEGRsImageDecoder(AlphaOption alpha_option,
                                       HighBitDepthDecodingOption hbd_option,
                                       ColorBehavior color_behavior,
                                       cc::AuxImage aux_image,
                                       wtf_size_t max_decoded_bytes)
    : ImageDecoder(alpha_option,
                   hbd_option,
                   color_behavior,
                   aux_image,
                   max_decoded_bytes) {
  basic_info_ = {};
}

JPEGRsImageDecoder::~JPEGRsImageDecoder() = default;

String JPEGRsImageDecoder::FilenameExtension() const {
  return "jpg";
}

const AtomicString& JPEGRsImageDecoder::MimeType() const {
  DEFINE_STATIC_LOCAL(const AtomicString, jpeg_mime_type, ("image/jpeg"));
  return jpeg_mime_type;
}

bool JPEGRsImageDecoder::ImageIsHighBitDepth() {
  return is_high_bit_depth_;
}

void JPEGRsImageDecoder::OnSetData(scoped_refptr<SegmentReader> data) {
  // Data accumulates automatically; decoding continues where it left off
}

bool JPEGRsImageDecoder::MatchesJPEGSignature(
    const FastSharedBufferReader& fast_reader) {
  uint8_t buffer[4];
  if (fast_reader.size() < sizeof(buffer)) {
    return false;
  }
  auto data = fast_reader.GetConsecutiveData(0, sizeof(buffer), buffer);
  return jpeg_rs_signature_check(
      rust::Slice<const uint8_t>(data.data(), data.size()));
}

void JPEGRsImageDecoder::DecodeSize() {
  Decode(0, /*only_size=*/true);
}

wtf_size_t JPEGRsImageDecoder::DecodeFrameCount() {
  // JPEG is always a single frame
  return 1;
}

void JPEGRsImageDecoder::InitializeNewFrame(wtf_size_t index) {
  DCHECK_EQ(index, 0u);
  DCHECK_LT(index, frame_buffer_cache_.size());

  auto& buffer = frame_buffer_cache_[index];

  // JPEG doesn't have alpha, but we decode to BGRA for Skia compatibility
  buffer.SetHasAlpha(false);
  buffer.SetOriginalFrameRect(gfx::Rect(Size()));
  buffer.SetRequiredPreviousFrameIndex(kNotFound);
}

void JPEGRsImageDecoder::Decode(wtf_size_t index) {
  Decode(index, false);
}

void JPEGRsImageDecoder::Decode(wtf_size_t index, bool only_size) {
  if (Failed()) {
    return;
  }

  // Early exit if already decoded
  if (only_size && IsDecodedSizeAvailable() && have_metadata_) {
    return;
  }

  if (!only_size && index < frame_buffer_cache_.size()) {
    auto status = frame_buffer_cache_[index].GetStatus();
    if (status == ImageFrame::kFrameComplete) {
      return;
    }
  }

  FastSharedBufferReader reader(data_.get());
  size_t data_size = reader.size();

  // Create decoder if needed
  if (!decoder_.has_value()) {
    decoder_ = jpeg_rs_decoder_create(kMaxDecodedPixels);
    decoder_state_ = DecoderState::kInitial;
    input_offset_ = 0;
  }

  // Read all available data
  Vector<uint8_t> data_buffer;
  size_t remaining = data_size - input_offset_;
  if (remaining > 0) {
    data_buffer.resize(remaining);
    auto data_span = reader.GetConsecutiveData(input_offset_, remaining,
                                               base::span(data_buffer));

    bool all_input = IsAllDataReceived();
    rust::Slice<const uint8_t> input_slice(data_span.data(), data_span.size());

    switch (decoder_state_) {
      case DecoderState::kInitial: {
        JpegRsProcessResult result =
            (*decoder_)->parse_headers(input_slice, all_input);

        if (result.status == JpegRsStatus::Error) {
          SetFailed();
          return;
        }
        if (result.status == JpegRsStatus::NeedMoreInput) {
          input_offset_ += result.bytes_consumed;
          return;
        }

        // Success - got basic info
        basic_info_ = (*decoder_)->get_basic_info();
        input_offset_ += result.bytes_consumed;

        if (!SetSize(basic_info_.width, basic_info_.height)) {
          return;
        }

        // Check for 12-bit JPEG
        if (basic_info_.bits_per_sample > 8) {
          is_high_bit_depth_ = true;
        }

        // Set pixel format - use BGRA8 for Skia compatibility
        (*decoder_)->set_pixel_format(JpegRsPixelFormat::Bgra8);

        // Extract ICC color profile
        if (!IgnoresColorSpace()) {
          auto icc_data = (*decoder_)->get_icc_profile();
          if (!icc_data.empty()) {
            Vector<uint8_t> icc_copy;
            icc_copy.AppendRange(icc_data.begin(), icc_data.end());
            auto profile = ColorProfile::Create(base::span(icc_copy));
            if (profile) {
              SetEmbeddedColorProfile(std::move(profile));
            }
          }
        }

        have_metadata_ = true;
        decoder_state_ = DecoderState::kHaveBasicInfo;

        if (only_size) {
          return;
        }

        [[fallthrough]];
      }

      case DecoderState::kHaveBasicInfo: {
        // Initialize frame buffer
        if (frame_buffer_cache_.empty()) {
          frame_buffer_cache_.resize(1);
        }

        if (!InitFrameBuffer(0)) {
          SetFailed();
          return;
        }

        ImageFrame& frame = frame_buffer_cache_[0];
        frame.SetHasAlpha(false);  // JPEG has no alpha

        const uint32_t width = basic_info_.width;
        const uint32_t height = basic_info_.height;

        // Get direct access to frame buffer
        const SkBitmap& bitmap = frame.Bitmap();
        uint8_t* frame_pixels = static_cast<uint8_t*>(bitmap.getPixels());
        size_t row_stride = bitmap.rowBytes();

        if (!frame_pixels) {
          SetFailed();
          return;
        }

        size_t buffer_size = row_stride * height;
        rust::Slice<uint8_t> output_slice(frame_pixels, buffer_size);

        // Decode directly into frame buffer
        JpegRsProcessResult result = (*decoder_)->decode_image_with_stride(
            input_slice, all_input, output_slice, row_stride);

        if (result.status == JpegRsStatus::Error) {
          SetFailed();
          return;
        }
        if (result.status == JpegRsStatus::NeedMoreInput) {
          input_offset_ += result.bytes_consumed;
          frame.SetPixelsChanged(true);
          return;
        }

        input_offset_ += result.bytes_consumed;
        frame.SetPixelsChanged(true);
        frame.SetStatus(ImageFrame::kFrameComplete);
        decoder_state_ = DecoderState::kComplete;
        return;
      }

      case DecoderState::kComplete:
        // Already done
        return;

      case DecoderState::kError:
      case DecoderState::kDecoding:
        return;
    }
  }
}

bool JPEGRsImageDecoder::CanReusePreviousFrameBuffer(
    wtf_size_t frame_index) const {
  return true;
}

SkColorType JPEGRsImageDecoder::GetSkColorType() const {
  // We always decode JPEG to BGRA8 (kN32_SkColorType on little-endian)
  return kN32_SkColorType;
}

}  // namespace blink
