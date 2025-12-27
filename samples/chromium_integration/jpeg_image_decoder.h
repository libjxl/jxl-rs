// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_PLATFORM_IMAGE_DECODERS_JPEG_JPEG_RS_IMAGE_DECODER_H_
#define THIRD_PARTY_BLINK_RENDERER_PLATFORM_IMAGE_DECODERS_JPEG_JPEG_RS_IMAGE_DECODER_H_

#include <optional>

#include "third_party/blink/renderer/platform/image-decoders/image_decoder.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

// Forward declaration for the Rust decoder
namespace blink::jpeg_rs {
struct JpegRsDecoder;
struct JpegRsBasicInfo;
struct JpegRsProcessResult;
enum class JpegRsStatus : uint8_t;
enum class JpegRsPixelFormat : uint8_t;
}  // namespace blink::jpeg_rs

namespace blink {

class FastSharedBufferReader;

// JPEG image decoder using jxl-rs (Rust) as backend.
// This is designed to be a drop-in replacement for libjpeg-turbo.
class PLATFORM_EXPORT JPEGRsImageDecoder final : public ImageDecoder {
 public:
  JPEGRsImageDecoder(AlphaOption alpha_option,
                     HighBitDepthDecodingOption hbd_option,
                     ColorBehavior color_behavior,
                     cc::AuxImage aux_image,
                     wtf_size_t max_decoded_bytes);
  ~JPEGRsImageDecoder() override;

  // ImageDecoder implementation
  String FilenameExtension() const override;
  const AtomicString& MimeType() const override;
  bool ImageIsHighBitDepth() override;

  // Check if data matches JPEG signature
  static bool MatchesJPEGSignature(const FastSharedBufferReader& fast_reader);

 private:
  // ImageDecoder implementation
  void OnSetData(scoped_refptr<SegmentReader> data) override;
  void DecodeSize() override;
  wtf_size_t DecodeFrameCount() override;
  void InitializeNewFrame(wtf_size_t index) override;
  void Decode(wtf_size_t index) override;
  bool CanReusePreviousFrameBuffer(wtf_size_t frame_index) const override;

  // Internal decode helper
  void Decode(wtf_size_t index, bool only_size);

  // Get appropriate Skia color type
  SkColorType GetSkColorType() const;

  // Decoder state
  enum class DecoderState {
    kInitial,
    kHaveBasicInfo,
    kDecoding,
    kComplete,
    kError
  };

  std::optional<rust::Box<jpeg_rs::JpegRsDecoder>> decoder_;
  DecoderState decoder_state_ = DecoderState::kInitial;
  jpeg_rs::JpegRsBasicInfo basic_info_;
  bool have_metadata_ = false;
  bool is_high_bit_depth_ = false;
  size_t input_offset_ = 0;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_PLATFORM_IMAGE_DECODERS_JPEG_JPEG_RS_IMAGE_DECODER_H_
