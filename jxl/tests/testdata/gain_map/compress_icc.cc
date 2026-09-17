// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_icc_codec.h"
#include "lib/jxl/memory_manager_internal.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;

  std::ifstream input(argv[1], std::ios::binary);
  if (!input) return 3;
  const std::vector<uint8_t> profile(
      (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

  JxlMemoryManager memory_manager{};
  if (!jxl::MemoryManagerInit(&memory_manager, nullptr)) return 4;

  jxl::BitWriter writer(&memory_manager);
  if (!jxl::WriteICC(jxl::Span<const uint8_t>(profile), &writer,
                     jxl::LayerType::Header, nullptr)) {
    return 5;
  }
  writer.ZeroPadToByte();

  const auto compressed = writer.GetSpan();
  std::ofstream output(argv[2], std::ios::binary);
  if (!output) return 6;
  output.write(reinterpret_cast<const char*>(compressed.data()),
               static_cast<std::streamsize>(compressed.size()));
  return output.good() ? 0 : 7;
}
