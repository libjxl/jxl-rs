# Root Cause Analysis: 6 Test Failures (candle, dice, alpha_premultiplied)

## Problem
Tests fail with `NaN at 0 0` for images with RGB+Alpha (4 channels):
- `api::decoder::tests::decode_test_file_candle`
- `api::decoder::tests::decode_test_file_dice`
- `api::decoder::tests::decode_test_file_alpha_premultiplied`
- `api::decoder::tests::compare_pipelines_candle`
- `api::decoder::tests::compare_pipelines_dice`
- `api::decoder::tests::compare_pipelines_alpha_premultiplied`

## Root Cause

Commit `ab25d62` "Add buffer bounds checking to fix special image types" added this code to `jxl/src/render/low_memory_pipeline/save/mod.rs`:

```rust
// Bounds check: if buffer index is out of range, skip save
if self.output_buffer_index >= buffers.len() {
    return Ok(());
}
```

This **silently skips** saving data when the buffer index is out of range. For RGB+Alpha images:
- The pipeline tries to save alpha channel data to `buffers[some_index]`
- But `buffers` array doesn't have enough elements
- The bounds check **silently returns Ok()** without writing anything
- Alpha channel buffer remains filled with NaN values (initial state)
- Test detects NaN and fails

## Why This Happened

1. The commit was trying to fix index out of bounds panics for special image types (patches, cmyk, blendmodes)
2. Instead of fixing the buffer allocation, it masked the problem with a silent skip
3. This "fix" caused a NEW problem: alpha channels aren't written, leaving NaN

## Comparison with Baseline

- **origin/main**: These same tests have **stack overflow** (different symptom, same underlying issue)
- **Our PR**: After ab25d62, tests have **NaN errors** (bounds check masks the real problem)
- Neither version works correctly!

## The Real Fix Needed

Don't silently skip! Instead:

1. **Option A**: Ensure `buffers` array is allocated with correct size for all channels
   - For RGB+Alpha: need 4 buffers (or 1 interleaved + 1 alpha)
   - For RGB: need 3 buffers (or 1 interleaved)

2. **Option B**: Fix `output_buffer_index` calculation to match actual buffer layout

3. **Option C**: Remove the silent skip and fix the actual allocation issue that causes out-of-bounds

The silent skip in ab25d62 is a **symptom-hiding hack** that must be removed or fixed properly.

## Files to Check

1. `jxl/src/render/low_memory_pipeline/save/mod.rs` - has the silent skip
2. `jxl/src/render/simple_pipeline/save.rs` - also has silent skip (per commit)
3. `jxl/src/render/low_memory_pipeline/mod.rs` - check_buffer_sizes (per commit)
4. `jxl/src/api/decoder.rs` lines 281-294 - buffer allocation logic

## Test Images

All failing images have **RGB+Alpha**:
- candle.jxl: 1000x810, lossy, 16-bit, RGB+Alpha, float
- dice.jxl: 800x600, lossy, 8-bit, RGB+Alpha
- alpha_premultiplied.jxl: 1024x1024, lossy, 12-bit, RGB+Alpha

Passing images like basic.jxl have only **RGB** (no alpha).

## Next Steps

1. Temporarily remove the silent skip to see the actual error
2. Fix the underlying buffer allocation or indexing issue
3. Ensure tests pass without needing the silent skip hack
