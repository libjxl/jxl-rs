
use jxl::api::{JxlDecoder, ProcessingResult};

#[test]
fn test_decode_modular_frame_header() {
    let file_contents = include_bytes!("../resources/test/extra_channels.jxl");
    let mut input: &[u8] = file_contents;

    let decoder = JxlDecoder::new(Default::default());

    let mut decoder = match decoder.process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("Needs more input for header"),
    };

    loop {
        let frame_decoder = match decoder.process(&mut input).unwrap() {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => panic!("Needs more input for frame"),
        };

        decoder = match frame_decoder.process(&mut input, &mut []).unwrap() {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => panic!("Needs more input on frame decode"),
        };

        if !decoder.has_more_frames() {
            break;
        }
    }
}
