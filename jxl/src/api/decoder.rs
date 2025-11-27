#[test]
    fn test_refill_with_potential_overlap() {
        // This test is designed to create a situation where SmallBuffer::refill
        // might have overlapping source and destination ranges if implemented
        // with a naive copy.
        // The chunk size is specifically chosen to be small to trigger the
        // buffer refill logic frequently.
        decode(
            &std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap(),
            1, // small chunk size
            false,
            None,
        )
        .unwrap();
    }