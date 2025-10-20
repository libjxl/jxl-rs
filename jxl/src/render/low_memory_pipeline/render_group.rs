use crate::{api::JxlOutputBuffer, error::Result};

use super::LowMemoryRenderPipeline;

impl LowMemoryRenderPipeline {
    // Renders a single group worth of data.
    pub(super) fn render_group(
        &mut self,
        (gx, gy): (usize, usize),
        buffers: &mut [Option<JxlOutputBuffer>],
    ) -> Result<()> {
        if self.border_pixels != (0, 0) {
            // TODO(veluca): implement this case.
            unimplemented!()
        }

        for stage in self.shared.stages.iter() {
            // Step 1: read input channels.

            // Step 2: go through stages one by one.
        }
        Ok(())
    }
}
