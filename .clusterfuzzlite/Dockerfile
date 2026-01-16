FROM gcr.io/oss-fuzz-base/base-builder-rust:v1

# Copy the project source
COPY . $SRC/jxl-rs
WORKDIR $SRC/jxl-rs

# Copy build script
COPY .clusterfuzzlite/build.sh $SRC/
