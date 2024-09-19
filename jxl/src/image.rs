use crate::error::{Error, Result};

mod private {
    pub trait Sealed {}
}

pub trait ImageDataType: private::Sealed + Copy + Default + 'static {
    fn data_type_id() -> usize;
}

macro_rules! impl_image_data_type {
    ($ty: ty, $id: literal) => {
        impl private::Sealed for $ty {}
        impl ImageDataType for $ty {
            fn data_type_id() -> usize {
                $id
            }
        }
    };
}

impl_image_data_type!(u8, 0);
impl_image_data_type!(u16, 1);
impl_image_data_type!(u32, 2);
impl_image_data_type!(f32, 3);
impl_image_data_type!(i8, 4);
impl_image_data_type!(i16, 5);
impl_image_data_type!(i32, 6);
impl_image_data_type!(half::f16, 7);

pub struct Image<T: ImageDataType> {
    size: (usize, usize),
    data: Vec<T>,
}

#[derive(Clone, Copy)]
pub struct ImageRect<'a, T: ImageDataType> {
    origin: (usize, usize),
    size: (usize, usize),
    image: &'a Image<T>,
}

pub struct ImageRectMut<'a, T: ImageDataType> {
    origin: (usize, usize),
    size: (usize, usize),
    image: &'a mut Image<T>,
}

impl<T: ImageDataType> Image<T> {
    pub fn new(xsize: usize, ysize: usize) -> Result<Image<T>> {
        let total_size = xsize
            .checked_mul(ysize)
            .ok_or(Error::ImageSizeTooLarge(xsize, ysize))?;
        let mut data = vec![];
        data.try_reserve_exact(total_size)?;
        data.resize(total_size, T::default());
        Ok(Image {
            size: (xsize, ysize),
            data,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn as_rect(&self) -> ImageRect<'_, T> {
        ImageRect {
            origin: (0, 0),
            size: self.size,
            image: self,
        }
    }

    pub fn as_rect_mut(&mut self) -> ImageRectMut<'_, T> {
        ImageRectMut {
            origin: (0, 0),
            size: self.size,
            image: self,
        }
    }
}

fn rect_size_check(
    origin: (usize, usize),
    size: (usize, usize),
    ssize: (usize, usize),
) -> Result<()> {
    if origin
        .0
        .checked_add(size.0)
        .ok_or(Error::ArithmeticOverflow)?
        > ssize.0
        || origin
            .1
            .checked_add(size.1)
            .ok_or(Error::ArithmeticOverflow)?
            > ssize.1
    {
        Err(Error::RectOutOfBounds(
            size.0, size.1, origin.0, origin.1, ssize.0, ssize.1,
        ))
    } else {
        Ok(())
    }
}

impl<'a, T: ImageDataType> ImageRect<'a, T> {
    pub fn rect(
        &'a self,
        origin: (usize, usize),
        size: (usize, usize),
    ) -> Result<ImageRect<'a, T>> {
        rect_size_check(origin, size, self.size)?;
        Ok(ImageRect {
            origin: (origin.0 + self.origin.0, origin.1 + self.origin.1),
            size,
            image: self.image,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn row(&self, row: usize) -> &[T] {
        debug_assert!(row < self.size.1);
        let start = (row + self.origin.0) * self.image.size.1 + self.origin.1;
        &self.image.data[start..start + self.size.0]
    }

    pub fn to_image(&self) -> Result<Image<T>> {
        let total_size = self.size.0 * self.size.1;
        let mut data = vec![];
        data.try_reserve_exact(total_size)?;
        data.extend((0..self.size.1).flat_map(|x| self.row(x).iter()));
        Ok(Image {
            size: self.size,
            data,
        })
    }
}

impl<'a, T: ImageDataType> ImageRectMut<'a, T> {
    pub fn rect(
        &'a mut self,
        origin: (usize, usize),
        size: (usize, usize),
    ) -> Result<ImageRectMut<'a, T>> {
        rect_size_check(origin, size, self.size)?;
        Ok(ImageRectMut {
            origin: (origin.0 + self.origin.0, origin.1 + self.origin.1),
            size,
            image: self.image,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn row(&mut self, row: usize) -> &mut [T] {
        debug_assert!(row < self.size.1);
        let start = (row + self.origin.0) * self.image.size.1 + self.origin.1;
        &mut self.image.data[start..start + self.size.0]
    }

    pub fn as_rect(&'a self) -> ImageRect<'a, T> {
        ImageRect {
            origin: self.origin,
            size: self.size,
            image: self.image,
        }
    }
}

#[cfg(feature = "debug_tools")]
pub mod debug_tools {
    use super::{ImageDataType, ImageRect};
    pub trait ToU8ForWriting {
        fn to_u8_for_writing(self) -> u8;
    }

    impl ToU8ForWriting for u8 {
        fn to_u8_for_writing(self) -> u8 {
            self
        }
    }

    impl ToU8ForWriting for u16 {
        fn to_u8_for_writing(self) -> u8 {
            (self as u32 * 255 / 65535) as u8
        }
    }

    impl ToU8ForWriting for f32 {
        fn to_u8_for_writing(self) -> u8 {
            (self * 255.0).clamp(0.0, 255.0).round() as u8
        }
    }

    impl ToU8ForWriting for u32 {
        fn to_u8_for_writing(self) -> u8 {
            (self as f32 / (2.0f32.powi(32) - 1.0)).to_u8_for_writing()
        }
    }

    impl ToU8ForWriting for half::f16 {
        fn to_u8_for_writing(self) -> u8 {
            self.to_f32().to_u8_for_writing()
        }
    }

    impl<'a, T: ImageDataType + ToU8ForWriting> ImageRect<'a, T> {
        pub fn to_pgm(&self) -> Vec<u8> {
            use std::io::Write;
            let mut ret = vec![];
            write!(&mut ret, "P5\n{} {}\n255\n", self.size.0, self.size.1).unwrap();
            ret.extend(
                (0..self.size.1)
                    .flat_map(|x| self.row(x).iter())
                    .map(|x| x.to_u8_for_writing()),
            );
            ret
        }
    }
}

#[cfg(test)]
mod test {
    use crate::error::Result;

    use super::Image;

    #[test]
    fn huge_image() {
        assert!(Image::<u8>::new(1 << 28, 1 << 28).is_err());
    }

    #[test]
    fn rect_basic() -> Result<()> {
        let mut image = Image::<u8>::new(32, 42)?;
        assert_eq!(image.as_rect_mut().rect((31, 40), (1, 1))?.size(), (1, 1));
        assert_eq!(image.as_rect_mut().rect((0, 0), (1, 1))?.size(), (1, 1));
        assert!(image.as_rect_mut().rect((30, 30), (3, 3)).is_err());
        image.as_rect_mut().rect((30, 30), (1, 1))?.row(0)[0] = 1;
        assert_eq!(image.as_rect_mut().row(30)[30], 1);
        Ok(())
    }

    #[cfg(feature = "debug_tools")]
    #[test]
    fn to_pgm() -> Result<()> {
        let image = Image::<u8>::new(32, 32)?;
        assert!(image.as_rect().to_pgm().starts_with(b"P5\n32 32\n255\n"));
        Ok(())
    }
}
