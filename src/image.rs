//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Image handling.
//

use avi;
use bmp;
use defs::{Rect, Point};
use std::any::Any;
use std::cmp::{min, max};
use std::convert::From;
use std::default::Default;
use std::path::Path;
use std::ptr;
use std::slice;
use tiff;
use utils;


#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FileType {
    /// Determined automatically from file name extension
    Auto,
    Bmp,
    Tiff
}


fn get_file_type_from_ext(file_name: &str) -> FileType {
    match Path::new(file_name).extension() {
        Some(ext) => match ext.to_str().unwrap().to_lowercase().as_str() {
                         "bmp" => FileType::Bmp,
                         "tif" | "tiff" => FileType::Tiff,
                         _ => panic!()
                     },
        _ => panic!()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    /// 8 bits per pixel, values from a 256-entry palette.
    Pal8,
    Mono8,
    /// LSB = R, MSB = B.
    RGB8,
    /// LSB = B, MSB = R.
    BGR8,
    /// LSB = B, MSB = A or unused.
    BGRA8,

    Mono16,
    RGB16,
    RGBA16,

    Mono32f,
    RGB32f,

    Mono64f,
    RGB64f,

    CfaRGGB8,
    CfaGRBG8,
    CfaGBRG8,
    CfaBGGR8,

    CfaRGGB16,
    CfaGRBG16,
    CfaGBRG16,
    CfaBGGR16
}


impl PixelFormat {
    /// Returns true if this is a Color Filter Array pixel format.
    pub fn is_cfa(&self) -> bool {
        [PixelFormat::CfaRGGB8,
         PixelFormat::CfaGRBG8,
         PixelFormat::CfaGBRG8,
         PixelFormat::CfaBGGR8,
         PixelFormat::CfaRGGB16,
         PixelFormat::CfaGRBG16,
         PixelFormat::CfaGBRG16,
         PixelFormat::CfaBGGR16].contains(&self)
    }
}


/// Demosaicing (debayering) method using when converting CFA images to Mono/RGB.
pub enum DemosaicMethod {
    /// Fast, but low-quality.
    ///
    /// Used internally during image alignment, quality estimation and ref. point alignment.
    ///
    Simple,

    /// High-quality and slower; used internally during stacking phase.
    HqLinear,
}


/// Asserts that `T` is the type of pixel values (in each channel) corresponding to `pix_fmt`.
fn verify_pix_type<T: Default + Any>(pix_fmt: PixelFormat) {
    let t = &T::default() as &dyn Any;
    match pix_fmt {
        PixelFormat::Pal8     |
        PixelFormat::Mono8    |
        PixelFormat::RGB8     |
        PixelFormat::BGR8     |
        PixelFormat::BGRA8    |
        PixelFormat::CfaRGGB8 |
        PixelFormat::CfaGRBG8 |
        PixelFormat::CfaGBRG8 |
        PixelFormat::CfaBGGR8 => assert!(t.is::<u8>()),

        PixelFormat::Mono16    |
        PixelFormat::RGB16     |
        PixelFormat::RGBA16    |
        PixelFormat::CfaRGGB16 |
        PixelFormat::CfaGRBG16 |
        PixelFormat::CfaGBRG16 |
        PixelFormat::CfaBGGR16 => assert!(t.is::<u16>()),

        PixelFormat::Mono32f | PixelFormat::RGB32f => assert!(t.is::<f32>()),

        PixelFormat::Mono64f | PixelFormat::RGB64f => assert!(t.is::<f64>()),
    }}


pub fn get_num_channels(pix_fmt: PixelFormat) -> usize {
    match pix_fmt {
        PixelFormat::Pal8      |
        PixelFormat::Mono8     |
        PixelFormat::Mono16    |
        PixelFormat::Mono32f   |
        PixelFormat::Mono64f   |
        PixelFormat::CfaRGGB8  |
        PixelFormat::CfaGRBG8  |
        PixelFormat::CfaGBRG8  |
        PixelFormat::CfaBGGR8  |
        PixelFormat::CfaRGGB16 |
        PixelFormat::CfaGRBG16 |
        PixelFormat::CfaGBRG16 |
        PixelFormat::CfaBGGR16 => 1,

        PixelFormat::RGB8   |
        PixelFormat::BGR8   |
        PixelFormat::RGB16  |
        PixelFormat::RGB32f |
        PixelFormat::RGB64f => 3,

        PixelFormat::BGRA8 |
        PixelFormat::RGBA16 => 4
    }
}

pub fn bytes_per_pixel(pix_fmt: PixelFormat) -> usize {
    match pix_fmt {
        PixelFormat::Pal8 | PixelFormat::Mono8 => 1,
        PixelFormat::RGB8 | PixelFormat::BGR8  => 3,
        PixelFormat::BGRA8   => 4,
        PixelFormat::Mono16  => 2,
        PixelFormat::RGB16   => 6,
        PixelFormat::RGBA16  => 8,
        PixelFormat::Mono32f => 4,
        PixelFormat::RGB32f  => 16,
        PixelFormat::Mono64f => 8,
        PixelFormat::RGB64f  => 24,

        _ => panic!()
    }
}


pub fn bytes_per_channel(pix_fmt: PixelFormat) -> usize {
    match pix_fmt {
        PixelFormat::Pal8     |
        PixelFormat::Mono8    |
        PixelFormat::RGB8     |
        PixelFormat::BGR8     |
        PixelFormat::BGRA8    |
        PixelFormat::CfaRGGB8 |
        PixelFormat::CfaGRBG8 |
        PixelFormat::CfaGBRG8 |
        PixelFormat::CfaBGGR8 => 1,

        PixelFormat::Mono16    |
        PixelFormat::RGB16     |
        PixelFormat::RGBA16    |
        PixelFormat::CfaRGGB16 |
        PixelFormat::CfaGRBG16 |
        PixelFormat::CfaGBRG16 |
        PixelFormat::CfaBGGR16 => 2,

        PixelFormat::Mono32f | PixelFormat::RGB32f => 4,

        PixelFormat::Mono64f | PixelFormat::RGB64f => 8,
    }
}



#[derive(Copy)]
pub struct Palette {
    pub pal: [u8; 3 * Palette::NUM_ENTRIES]
}


impl Palette {
    pub const NUM_ENTRIES: usize = 256;
}


impl Clone for Palette {
    fn clone(&self) -> Palette { *self }
}


impl Default for Palette {
    fn default() -> Palette { Palette{ pal: [0; 3 * Palette::NUM_ENTRIES] }}
}


pub struct Image {
    width: u32,
    height: u32,
    pix_fmt: PixelFormat,
    palette: Option<Palette>,
    pixels: Vec<u8>,
    bytes_per_line: usize
}


#[derive(Debug)]
pub enum ImageError {
    AviError(avi::AviError),
    BmpError(bmp::BmpError),
    TiffError(tiff::TiffError)
}


impl Image {
    pub fn load(file_name: &str, file_type: FileType) -> Result<Image, ImageError> {
        let ftype = if file_type == FileType::Auto { get_file_type_from_ext(file_name) } else { file_type };
        match ftype {
            FileType::Bmp => bmp::load_bmp(file_name).map_err(ImageError::BmpError),
            FileType::Tiff => tiff::load_tiff(file_name).map_err(ImageError::TiffError),

            _ => panic!()
        }
    }


    pub fn save(&self, file_name: &str, file_type: FileType) -> Result<(), ImageError> {
        let ftype = if file_type == FileType::Auto { get_file_type_from_ext(file_name) } else { file_type };
        match ftype {
            FileType::Bmp => bmp::save_bmp(&self, file_name).map_err(ImageError::BmpError),
            FileType::Tiff => tiff::save_tiff(&self, file_name).map_err(ImageError::TiffError),

            _ => panic!()
        }
    }


    /// Returns width, height, pixel format, palette.
    pub fn get_metadata(file_name: &str, file_type: FileType) -> Result<(u32, u32, PixelFormat, Option<Palette>), ImageError> {
        let ftype = if file_type == FileType::Auto { get_file_type_from_ext(file_name) } else { file_type };
        match ftype {
            FileType::Bmp => bmp::get_bmp_metadata(file_name).map_err(ImageError::BmpError),
            FileType::Tiff => tiff::get_tiff_metadata(file_name).map_err(ImageError::TiffError),

            _ => panic!()
        }
    }


    pub fn get_width(&self) -> u32 {
        self.width
    }


    pub fn get_height(&self) -> u32 {
        self.height
    }


    pub fn get_pixel_format(&self) -> PixelFormat {
        self.pix_fmt
    }


    /// Creates a new image using the specified storage.
    ///
    /// `pixels` must have enough space. `palette` is used only if `pix_fmt` equals `Pal8`.
    ///
    pub fn new_from_pixels(width: u32, height: u32, pix_fmt: PixelFormat, pal: Option<Palette>, pixels: Vec<u8>) -> Image {
        assert!(pixels.len() >= (width * height) as usize * bytes_per_pixel(pix_fmt));

        Image{ width: width,
               height: height,
               pix_fmt: pix_fmt,
               palette: pal,
               pixels: pixels,
               bytes_per_line: width as usize * bytes_per_pixel(pix_fmt)
        }
    }


    /// Creates a new image.
    ///
    /// `palette` is used only if `pix_fmt` equals `Pal8`.
    ///
    pub fn new(width: u32, height: u32, pix_fmt: PixelFormat, palette: Option<Palette>, zero_fill: bool) -> Image {
        let pixels: Vec<u8>;
        let byte_count = (width * height) as usize * bytes_per_pixel(pix_fmt);
        if zero_fill {
            pixels = vec![0; byte_count];
        } else {
            pixels = utils::alloc_uninitialized(byte_count);
        }

        Image::new_from_pixels(width, height, pix_fmt, palette, pixels)
    }


    pub fn get_bytes_per_line(&self) -> usize {
        self.bytes_per_line
    }


    /// Returns pixels.
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn get_pixels<T: Any + Default>(&self) -> &[T] {
        verify_pix_type::<T>(self.pix_fmt);

        let ptr: *const u8 = self.pixels[..].as_ptr();
        unsafe {
            slice::from_raw_parts(ptr as *const T, (self.width * self.height) as usize * get_num_channels(self.pix_fmt))
        }
    }


    /// Returns mutable pixels.
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn get_pixels_mut<T: Any + Default>(&mut self) -> &mut [T] {
        verify_pix_type::<T>(self.pix_fmt);

        let ptr: *const u8 = self.pixels[..].as_ptr();
        unsafe {
            slice::from_raw_parts_mut(ptr as *mut T, (self.width * self.height) as usize * get_num_channels(self.pix_fmt))
        }
    }


    /// For a Mono8 image, returns pixels starting from `start` coordinates.
    pub fn get_mono8_pixels_from(&self, start: Point) -> &[u8] {
        assert!(self.pix_fmt == PixelFormat::Mono8);
        &self.pixels[(start.y as usize) * self.bytes_per_line + start.x as usize ..]
    }


    /// Returns all pixels as raw bytes (regardless of pixel format).
    pub fn get_raw_pixels(&self) -> &[u8] {
        &self.pixels[..]
    }


    /// Returns a line as raw bytes (regardless of pixel format).
    pub fn get_line_raw(&self, y: u32) -> &[u8] {
        &self.pixels[range!(y as usize * self.bytes_per_line, self.bytes_per_line)]
    }


    /// Returns a mutable line as raw bytes (regardless of pixel format).
    pub fn get_line_raw_mut(&mut self, y: u32) -> &mut [u8] {
        &mut self.pixels[range!(y as usize * self.bytes_per_line, self.bytes_per_line)]
    }


    /// Returns image line.
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn get_line<T: Any + Default>(&self, y: u32) -> &[T] {
        assert!(y < self.height);
        let vals_per_line = self.width as usize * get_num_channels(self.pix_fmt);

        &self.get_pixels::<T>()[range!(y as usize * vals_per_line, vals_per_line)]
    }


    /// Returns mutable image line.
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn get_line_mut<T: Any + Default>(&mut self, y: u32) -> &mut [T] {
        assert!(y < self.height);
        let vals_per_line = self.width as usize * get_num_channels(self.pix_fmt);

        &mut self.get_pixels_mut::<T>()[range!(y as usize * vals_per_line, vals_per_line)]
    }


    pub fn get_palette(&self) -> &Option<Palette> {
        &self.palette
    }


    pub fn get_palette_mut(&mut self) -> &mut Option<Palette> {
        &mut self.palette
    }


    pub fn get_img_rect(&self) -> Rect { Rect{ x: 0, y: 0, width: self.width as u32, height: self.height as u32 } }


    /// Calculates and returns image moments: M00, M10, M01. `T`: type of pixel values.
    ///
    /// `pixels` must not contain palette entries. To use the function with a palettized image, convert it to other format first.
    ///
    fn get_moments<T: Any + Default>(&self, img_fragment: Rect) -> (f64, f64, f64)
        where T: Copy, f64: From<T> {

        let mut m00: f64 = 0.0; // Image moment 00, i.e. sum of pixels' brightness
        let mut m10: f64 = 0.0; // Image moment 10
        let mut m01: f64 = 0.0; // Image moment 01

        let pixels = self.get_pixels::<T>();
        let num_channels = get_num_channels(self.pix_fmt);

        let mut offs = img_fragment.y as usize * self.width as usize;

        for y in range!(img_fragment.y, img_fragment.height as i32) {
            for x in img_fragment.x .. img_fragment.x + img_fragment.width as i32 {
                let mut current_brightness: f64 = 0.0;

                for i in 0..num_channels {
                    current_brightness += f64::from(pixels[offs + num_channels * (x as usize) + i]);
                }

                m00 += current_brightness;
                m10 += (x - img_fragment.x) as f64 * current_brightness;
                m01 += (y - img_fragment.y) as f64 * current_brightness;
            }

            offs += self.width as usize;
        }

        (m00, m10, m01)
    }



    /// Finds the centroid of the specified image fragment.
    ///
    /// Returned coords are relative to `img_fragment`. Image must not be `PixelFormat::Pal8`.
    ///
    pub fn get_centroid(&self, img_fragment: Rect) -> Point {
        let m00: f64;
        let m10: f64;
        let m01: f64;

        match self.pix_fmt {
            PixelFormat::Pal8     |
            PixelFormat::Mono8    |
            PixelFormat::RGB8     |
            PixelFormat::BGR8     |
            PixelFormat::BGRA8    |
            PixelFormat::CfaRGGB8 |
            PixelFormat::CfaGRBG8 |
            PixelFormat::CfaGBRG8 |
            PixelFormat::CfaBGGR8 => {
                let moments = self.get_moments::<u8>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            },

            PixelFormat::Mono16    |
            PixelFormat::RGB16     |
            PixelFormat::RGBA16    |
            PixelFormat::CfaRGGB16 |
            PixelFormat::CfaGRBG16 |
            PixelFormat::CfaGBRG16 |
            PixelFormat::CfaBGGR16 => {
                let moments = self.get_moments::<u16>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            },

            PixelFormat::Mono32f | PixelFormat::RGB32f => {
                let moments = self.get_moments::<f32>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            },

            PixelFormat::Mono64f | PixelFormat::RGB64f => {
                let moments = self.get_moments::<f64>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            }
        }

        if m00 == 0.0 {
            return Point{ x: img_fragment.width as i32 / 2, y: img_fragment.height as i32 / 2 };
        } else {
            return Point{ x: (m10/m00) as i32, y: (m01/m00) as i32 };
        }
    }


    /// Converts a fragment of the image to `dest_img`'s pixel format and writes it into `dest_img`.
    ///
    /// The fragment to convert starts at `src_pos` in `&self`, has `width`x`height` pixels and will
    /// be written to `dest_img` starting at `dest_pos`. Cropping is performed if necessary.
    /// If `&self` is in raw color format, the CFA pattern will be appropriately adjusted,
    /// depending on `src_pos` and `dest_pos`.
    ///
    pub fn convert_pix_fmt_of_subimage_into(&self,
                                            dest_img: &mut Image,
                                            src_pos: Point,
                                            dest_pos: Point,
                                            width: u32,
                                            height: u32,
                                            demosaic_method: Option<DemosaicMethod>) {

        // Converting to Pal8 or raw color (CFA) is not supported
        assert!(!(dest_img.pix_fmt == PixelFormat::Pal8 && self.pix_fmt != PixelFormat::Pal8));
        assert!(!dest_img.pix_fmt.is_cfa());

        // Source position cropped so that the source rectangle fits in `src_img`
        let mut actual_src_pos = Point{ x: max(0, src_pos.x),
                                        y: max(0, src_pos.y) };

        let mut actual_width = min(width as i32, self.width as i32 - actual_src_pos.x) as usize;
        let mut actual_height = min(height as i32, self.width as i32 - actual_src_pos.y) as usize;

        // Destination position based on `src_pos` and further cropped so that the dest. rectangle fits in `dest_img`
        let mut actual_dest_pos = Point{ x: dest_pos.x + (src_pos.x - actual_src_pos.x),
                                         y: dest_pos.y + (src_pos.y - actual_src_pos.y) };

        if actual_dest_pos.x >= dest_img.width as i32 ||
           actual_dest_pos.y >= dest_img.height as i32 ||
           actual_src_pos.x >= self.width as i32 ||
           actual_src_pos.y >= self.height as i32 {

            return;
        }

        actual_dest_pos.x = max(0, actual_dest_pos.x);
        actual_dest_pos.y = max(0, actual_dest_pos.y);

        actual_width = min(actual_width as i32, dest_img.width as i32 - actual_dest_pos.x) as usize;
        actual_height = min(actual_height as i32, dest_img.height as i32 - actual_dest_pos.y) as usize;

        // Reflect in the source rectangle any cropping imposed by `dest_img`
        actual_src_pos.x += actual_dest_pos.x - dest_pos.x;
        actual_src_pos.y += actual_dest_pos.y - dest_pos.y;

        if self.pix_fmt == dest_img.pix_fmt {
            // No conversion required, just copy the data

            let bpp = bytes_per_pixel(self.pix_fmt);
            let copy_line_len = actual_width * bpp;

            let mut src_ofs = actual_src_pos.y as usize * self.width as usize * bpp;
            let mut dest_ofs = actual_dest_pos.y as usize * dest_img.width as usize * bpp;

            for _ in 0..actual_height {
                let copy_to = dest_ofs + actual_dest_pos.x as usize * bpp;
                let copy_from = src_ofs + actual_src_pos.x as usize * bpp;

                dest_img.pixels[range!(copy_to, copy_line_len)]
                    .copy_from_slice(&self.pixels[range!(copy_from, copy_line_len)]);

                src_ofs += self.width as usize * bpp;
                dest_ofs += dest_img.width as usize * bpp;
            }

            return;
        }

        let src_step = bytes_per_pixel(self.pix_fmt);
        let dest_step = bytes_per_pixel(dest_img.pix_fmt);

        for y in 0..actual_height {
            let mut src_ofs = (y + actual_src_pos.y as usize) * self.bytes_per_line +
                actual_src_pos.x as usize * bytes_per_pixel(self.pix_fmt);
            let mut dest_ofs = (y + actual_dest_pos.y as usize) * dest_img.bytes_per_line +
                actual_dest_pos.x as usize * bytes_per_pixel(dest_img.pix_fmt);

            /// Returns a slice of `dest_img`'s pixel values of type `T`, beginning at byte offset `dest_ofs`.
            macro_rules! dest { ($len:expr, $T:ty) => { unsafe { slice::from_raw_parts_mut(dest_img.pixels[dest_ofs..].as_mut_ptr() as *mut $T, $len) } }};

            /// Executes the code in block `b` in a loop encompassing a whole line of destination area in `dest_img`.
            macro_rules! convert_whole_line {
                ($b:block) => {
                    for _ in 0..actual_width {
                        $b

                        src_ofs += src_step;
                        dest_ofs += dest_step;
                    }
                }
            };

            match self.pix_fmt {
                PixelFormat::Mono8 => {
                    match dest_img.pix_fmt {
                        PixelFormat::Mono16 =>
                            convert_whole_line!({ dest!(1, u16)[0] = (self.pixels[src_ofs] as u16) << 8 }),

                        PixelFormat::Mono32f =>
                            convert_whole_line!({ dest!(1, f32)[0] = self.pixels[src_ofs] as f32 / 0xFF as f32; }),

                        PixelFormat::Mono64f =>
                            convert_whole_line!({ dest!(1, f64)[0] = self.pixels[src_ofs] as f64 / f64::from(0xFF); }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!({ for i in dest!(3, f32) { *i = self.pixels[src_ofs] as f32 / 0xFF as f32; } }),

                        PixelFormat::RGB64f =>
                            convert_whole_line!({ for i in dest!(3, f64) { *i = self.pixels[src_ofs] as f64 * 1.0 / 0xFF as f64; } }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = self.pixels[src_ofs]; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = self.pixels[src_ofs]; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!({ for i in dest!(3, u16) { *i = (self.pixels[src_ofs] as u16) << 8; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::Mono16 => {
                    /// Returns the current source pixel value as `u16`.
                    macro_rules! src { () => { unsafe { *(self.pixels[src_ofs..].as_ptr() as *const u16) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!() >> 8) as u8; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = src!() as f32 / 0xFFFF as f32; }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!({ for i in dest!(3, f32) { *i = src!() as f32 / 0xFFFF as f32; } }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!() >> 8) as u8; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = (src!() >> 8) as u8; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!({ for i in dest!(3, u16) { *i = src!(); } }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = src!() as f64 / 0xFFFF as f64; }),

                        PixelFormat::RGB64f => convert_whole_line!({ for i in dest!(3, f64) { *i = src!() as f64 / 0xFFFF as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::Mono32f => {
                    /// Returns the current source pixel value as `f32`.
                    macro_rules! src { () => { unsafe { *(self.pixels[src_ofs..].as_ptr() as *const f32) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!() * 0xFF as f32) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!() * 0xFFFF as f32) as u16; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u8);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!() * 0xFF as f32) as u8; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = (src!() * 0xFF as f32) as u8; } }),

                        PixelFormat::RGB16 => convert_whole_line!({ for i in dest!(3, u16) { *i = (src!() * 0xFFFF as f32) as u16; } }),

                        PixelFormat::RGB32f => convert_whole_line!({ for i in dest!(3, f32) { *i = src!(); } }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = src!() as f64; }),

                        PixelFormat::RGB64f => convert_whole_line!({ for i in dest!(3, f64) { *i = src!() as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::Mono64f => {
                    /// Returns the current source pixel value as `f64`.
                    macro_rules! src { () => { unsafe { *(self.pixels[src_ofs..].as_ptr() as *const f64) } }};

                    match dest_img.pix_fmt {

                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!() * 0xFF as f64) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!() * 0xFFFF as f64) as u16; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u8);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!() * 0xFF as f64) as u8; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = (src!() * 0xFF as f64) as u8; } }),

                        PixelFormat::RGB16 => convert_whole_line!({ for i in dest!(3, u16) { *i = (src!() * 0xFFFF as f64) as u16; } }),

                        PixelFormat::RGB32f => convert_whole_line!({ for i in dest!(3, f32) { *i = src!() as f32; } }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = src!() as f32; }),

                        PixelFormat::RGB64f => convert_whole_line!({ for i in dest!(3, f64) { *i = src!(); } }),

                        _ => panic!()
                    }
                },

                // When converting from a color format to mono, use sum (scaled) of all channels as the pixel brightness.

                PixelFormat::Pal8 => {
                    let pal: &Palette = self.palette.iter().next().unwrap();

                    /// Returns the current source pixel converted to RGB (8-bit). Parameter `x` is from 0..3.
                    macro_rules! src { ($x:expr) => { pal.pal[(3 * self.pixels[src_ofs]) as usize + $x] } };

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = ((src!(0) as u32 + src!(1) as u32 + src!(2) as u32) / 3) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (((src!(0) as u32 + src!(1) as u32 + src!(2) as u32) / 3) << 8) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!(0) as u32 + src!(1) as u32 + src!(2) as u32) as f32 / (3.0 * 0xFF as f32); }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = (src!(0) as u32 + src!(1) as u32 + src!(2) as u32) as f64 / (3.0 * 0xFF as f64); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = src!(2-i); }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = src!(2-i); }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u8)[i] = src!(i); } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!(i) as u16) << 8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = (src!(i) as f32) / 0xFF as f32; } }),

                        PixelFormat::RGB64f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f64)[i] = (src!(i) as f64) / 0xFF as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::RGB8 => {
                    /// Returns the current source pixel as `u8` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const u8, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!()[0] + src!()[1] + src!()[2]) as u16 / 3 * 0xFF; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) as f32 / (3.0 * 0xFF as f32); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = src!()[2-i]; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = src!()[2-i]; }
                            }),

                        PixelFormat::RGBA16 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u16);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!()[i] as u16) << 8; }
                            }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] as u16) << 8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32 / 0xFF as f32; } }),


                        _ => panic!()
                    }
                },

                PixelFormat::BGR8 => {
                    /// Returns the current source pixel as `u8` BGR values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const u8, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!()[0] + src!()[1] + src!()[2]) as u16 / 3 * 0xFF; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) as f32 / (3.0 * 0xFF as f32); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = src!()[2-i]; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!({
                                let rgb = dest!(3, u8);
                                for i in 0..3 { rgb[i] = src!()[2-i]; }
                            }),

                        PixelFormat::RGBA16 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u16);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!()[i] as u16) << 8; }
                            }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] as u16) << 8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32 / 0xFF as f32; } }),


                        _ => panic!()
                    }
                },

                PixelFormat::RGB16 => {
                    /// Returns the current source pixel as `u16` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const u16, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (((src!()[0] as u32 + src!()[1] as u32 + src!()[2] as u32) / 3) >> 8) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = ((src!()[0] as u32 + src!()[1] as u32 + src!()[2] as u32) / 3) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) as f32 / (3.0 * 0xFFFF as f32); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!()[2-i] >> 8) as u8; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = (src!()[2-i] >> 8) as u8; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u8)[i] = (src!()[i] >> 8) as u8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32 / 0xFFFF as f32; } }),


                        _ => panic!()
                    }
                },

                PixelFormat::RGB32f => {
                    /// Returns the current source pixel as `f32` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const f32, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFF as f32/3.0) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFFFF as f32/3.0) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3.0; }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = ((src!()[0] + src!()[1] + src!()[2]) / 3.0) as f64; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!()[2-i] * 0xFF as f32) as u8; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = (src!()[2-i] * 0xFF as f32) as u8; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!({ for i in 0..3 { dest!(3, u8)[i] = (src!()[i] * 0xFF as f32) as u8; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] * 0xFFFF as f32) as u16; } }),

                        PixelFormat::RGB64f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f64)[i] = src!()[i] as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::RGB64f => {
                    /// Returns the current source pixel as `f64` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const f64, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFF as f64/3.0) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFFFF as f64/3.0) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = ((src!()[0] + src!()[1] + src!()[2]) / 3.0) as f32; }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3.0; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!()[2-i] * 0xFF as f64) as u8; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = (src!()[2-i] * 0xFF as f64) as u8; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!({ for i in 0..3 { dest!(3, u8)[i] = (src!()[i] * 0xFF as f64) as u8; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] * 0xFFFF as f64) as u16; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32; } }),

                        _ => panic!()
                    }
                },

                _ => panic!()
            }
        }
    }


    /// Returns a fragment of the image converted to the specified pixel format.
    ///
    /// The fragment to convert starts at `src_pos` in `&self` and has `width`x`height` pixels.
    /// Cropping is performed if necessary. If `&self` is in raw color format, the CFA pattern
    /// will be appropriately adjusted, depending on `src_pos`.
    ///
    pub fn convert_pix_fmt_of_subimage(&self,
                                       dest_pix_fmt: PixelFormat,
                                       src_pos: Point,
                                       width: u32,
                                       height: u32,
                                       demosaic_method: Option<DemosaicMethod>) -> Image {
        let mut new_pal: Option<Palette> = None;
        if self.pix_fmt == PixelFormat::Pal8 {
            new_pal = Some(self.palette.iter().next().unwrap().clone());
        }

        let mut dest_img = Image::new(width, height, dest_pix_fmt, new_pal, false);

        self.convert_pix_fmt_of_subimage_into(&mut dest_img, src_pos, Point{ x: 0, y: 0 }, width, height, demosaic_method);

        dest_img
    }


    /// Returns the image converted to the specified pixel format.
    pub fn convert_pix_fmt(&self,
                           dest_pix_fmt: PixelFormat,
                           demosaic_method: Option<DemosaicMethod>) -> Image {
        self.convert_pix_fmt_of_subimage(dest_pix_fmt, Point{ x: 0, y: 0 }, self.width, self.height, demosaic_method)
    }


    pub fn get_copy(&self) -> Image {
        self.get_fragment_copy(Point{ x: 0, y: 0 }, self.width, self.height, false)
    }


    /// Returns a copy of image's fragment. The fragment boundaries may extend outside of the image.
    ///
    /// The fragment to copy is `width`x`height` pixels and starts at `src_pos`.
    /// If `clear_to_zero` is true, fragment's areas outside of the image will be cleared to zero.
    ///
    pub fn get_fragment_copy(&self,
                             src_pos: Point,
                             width: u32,
                             height: u32,
                             clear_to_zero: bool) -> Image {
        let mut dest_img = Image::new(width, height, self.pix_fmt, self.palette, clear_to_zero);

        self.resize_and_translate_into(&mut dest_img, src_pos, width, height, Point{ x: 0, y: 0 }, clear_to_zero);

        dest_img
    }

    /// Copies (with cropping or padding) a fragment of image to another. There is no scaling.
    ///
    /// Pixel formats of source and destination must be the same; `src_img` must not equal `dest_img`.
    /// The fragment to copy is `width`x`height` pixels and starts at `src_pos` in `&self`
    /// and at `dest_pos` at `dest_img`. If `clear_to_zero` is true, `dest_img`'s areas not copied on
    /// will be cleared to zero.
    /// NOTE: care must be taken if pixel format is raw color (CFA). The caller may need to adjust
    /// the CFA pattern if source and destination X, Y offets are not simultaneously odd/even.
    ///
    pub fn resize_and_translate_into(
        &self,
        dest_img: &mut Image,
        src_pos: Point,
        width: u32,
        height: u32,
        dest_pos: Point,
        clear_to_zero: bool) {

        assert!(self.pix_fmt == dest_img.pix_fmt);

        let src_w = self.width;
        let src_h = self.height;
        let dest_w = dest_img.width;
        let dest_h = dest_img.height;

        let b_per_pix = bytes_per_pixel(self.pix_fmt);

        // Start and end (inclusive) coordinates to fill in the output image
        let mut dest_x_start = dest_pos.x;
        let mut dest_x_end = dest_pos.x + width as i32 - 1;

        let mut dest_y_start = dest_pos.y;
        let mut dest_y_end = dest_pos.y + height as i32 - 1;

        // Actual source coordinates to use
        let mut src_x_start = src_pos.x;
        let mut src_y_start = src_pos.y;

        // Perform any necessary cropping

        // Source image, left and top
        if src_pos.x < 0 {
            src_x_start -= src_pos.x;
            dest_x_start -= src_pos.x;
        }
        if src_pos.y < 0 {
            src_y_start -= src_pos.y;
            dest_y_start -= src_pos.y;
        }

        // Source image, right and bottom
        if src_pos.x + width as i32 > src_w as i32 {
            dest_x_end -= src_pos.x + width as i32 - src_w as i32;
        }

        if src_pos.y + height as i32 > src_h as i32 {
            dest_y_end -= src_pos.y + height as i32 - src_h as i32;
        }

        // Destination image, left and top
        if dest_x_start < 0 {
            src_x_start -= dest_x_start;
            dest_x_start = 0;
        }
        if dest_y_start < 0 {
            src_y_start -= dest_y_start;
            dest_y_start = 0;
        }

        // Destination image, right and bottom
        if dest_x_end >= dest_w as i32 {
            dest_x_end = dest_w as i32 - 1;
        }
        if dest_y_end >= dest_h as i32 {
            dest_y_end = dest_h as i32 - 1;
        }

        if dest_y_end < dest_y_start || dest_x_end < dest_x_start {
            // Nothing to copy

            if clear_to_zero {
                // Also works for floating-point pixels; all zero bits = 0.0
                unsafe { ptr::write_bytes(dest_img.pixels[..].as_mut_ptr(), 0, (dest_img.width * dest_img.height) as usize); }
            }
            return;
        }

        if clear_to_zero {
            // Unchanged lines at the top
            for y in 0..dest_y_start as u32 {
                unsafe { ptr::write_bytes(dest_img.get_line_raw_mut(y).as_mut_ptr(), 0, dest_img.bytes_per_line); }
            }

            // Unchanged lines at the bottom
            for y in dest_y_end as u32 + 1 .. dest_img.height {
                unsafe { ptr::write_bytes(dest_img.get_line_raw_mut(y).as_mut_ptr(), 0, dest_img.bytes_per_line); }
            }

            for y in dest_y_start as u32 .. dest_y_end as u32 + 1 {
                // Columns to the left of the target area
                unsafe { ptr::write_bytes(dest_img.get_line_raw_mut(y).as_mut_ptr(), 0, dest_x_start as usize * b_per_pix); }

                // Columns to the right of the target area
                let dest_ptr: *mut u8 = dest_img.pixels[y as usize * dest_img.bytes_per_line + (dest_x_end as usize + 1) * b_per_pix ..].as_mut_ptr();
                unsafe { ptr::write_bytes(dest_ptr, 0, (dest_img.width as usize - 1 - dest_x_end as usize) * b_per_pix); }
            }
        }

        // Copy the pixels line by line
        for y in dest_y_start .. dest_y_end  + 1 {
            let line_copy_bytes = (dest_x_end - dest_x_start + 1) as usize * b_per_pix;

            let src_line_ofs = src_x_start as usize * b_per_pix;
            let src_line: &[u8] = &self.get_line_raw((y - dest_y_start + src_y_start) as u32)[range!(src_line_ofs, line_copy_bytes)];

            let dest_line_ofs = dest_x_start as usize * b_per_pix;
            let dest_line: &mut [u8] = &mut dest_img.get_line_raw_mut(y as u32)[range!(dest_line_ofs, line_copy_bytes)];

            dest_line.copy_from_slice(src_line);
        }
    }

}


impl Clone for Image {
    fn clone(&self) -> Image {
        let new_pixels = self.pixels.clone();
        Image::new_from_pixels(self.width, self.height, self.pix_fmt, self.palette, new_pixels)
    }
}