// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   BMP support.
//

use image::{Image, Palette, PixelFormat, bytes_per_pixel};
use std::fs::{File, OpenOptions};
use std::io;
use std::io::prelude::*;
use std::io::{Seek, SeekFrom};
use std::mem::{size_of};
use utils;


impl Default for BmpPalette {
    fn default() -> BmpPalette { BmpPalette{ pal: [0; BMP_PALETTE_SIZE] }}
}


#[repr(C, packed)]
#[derive(Default)]
struct BitmapFileHeader {
    pub ftype:     [u8; 2],
    pub size:      u32,
    pub reserved1: u16,
    pub reserved2: u16,
    pub off_bits:  u32
}


pub const BMP_PALETTE_SIZE: usize = 256*4;
pub const BI_RGB: u32 = 0;
pub const BI_BITFIELDS: u32 = 3;


#[repr(C, packed)]
#[derive(Default)]
pub struct BitmapInfoHeader {
   pub size:             u32,
   pub width:            i32,
   pub height:           i32,
   pub planes:           u16,
   pub bit_count:        u16,
   pub compression:      u32,
   pub size_image:       u32,
   pub x_pels_per_meter: i32,
   pub y_pels_per_meter: i32,
   pub clr_used:         u32,
   pub clr_important:    u32
}


#[repr(C, packed)]
pub struct BmpPalette {
    pub pal: [u8; BMP_PALETTE_SIZE]
}


pub fn convert_bmp_palette(num_used_pal_entries: u32, bmp_pal: &BmpPalette) -> Palette {
    let mut pal = Palette::default(); 
    for i in 0..num_used_pal_entries as usize {
        pal.pal[3*i + 0] = bmp_pal.pal[i*4 + 2];
        pal.pal[3*i + 1] = bmp_pal.pal[i*4 + 1];
        pal.pal[3*i + 2] = bmp_pal.pal[i*4 + 0];
    }
    
    pal
}


pub fn is_mono8_palette(palette: &Palette) -> bool {
    for i in 0..Palette::NUM_ENTRIES {
        if palette.pal[3*i + 0] as usize != i ||
           palette.pal[3*i + 1] as usize != i ||
           palette.pal[3*i + 2] as usize != i {
               
            return false;
        }
    }
    
    true
}


#[derive(Debug)]
pub enum BmpError {
    Io(io::Error),
    MalformedFile,
    UnsupportedFormat
}


impl From<io::Error> for BmpError {
    fn from(err: io::Error) -> BmpError { BmpError::Io(err) }
}


/// Returns metadata (width, height, ...) without reading the pixel data.
pub fn get_bmp_metadata(file_name: &str) -> Result<(u32, u32, PixelFormat, Option<Palette>), BmpError> {
    let mut file = try!(OpenOptions::new().read(true)
                                          .write(false)
                                          .open(file_name));

    let (img_width, img_height, _, pix_fmt, palette) = try!(get_bmp_metadata_priv(&mut file));
    
    Ok((img_width, img_height, pix_fmt, palette))
}


/// Returns width, height, file bits per pixel, pixel format, palette.
///
/// After return, `file`'s cursor will be positioned at the beginning of pixel data.
///
fn get_bmp_metadata_priv(file: &mut File) -> Result<(u32, u32, usize, PixelFormat, Option<Palette>), BmpError> {
    let file_hdr: BitmapFileHeader = try!(utils::read_struct(file));
    let info_hdr: BitmapInfoHeader = try!(utils::read_struct(file));

    // Fields in a BMP are always little-endian, so remember to swap them
    
    let bits_per_pixel = u16::from_le(info_hdr.bit_count);
    let img_width = i32::from_le(info_hdr.width) as u32;
    let img_height = i32::from_le(info_hdr.height) as u32;

    if img_width == 0 || img_height == 0 ||
       file_hdr.ftype[0] != 'B' as u8 || file_hdr.ftype[1] != 'M' as u8 ||
       u16::from_le(info_hdr.planes) != 1 ||
       bits_per_pixel != 8 && bits_per_pixel != 24 && bits_per_pixel != 32 ||
       u32::from_le(info_hdr.compression) != BI_RGB && u32::from_le(info_hdr.compression) != BI_BITFIELDS {
           
        return Err(BmpError::UnsupportedFormat);
    }

    let mut pix_fmt;

    if bits_per_pixel == 8 {
        pix_fmt = PixelFormat::Pal8;
    } else if bits_per_pixel == 24 || bits_per_pixel == 32 {
        pix_fmt = PixelFormat::RGB8;
    } else {
        panic!(); // Cannot happen (due to the previous checks)
    }

    let mut pal: Option<Palette> = None;

    if pix_fmt == PixelFormat::Pal8 {
        let mut num_used_pal_entries = u32::from_le(info_hdr.clr_used);
        
        if num_used_pal_entries == 0 {
            num_used_pal_entries = 256;
        }

        // Seek to the beginning of palette
        try!(file.seek(SeekFrom::Start((size_of::<BitmapFileHeader>() + u32::from_le(info_hdr.size) as usize) as u64)));

        let bmp_palette: BmpPalette = try!(utils::read_struct(file));

        // Convert to an RGB-order palette
        let palette = convert_bmp_palette(num_used_pal_entries, &bmp_palette);

        if is_mono8_palette(&palette) {
            pix_fmt = PixelFormat::Mono8;
        }
        
        pal = Some(palette);
    }
    
    try!(file.seek(SeekFrom::Start(u32::from_le(file_hdr.off_bits) as u64)));    
    
    Ok((img_width, img_height, bits_per_pixel as usize, pix_fmt, pal))
}


pub fn load_bmp(file_name: &str) -> Result<Image, BmpError> {

    let mut file = try!(OpenOptions::new().read(true)
                                          .write(false)
                                          .open(file_name));

    let (img_width, img_height, bits_per_pix, pix_fmt, palette) = try!(get_bmp_metadata_priv(&mut file));

    let src_bytes_per_pixel = bits_per_pix / 8;

    let dest_bytes_per_line = img_width as usize * bytes_per_pixel(pix_fmt);
    let dest_byte_count = img_height as usize * dest_bytes_per_line;
    
    let mut pixels = utils::alloc_uninitialized(dest_byte_count);

    if [PixelFormat::Pal8, PixelFormat::Mono8].contains(&pix_fmt) {
        let bmp_stride: usize = upmult!(img_width as usize, 4); // Line length in bytes in the BMP file's pixel data
        let skip = bmp_stride - img_width as usize;   // Number of padding bytes at the end of a line

        let mut y = img_height - 1;
        loop {
            try!(file.read_exact(&mut pixels[range!(y as usize * dest_bytes_per_line, dest_bytes_per_line)]));

            if skip > 0 {
                try!(file.seek(SeekFrom::Current(skip as i64)));
            }

            if y == 0 {
                break;
            } else {
                y -= 1; // Lines in BMP are stored bottom-to-top
            }
        }
    } else if pix_fmt == PixelFormat::RGB8 {
        let bmp_stride = upmult!(img_width as usize * src_bytes_per_pixel, 4); // Line length in bytes in the BMP file's pixel data
        let skip = bmp_stride - img_width as usize * src_bytes_per_pixel;      // Number of padding bytes at the end of a line

        let mut src_line = utils::alloc_uninitialized(img_width as usize * src_bytes_per_pixel);

        let mut y = img_height - 1;
        loop {
            let dest_line = &mut pixels[range!(y as usize * dest_bytes_per_line, dest_bytes_per_line)];

            try!(file.read_exact(&mut src_line));

            if src_bytes_per_pixel == 3 {
                // Rearrange the channels to RGB order
                for x in 0..img_width as usize {
                    dest_line[x*3 + 0] = src_line[x*3 + 2];
                    dest_line[x*3 + 1] = src_line[x*3 + 1];
                    dest_line[x*3 + 2] = src_line[x*3 + 0];
                }
            }
            else if src_bytes_per_pixel == 4 {
                // Remove the unused 4th byte from each pixel and rearrange the channels to RGB order
                for x in 0..img_width as usize {
                    dest_line[x*3 + 0] = src_line[x*4 + 3];
                    dest_line[x*3 + 1] = src_line[x*4 + 2];
                    dest_line[x*3 + 2] = src_line[x*4 + 1];
                }
            }

            if skip > 0 {
                try!(file.seek(SeekFrom::Current(skip as i64)));
            }

            if y == 0 {
                break;
            } else {
                y -= 1; // Lines in BMP are stored bottom-to-top
            }
        }
    }

    Ok(Image::new_from_pixels(img_width, img_height, pix_fmt, palette, pixels))
}


pub fn save_bmp(img: &Image, file_name: &str) -> Result<(), BmpError> {
    let pix_fmt = img.get_pixel_format();
    
    if ![PixelFormat::Pal8, PixelFormat::RGB8, PixelFormat::Mono8].contains(&pix_fmt) {
        return Err(BmpError::UnsupportedFormat);
    }
    
    let width = img.get_width();
    let height = img.get_height();
    let bytes_per_pix = bytes_per_pixel(pix_fmt);
    let bmp_line_width = upmult!(width as usize * bytes_per_pix, 4);

    let mut pix_data_offs = size_of::<BitmapFileHeader>() +
                            size_of::<BitmapInfoHeader>();
    if [PixelFormat::Pal8, PixelFormat::Mono8].contains(&pix_fmt) {
        pix_data_offs += size_of::<BmpPalette>();
    }

    // Fields in a BMP are always little-endian

    let bmfh = BitmapFileHeader{
        ftype:     ['B' as u8, 'M' as u8],
        size:      u32::to_le((pix_data_offs + height as usize * bmp_line_width) as u32),
        reserved1: 0,
        reserved2: 0,
        off_bits:  u32::to_le(pix_data_offs as u32)
    };
    
    let bmih = BitmapInfoHeader{
        size:             u32::to_le(size_of::<BitmapInfoHeader>() as u32),
        width:            i32::to_le(width as i32),
        height:           i32::to_le(height as i32),
        planes:           u16::to_le(1),
        bit_count:        u16::to_le((bytes_per_pix * 8) as u16),
        compression:      u32::to_le(BI_RGB),
        size_image:       u32::to_le(0),
        x_pels_per_meter: i32::to_le(1000),
        y_pels_per_meter: i32::to_le(1000),
        clr_used:         u32::to_le(0),
        clr_important:    u32::to_le(0)
    };

    let mut file = try!(OpenOptions::new().read(false)
                                          .write(true)
                                          .create(true)
                                          .open(file_name));

    try!(utils::write_struct(&bmfh, &mut file));
    try!(utils::write_struct(&bmih, &mut file));
    
    if [PixelFormat::Pal8, PixelFormat::Mono8].contains(&pix_fmt) {
        let mut bmp_palette = BmpPalette::default();

        if pix_fmt == PixelFormat::Pal8 {
            let img_pal: &Palette = img.get_palette().iter().next().unwrap();

            for i in 0..256 {
                bmp_palette.pal[4*i + 0] = img_pal.pal[3*i + 2];
                bmp_palette.pal[4*i + 1] = img_pal.pal[3*i + 1];
                bmp_palette.pal[4*i + 2] = img_pal.pal[3*i + 0];
                bmp_palette.pal[4*i + 3] = 0;
            }
        } else {
            for i in 0..256 {
                bmp_palette.pal[4*i + 0] = i as u8;
                bmp_palette.pal[4*i + 1] = i as u8;
                bmp_palette.pal[4*i + 2] = i as u8;
                bmp_palette.pal[4*i + 3] = 0 as u8;
            }
        }

        try!(utils::write_struct(&bmp_palette, &mut file));
    }

    let pix_bytes_per_line = width as usize * bytes_per_pix;
    let line_padding = vec![0; bmp_line_width - pix_bytes_per_line];
    let mut bmp_line = utils::alloc_uninitialized(pix_bytes_per_line);
    
    for i in 0..height {
        let src_line = img.get_line_raw(height - i - 1);

        if [PixelFormat::Pal8, PixelFormat::Mono8].contains(&pix_fmt) {
            try!(file.write_all(&src_line));
        } else {
            // Rearrange the channels to BGR order
            for x in 0..width as usize {
                bmp_line[x*3 + 0] = src_line[x*3 + 2];
                bmp_line[x*3 + 1] = src_line[x*3 + 1];
                bmp_line[x*3 + 2] = src_line[x*3 + 0];
            }
            try!(file.write_all(&bmp_line));
        }
        if !line_padding.is_empty() {
            try!(file.write_all(&line_padding));
        }
    }

    Ok(())
}