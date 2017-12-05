//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   TIFF support.
//

use image::{Image, Palette, PixelFormat, bytes_per_channel, bytes_per_pixel, get_num_channels};
use std::fs::{File, OpenOptions};
use std::io;
use std::io::prelude::*;
use std::io::{Seek, SeekFrom};
use std::mem::{size_of, size_of_val};
use utils;


#[derive(Debug)]
pub enum TiffError {
    Io(io::Error),
    UnknownVersion,
    UnsupportedPixelFormat,
    ChannelBitDepthsDiffer,
    CompressionNotSupported,
    UnsupportedPlanarConfig
}


impl From<io::Error> for TiffError {
    fn from(err: io::Error) -> TiffError { TiffError::Io(err) }
}


#[repr(C, packed)]
struct TiffField {
    tag:   u16,
    ftype: u16,
    count: u32,
    value: u32
}

#[repr(C, packed)]
struct TiffHeader {
    id:         u16,
    version:    u16,
    dir_offset: u32
}


const TAG_TYPE_BYTE    : u16 = 1;
const TAG_TYPE_ASCII   : u16 = 2;
const TAG_TYPE_WORD    : u16 = 3; // 16 bits
const TAG_TYPE_DWORD   : u16 = 4; // 32 bits
const TAG_TYPE_RATIONAL: u16 = 5;

const TIFF_VERSION: u16 = 42;

const TAG_IMAGE_WIDTH                 : u16 = 0x100;
const TAG_IMAGE_HEIGHT                : u16 = 0x101;
const TAG_BITS_PER_SAMPLE             : u16 = 0x102;
const TAG_COMPRESSION                 : u16 = 0x103;
const TAG_PHOTOMETRIC_INTERPRETATION  : u16 = 0x106;
const TAG_STRIP_OFFSETS               : u16 = 0x111;
const TAG_SAMPLES_PER_PIXEL           : u16 = 0x115;
const TAG_ROWS_PER_STRIP              : u16 = 0x116;
const TAG_STRIP_BYTE_COUNTS           : u16 = 0x117;
const TAG_PLANAR_CONFIGURATION        : u16 = 0x11C;

const NO_COMPRESSION: u32 = 1;
const PLANAR_CONFIGURATION_CHUNKY: u32 = 1;
const INTEL_BYTE_ORDER: u16 = (('I' as u16) << 8) + 'I' as u16;
const MOTOROLA_BYTE_ORDER: u16 = (('M' as u16) << 8) + 'M' as u16;

const PHMET_WHITE_IS_ZERO: u32 = 0;
const PHMET_BLACK_IS_ZERO: u32 = 1;
const PHMET_RGB: u32 = 2;


/// Reverses 8-bit grayscale values.
fn negate_grayscale_8(img: &mut Image) {
    for p in img.get_pixels_mut::<u8>() {
        *p = 0xFF - *p;
    }
}


/// Reverses 16-bit grayscale values.
fn negate_grayscale_16(img: &mut Image) {
    for p in img.get_pixels_mut::<u16>() {
        *p = 0xFFFF - *p;
    }
}


/// If `do_swap` is true, returns `x` with bytes swapped; otherwise, returns `x`.
macro_rules! cnd_swap { ($x:expr, $do_swap:expr) => { if $do_swap { $x.swap_bytes() } else { $x } }}

/// If `do_swap` is true, returns `x` with its two lower bytes swapped; otherwise, returns `x`.
fn cnd_swap_16_in_32(x: u32, do_swap: bool) -> u32 {
    if do_swap {
        ((x & 0xFF) << 8) | (x >> 8)
    } else {
        x
    }    
}


/// Returns bits per sample.
///
/// # Parameters
///
/// * `tiff_field` - The "bits per sample" TIFF field
/// * `endianess_diff` - `true` if the file and machine endianess differs
///
fn parse_tag_bits_per_sample(file: &mut File,
                             tiff_field: &TiffField,
                             endianess_diff: bool) -> Result<usize, TiffError> {
                                 
    assert!(tiff_field.tag == TAG_BITS_PER_SAMPLE);
    assert!(tiff_field.ftype == TAG_TYPE_WORD);

    let bits_per_sample;
    
    if tiff_field.count == 1 {
        return Ok(cnd_swap!(tiff_field.value, endianess_diff) as usize);
    } else {
        // Some files may have as many "bits per sample" values specified
        // as there are channels. Make sure they are all the same.
        
        try!(file.seek(SeekFrom::Start(tiff_field.value as u64)));

        let field_buf = try!(utils::read_vec::<u16>(file, tiff_field.count as usize));

        let first = field_buf[0];
        for val in &field_buf {
            if *val != first { return Err(TiffError::ChannelBitDepthsDiffer); }
        }

        bits_per_sample = cnd_swap!(first, endianess_diff);
    }

    if bits_per_sample != 8 && bits_per_sample != 16 {
        Err(TiffError::UnsupportedPixelFormat)
    } else {
        Ok(bits_per_sample as usize)
    }
}


/// Sets correct byte order in `tiff_field`'s data.
fn preprocess_tiff_field(tiff_field: &mut TiffField, endianess_diff: bool) {
    tiff_field.tag = cnd_swap!(tiff_field.tag, endianess_diff);
    tiff_field.ftype = cnd_swap!(tiff_field.ftype, endianess_diff);
    tiff_field.count = cnd_swap!(tiff_field.count, endianess_diff);
    if tiff_field.count > 1 || tiff_field.ftype == TAG_TYPE_DWORD {
        tiff_field.value = cnd_swap!(tiff_field.value, endianess_diff);
    } else if tiff_field.count == 1 && tiff_field.ftype == TAG_TYPE_WORD {
        // This is a special case where a 16-bit value is stored in
        // a 32-bit field, always in the lower-address bytes. So if
        // the machine is big-endian, the value always has to be
        // shifted right by 16 bits first, regardless of the file's
        // endianess, and only then swapped, if the machine and file
        // endianesses differ.
        if utils::is_machine_big_endian() {
            tiff_field.value >>= 16;
        }

        tiff_field.value = cnd_swap_16_in_32(tiff_field.value, endianess_diff);
    }
}


fn determine_pixel_format(samples_per_pixel: usize, bits_per_sample: usize) -> PixelFormat { 
    match samples_per_pixel {
        1 => match bits_per_sample {
                 8 => PixelFormat::Mono8,
                 16 => PixelFormat::Mono16,
                 _ => panic!()
             },
                      
        3 => match bits_per_sample {
                 8 => PixelFormat::RGB8,
                 16 => PixelFormat::RGB16,
                 _ => panic!()
             },
        
        _ => panic!()
    }
}


fn validate_tiff_format(samples_per_pixel: usize, photometric_interpretation: u32) -> Result<(), TiffError> {
    if samples_per_pixel == 1 && photometric_interpretation != PHMET_BLACK_IS_ZERO && photometric_interpretation != PHMET_WHITE_IS_ZERO ||
       samples_per_pixel == 3 && photometric_interpretation != PHMET_RGB ||
       samples_per_pixel != 1 && samples_per_pixel != 3 {
           
        Err(TiffError::UnsupportedPixelFormat)
    } else {
        Ok(())
    }
}


pub fn load_tiff(file_name: &str) -> Result<Image, TiffError> {
    let mut file = try!(OpenOptions::new().read(true)
                                          .write(false)
                                          .open(file_name));

    let tiff_header: TiffHeader = try!(utils::read_struct(&mut file));

    let endianess_diff = utils::is_machine_big_endian() && tiff_header.id == INTEL_BYTE_ORDER;

    if cnd_swap!(tiff_header.version, endianess_diff) != TIFF_VERSION {
        return Err(TiffError::UnknownVersion);
    } 

    // Seek to the first TIFF directory
    try!(file.seek(SeekFrom::Start(cnd_swap!(tiff_header.dir_offset, endianess_diff) as u64)));

    let mut num_dir_entries: u16 = try!(utils::read_struct(&mut file));
    num_dir_entries = cnd_swap!(num_dir_entries, endianess_diff);

    // All the `Option`s below need to be read; if any is missing, we will panic
    let mut img_width: Option<u32> = None;
    let mut img_height: Option<u32> = None;
    let mut num_strips: Option<usize> = None;
    let mut bits_per_sample: Option<usize> = None;
    let mut rows_per_strip: Option<usize> = None;
    let mut photometric_interpretation: u32 = PHMET_BLACK_IS_ZERO;
    let mut samples_per_pixel: Option<usize> = None;
    let mut strip_offsets: Option<Vec<u32>> = None;

    let mut next_field_pos = file.seek(SeekFrom::Current(0)).unwrap();
    for _ in 0..num_dir_entries {
        try!(file.seek(SeekFrom::Start(next_field_pos)));

        let mut tiff_field: TiffField = try!(utils::read_struct(&mut file));
        
        next_field_pos = file.seek(SeekFrom::Current(0)).unwrap();

        preprocess_tiff_field(&mut tiff_field, endianess_diff);

        match tiff_field.tag {
            TAG_IMAGE_WIDTH => img_width = Some(tiff_field.value),

            TAG_IMAGE_HEIGHT => img_height = Some(tiff_field.value),

            TAG_BITS_PER_SAMPLE => bits_per_sample = Some(try!(parse_tag_bits_per_sample(&mut file, &tiff_field, endianess_diff))),

            TAG_COMPRESSION => if tiff_field.value != NO_COMPRESSION { return Err(TiffError::CompressionNotSupported); },

            TAG_PHOTOMETRIC_INTERPRETATION => photometric_interpretation = tiff_field.value,

            TAG_STRIP_OFFSETS => {
                num_strips = Some(tiff_field.count as usize);
                if num_strips.unwrap() == 1 {
                    strip_offsets = Some(vec![tiff_field.value]);
                } else {
                    try!(file.seek(SeekFrom::Start(tiff_field.value as u64)));
                    strip_offsets = Some(try!(utils::read_vec(&mut file, num_strips.unwrap())));
                    for sofs in strip_offsets.iter_mut().next().unwrap() {
                        *sofs = cnd_swap!(*sofs, endianess_diff);
                    }
                }
            },

            TAG_SAMPLES_PER_PIXEL => samples_per_pixel = Some(tiff_field.value as usize),

            TAG_ROWS_PER_STRIP => rows_per_strip = Some(tiff_field.value as usize),

            TAG_PLANAR_CONFIGURATION => if tiff_field.value != PLANAR_CONFIGURATION_CHUNKY { return Err(TiffError::UnsupportedPlanarConfig); },

            _ => { } // Ignore unknown tags
        }
    }

    if 0 == rows_per_strip.unwrap() && 1 == num_strips.unwrap() {
        // If there is only 1 strip, it contains all the rows
        rows_per_strip = Some(*img_height.iter().next().unwrap() as usize	);
    }

    try!(validate_tiff_format(samples_per_pixel.unwrap(), photometric_interpretation));

    let pix_fmt = determine_pixel_format(samples_per_pixel.unwrap(), bits_per_sample.unwrap()); 

    let bytes_per_line = img_width.unwrap() as usize * bytes_per_pixel(pix_fmt);
    let mut pixels = utils::alloc_uninitialized::<u8>(img_height.unwrap() as usize * bytes_per_line as usize);

    let mut curr_line = 0;
    for strip_ofs in strip_offsets.unwrap() {
        try!(file.seek(SeekFrom::Start(strip_ofs as u64)));

        let mut strip_row = 0;
        while strip_row < rows_per_strip.unwrap() && curr_line < img_height.unwrap() {
            let img_line = &mut pixels[range!(curr_line as usize * bytes_per_line, bytes_per_line)];
            
            try!(file.read_exact(img_line));

            strip_row += 1;
            curr_line += 1;
        }
    }
    
    let mut img = Image::new_from_pixels(img_width.unwrap(), img_height.unwrap(), pix_fmt, None, pixels);

    if photometric_interpretation == PHMET_WHITE_IS_ZERO {
        // Reverse the values so that "black" is zero, "white" is 255 or 65535.
        match pix_fmt {
            PixelFormat::Mono8 => negate_grayscale_8(&mut img),
            PixelFormat::Mono16 => negate_grayscale_16(&mut img),
            _ => panic!()
        }
    }
    
    if (pix_fmt == PixelFormat::Mono16 || pix_fmt == PixelFormat::RGB16) && endianess_diff {
        utils::swap_words16(&mut img);
    }
    
    Ok(img)
}


/// Returns metadata (width, height, ...) without reading the pixel data.
pub fn get_tiff_metadata(file_name: &str) -> Result<(u32, u32, PixelFormat, Option<Palette>), TiffError> {
    let mut file = try!(OpenOptions::new().read(true)
                                          .write(false)
                                          .open(file_name));

    let tiff_header: TiffHeader = try!(utils::read_struct(&mut file));

    let endianess_diff = utils::is_machine_big_endian() && tiff_header.id == INTEL_BYTE_ORDER;

    if cnd_swap!(tiff_header.version, endianess_diff) != TIFF_VERSION {
        return Err(TiffError::UnknownVersion);
    } 

    // Seek to the first TIFF directory
    try!(file.seek(SeekFrom::Start(cnd_swap!(tiff_header.dir_offset, endianess_diff) as u64)));

    let mut num_dir_entries: u16 = try!(utils::read_struct(&mut file));
    num_dir_entries = cnd_swap!(num_dir_entries, endianess_diff);

    // All the `Option`s below need to be read; if any is missing, we will panic
    let mut img_width: Option<u32> = None;
    let mut img_height: Option<u32> = None;
    let mut bits_per_sample: Option<usize> = None;
    let mut photometric_interpretation: u32 = PHMET_BLACK_IS_ZERO;
    let mut samples_per_pixel: Option<usize> = None;

    let mut next_field_pos = file.seek(SeekFrom::Current(0)).unwrap();
    for _ in 0..num_dir_entries {
        try!(file.seek(SeekFrom::Start(next_field_pos)));

        let mut tiff_field: TiffField = try!(utils::read_struct(&mut file));
        
        next_field_pos = file.seek(SeekFrom::Current(0)).unwrap();

        preprocess_tiff_field(&mut tiff_field, endianess_diff);

        match tiff_field.tag {
            TAG_IMAGE_WIDTH => img_width = Some(tiff_field.value),

            TAG_IMAGE_HEIGHT => img_height = Some(tiff_field.value),

            TAG_BITS_PER_SAMPLE => bits_per_sample = Some(try!(parse_tag_bits_per_sample(&mut file, &tiff_field, endianess_diff))),

            TAG_COMPRESSION => if tiff_field.value != NO_COMPRESSION { return Err(TiffError::CompressionNotSupported); },

            TAG_PHOTOMETRIC_INTERPRETATION => photometric_interpretation = tiff_field.value,

            TAG_SAMPLES_PER_PIXEL => samples_per_pixel = Some(tiff_field.value as usize),

            TAG_PLANAR_CONFIGURATION => if tiff_field.value != PLANAR_CONFIGURATION_CHUNKY { return Err(TiffError::UnsupportedPlanarConfig); },

            _ => { } // Ignore unknown tags
        }
    }

    try!(validate_tiff_format(samples_per_pixel.unwrap(), photometric_interpretation));

    let pix_fmt = determine_pixel_format(samples_per_pixel.unwrap(), bits_per_sample.unwrap()); 

    Ok((img_width.unwrap(), img_height.unwrap(), pix_fmt, None))
}


pub fn save_tiff(img: &Image, file_name: &str) -> Result<(), TiffError>   {
    match img.get_pixel_format() {
        PixelFormat::Mono8 |
        PixelFormat::Mono16 |
        PixelFormat::RGB8 |
        PixelFormat::RGB16 => { },
        
        _ => panic!()
    }

    let mut file = try!(OpenOptions::new().read(false)
                                          .write(true)
                                          .create(true)
                                          .open(file_name));
    let is_be = utils::is_machine_big_endian();

    // Note: a 16-bit value (TAG_TYPE_WORD) stored in the 32-bit `tiff_field.value` has to be
    // always "left-aligned", i.e. stored in the lower-address two bytes in the file,
    // regardless of the file's and machine's endianess.
    //
    // This means that on a big-endian machine it has to be always shifted left by 16 bits
    // prior to writing to file.

    let tiff_header = TiffHeader { id: if is_be { MOTOROLA_BYTE_ORDER } else { INTEL_BYTE_ORDER },
                                   version: TIFF_VERSION,
                                   dir_offset: size_of::<TiffHeader>() as u32 };

    try!(utils::write_struct(&tiff_header, &mut file));
     
    let num_dir_entries: u16 = 10;
    try!(utils::write_struct(&num_dir_entries, &mut file));

    let next_dir_offset = 0u32;

    let mut field = TiffField { tag: TAG_IMAGE_WIDTH,
                                ftype: TAG_TYPE_WORD,
                                count: 1,
                                value: img.get_width() as u32 };
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_IMAGE_HEIGHT,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        value: img.get_height() as u32 };
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_BITS_PER_SAMPLE,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        value: bytes_per_channel(img.get_pixel_format()) as u32 * 8 };
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_COMPRESSION,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        value: NO_COMPRESSION };
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_PHOTOMETRIC_INTERPRETATION,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        value: match img.get_pixel_format()
                               {
                                   PixelFormat::Mono8 | PixelFormat::Mono16 => PHMET_BLACK_IS_ZERO,
                                   PixelFormat::RGB8 | PixelFormat::RGB16 => PHMET_RGB,
                                   _ => panic!()
                               }
                      }; 
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_STRIP_OFFSETS,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        // We write the header, num. of directory entries, 10 fields and a next directory offset (==0); pixel data starts next
                        value: (size_of_val(&tiff_header) +
                               size_of_val(&num_dir_entries) +
                               10 * size_of_val(&field) +
                               size_of_val(&next_dir_offset)) as u32
                      };
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_SAMPLES_PER_PIXEL,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        value: get_num_channels(img.get_pixel_format()) as u32 };
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_ROWS_PER_STRIP,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        value: img.get_height() as u32 }; // There is only one strip for the whole image
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_STRIP_BYTE_COUNTS,
                        ftype: TAG_TYPE_DWORD,
                        count: 1,
                        value: img.get_bytes_per_line() as u32 * img.get_height() }; // There is only one strip for the whole image
    try!(utils::write_struct(&field, &mut file));

    field = TiffField { tag: TAG_PLANAR_CONFIGURATION,
                        ftype: TAG_TYPE_WORD,
                        count: 1,
                        value: PLANAR_CONFIGURATION_CHUNKY };
    if is_be { field.value <<= 16; }
    try!(utils::write_struct(&field, &mut file));

    // Write the next directory offset (0 = no other directories)
    try!(utils::write_struct(&next_dir_offset, &mut file));

    try!(file.write_all(img.get_raw_pixels()));

    Ok(())
}