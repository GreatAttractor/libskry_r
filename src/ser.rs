//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   SER support.
//

use image::{bytes_per_channel, bytes_per_pixel, Image, ImageError, Palette, PixelFormat};
use img_seq_priv::ImageProvider;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use utils;


#[derive(PartialEq)]
enum SerColorFormat {
    Mono      = 0,
    BayerRGGB = 8,
    BayerGRBG = 9,
    BayerGBRG = 10,
    BayerBGGR = 11,
    BayerCYYM = 16,
    BayerYCMY = 17,
    BayerYMCY = 18,
    BayerMYYC = 19,
    RGB       = 100,
    BGR       = 101
}


#[derive(Debug)]
pub enum SerError {
    Io(io::Error),
    UnsupportedFormat,
    InvalidBitDepth
}


impl From<io::Error> for SerError {
    fn from(err: io::Error) -> SerError { SerError::Io(err) }
}


fn get_ser_color_fmt(color_id: u32) -> Result<SerColorFormat, SerError> {

    match color_id {
        color_id if color_id == SerColorFormat::Mono      as u32 => Ok(SerColorFormat::Mono),
//TODO: uncomment once demosaicing is ported
//        color_id if color_id == SerColorFormat::BayerRGGB as u32 => Ok(SerColorFormat::BayerRGGB),
//        color_id if color_id == SerColorFormat::BayerGRBG as u32 => Ok(SerColorFormat::BayerGRBG),
//        color_id if color_id == SerColorFormat::BayerGBRG as u32 => Ok(SerColorFormat::BayerGBRG),
//        color_id if color_id == SerColorFormat::BayerBGGR as u32 => Ok(SerColorFormat::BayerBGGR),
        color_id if color_id == SerColorFormat::RGB       as u32 => Ok(SerColorFormat::RGB),
        color_id if color_id == SerColorFormat::BGR       as u32 => Ok(SerColorFormat::BGR),
        _ => Err(SerError::UnsupportedFormat)
    }
}


/// See comment for SerHeader::little_endian
const SER_LITTLE_ENDIAN: u32 = 0;


#[repr(C, packed)]
struct SerHeader {
    signature: [u8; 14],
    camera_series_id: u32,
    color_id: u32,
    // Online documentation claims this is 0 when 16-bit pixel data
    // is big-endian, but the meaning is actually reversed.
    little_endian: u32,
    img_width: u32,
    img_height: u32,
    bits_per_channel: u32,
    frame_count: u32,
    observer: [u8; 40],
    instrument: [u8; 40],
    telescope: [u8; 40],
    date_time: i64,
    date_time_utc: i64
}


pub struct SerFile {
    file_name: String,
    /// Becomes empty after calling `deactivate()`.
    file: Option<File>,
    /// Concerns 16-bit pixel data
    little_endian_data: bool,
    ser_color_fmt: SerColorFormat,
    pix_fmt: PixelFormat,
    num_images: usize,
    width: u32,
    height: u32
}


/// Reverses RGB<->BGR.
fn reverse_rgb<T>(line: &mut [T]) {
    for x in 0 .. line.len()/3 {
        line.swap(3*x, 3*x + 2);
    }
}


impl SerFile {

    pub fn new(file_name: &str) -> Result<Box<dyn ImageProvider>, SerError> {

        let mut file = OpenOptions::new().read(true).write(false).open(file_name)?;

        let fheader: SerHeader = utils::read_struct(&mut file)?;

        let ser_color_fmt = get_ser_color_fmt(u32::from_le(fheader.color_id))?;

        let bits_per_channel = u32::from_le(fheader.bits_per_channel);
        if bits_per_channel > 16 {
            return Err(SerError::InvalidBitDepth);
        }

        let pix_fmt = match ser_color_fmt {
            SerColorFormat::Mono => if bits_per_channel <= 8 { PixelFormat::Mono8 } else { PixelFormat::Mono16 },
            SerColorFormat::RGB | SerColorFormat::BGR => if bits_per_channel <= 8 { PixelFormat::RGB8 } else { PixelFormat::RGB16 },
//TODO: uncomment once demosaicing is ported
//            SerColorFormat::BayerBGGR => if bits_per_channel <= 8 { PixelFormat::CfaBGGR8 } else { PixelFormat::CfaBGGR16 },
//            SerColorFormat::BayerGBRG => if bits_per_channel <= 8 { PixelFormat::CfaGBRG8 } else { PixelFormat::CfaGBRG16 },
//            SerColorFormat::BayerGRBG => if bits_per_channel <= 8 { PixelFormat::CfaGRBG8 } else { PixelFormat::CfaGRBG16 },
//            SerColorFormat::BayerRGGB => if bits_per_channel <= 8 { PixelFormat::CfaRGGB8 } else { PixelFormat::CfaRGGB16 },
            _ => panic!() // cannot happen due, thanks get_ser_color_fmt()
        };

        let little_endian_data = u32::from_le(fheader.little_endian) == SER_LITTLE_ENDIAN;
        let width = u32::from_le(fheader.img_width);
        let height = u32::from_le(fheader.img_height);
        let num_images = u32::from_le(fheader.frame_count) as usize;

        Ok(Box::new(
            SerFile{ file_name: String::from(file_name),
                     file: None,
                     ser_color_fmt,
                     little_endian_data,
                     pix_fmt,
                     num_images,
                     width,
                     height }
        ))
    }
}

impl ImageProvider for SerFile {
    fn get_img(&mut self, idx: usize) -> Result<Image, ImageError> {
        assert!(idx < self.num_images);

        if self.file.is_none() {
            self.file = Some(OpenOptions::new().read(true).write(false).open(&self.file_name)?);
        }

        let file: &mut File = self.file.iter_mut().next().unwrap();

        let mut img = Image::new(self.width, self.height, self.pix_fmt, None, false);

        let frame_size = (self.width * self.height) as usize * bytes_per_pixel(self.pix_fmt);

        file.seek(SeekFrom::Start((size_of::<SerHeader>() + idx * frame_size) as u64))?;

        for y in 0..self.height {
            file.read_exact(img.get_line_raw_mut(y))?;

            if self.ser_color_fmt == SerColorFormat::BGR {
                match self.pix_fmt {
                    PixelFormat::RGB8 => reverse_rgb(img.get_line_mut::<u8>(y)),
                    PixelFormat::RGB16 => reverse_rgb(img.get_line_mut::<u16>(y)),
                    _ => panic!() // cannot happen
                }
            }
        }

        if bytes_per_channel(self.pix_fmt) > 1 && (utils::is_machine_big_endian() ^ !self.little_endian_data) {
            utils::swap_words16(&mut img);
        }

        Ok(img)
    }


    fn get_img_metadata(&self, _: usize) -> Result<(u32, u32, PixelFormat, Option<Palette>), ImageError> {
        Ok((self.width, self.height, self.pix_fmt, None))
    }


    fn img_count(&self) -> usize {
        self.num_images
    }


    fn deactivate(&mut self) {
        self.file = None;
    }
}
