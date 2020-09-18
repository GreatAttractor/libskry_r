//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   AVI support.
//

use bmp;
use image::{Image, ImageError, Palette, PixelFormat, bytes_per_pixel};
use img_seq_priv::{ImageProvider};
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{Read, Seek, SeekFrom};
use std::mem::{size_of, size_of_val};
use std::slice;
use utils;


/// Four Character Code (FCC)
type FourCC = [u8; 4];


fn fcc_equals(fcc1: &FourCC, fcc2: &[u8]) -> bool {
    fcc1[0] == fcc2[0] &&
    fcc1[1] == fcc2[1] &&
    fcc1[2] == fcc2[2] &&
    fcc1[3] == fcc2[3]
}

#[repr(C, packed)]
struct AviFileHeader {
    riff:      FourCC,
    file_size: u32,
    avi:       FourCC
}


#[repr(C, packed)]
struct AviFrame {
    left:   i16,
    top:    i16,
    right:  i16,
    bottom: i16
}


#[repr(C, packed)]
struct AviStreamHeader {
    fcc_type:       FourCC,
    fcc_handler:    FourCC,
    flags:          u32,
    priority:       u16,
    language:       u16,
    initial_frames: u32,
    scale:          u32,
    rate:           u32,
    start:          u32,
    length:         u32,
    suggested_buffer_size: u32,
    quality:        u32,
    sample_size:    u32,
    frame:          AviFrame
}


#[repr(C, packed)]
struct AviList {
    list:      FourCC, // Contains "LIST"
    list_size: u32, // Does not include `list` and `list_type`
    list_type: FourCC
}


#[repr(C, packed)]
struct AviChunk {
    ck_id:   FourCC,
    ck_size: u32 // Does not include `ck_id` and `ck_size`
}


/// List or chunk (used when skipping `JUNK` chunks)
#[repr(C, packed)]
struct AviFragment {
    fcc:  FourCC,
    size: u32
}


#[repr(C, packed)]
struct AviMainHeader {
    microsec_per_frame:    u32,
    max_bytes_per_sec:     u32,
    padding_granularity:   u32,
    flags:                 u32,
    total_frames:          u32,
    initial_frames:        u32,
    streams:               u32,
    suggested_buffer_size: u32,
    width:                 u32,
    height:                u32,
    reserved:              [u32; 4]
}


#[repr(C, packed)]
struct AviOldIndex {
    chunk_id: FourCC,
    flags: u32,

    /// Offset of frame contents counted from the beginning of the `movi` list's `list_type` field OR absolute file offset.
    offset: u32,

    frame_size: u32
}


const AVIF_HAS_INDEX: u32 = 0x00000010;


#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum AviPixelFormat {
    /// DIB, RGB 8-bit.
    DibRGB8,
    /// DIB, 256-color 8-bit RGB palette.
    DibPal8,
    /// DIB, 256-color grayscale palette.
    DibMono8,
    /// 8 bits per pixel, luminance only.
    Y800
}


#[derive(Debug)]
pub enum AviError {
    Io(io::Error),
    MalformedFile,
    IndexNotPresent,
    UnsupportedFormat,
    InvalidFrame(usize)
}


impl From<io::Error> for AviError {
    fn from(err: io::Error) -> AviError { AviError::Io(err) }
}


impl From<io::Error> for ImageError {
    fn from(err: io::Error) -> ImageError { ImageError::AviError(AviError::Io(err)) }
}


impl From<AviError> for ImageError {
    fn from(err: AviError) -> ImageError { ImageError::AviError(err) }
}


fn is_dib(avi_pix_fmt: AviPixelFormat) -> bool {
    match avi_pix_fmt {
        AviPixelFormat::DibRGB8 |
        AviPixelFormat::DibPal8 |
        AviPixelFormat::DibMono8 => true,

        _ => false
    }
}


fn avi_to_image_pix_fmt(avi_pix_fmt: AviPixelFormat) -> PixelFormat {
    match avi_pix_fmt {
        AviPixelFormat::DibMono8 | AviPixelFormat::Y800 => PixelFormat::Mono8,
        AviPixelFormat::DibRGB8 => PixelFormat::RGB8,
        AviPixelFormat::DibPal8 => PixelFormat::Pal8
    }
}


pub struct AviFile {
    file_name: String,

    /// Becomes empty after calling `deactivate()`.
    file: Option<File>,

    /// Absolute file offsets (point to each frame's `AVI_chunk`).
    frame_offsets: Vec<u64>,

    /// Valid for an AVI with palette.
    palette: Option<Palette>,

    avi_pix_fmt: AviPixelFormat,

    num_images: usize,

    width: u32,
    height: u32
}


impl AviFile {
    pub fn new(file_name: &str) -> Result<Box<dyn ImageProvider>, AviError> {
        // Expected AVI file structure:
        //
        //     RIFF/AVI                         // AVI_file_header
        //     LIST: hdrl
        //     | avih                           // AVI_main_header
        //     | LIST: strl
        //     | | strh                         // AVI_stream_header
        //     | | for DIB: strf                // bitmap_info_header
        //     | | for DIB/8-bit: BMP palette   // BMP_palette
        //     | | ...
        //     | | (ignored)
        //     | | ...
        //     | |_____
        //     |_________
        //     ...
        //     (ignored; possibly 'JUNK' chunks, LIST:INFO)
        //     ...
        //     LIST: movi
        //     | ...
        //     | (frames)
        //     | ...
        //     |_________
        //     ...
        //     (ignored)
        //     ...
        //     idx1
        //       ...
        //       (index entries)                // AVI_old_index
        //       ...

        let mut chunk: AviChunk;
        let mut list: AviList;
        let mut last_chunk_pos: u64;
        let mut last_chunk_size: u32;

        let mut file = OpenOptions::new().read(true)
                                         .write(false)
                                         .open(file_name)?;

        let fheader: AviFileHeader = utils::read_struct(&mut file)?;

        if !fcc_equals(&fheader.riff, "RIFF".as_bytes()) ||
           !fcc_equals(&fheader.avi, "AVI ".as_bytes()) {

            return Err(AviError::MalformedFile);
        }

        let header_list_pos = file.seek(SeekFrom::Current(0)).unwrap();
        let header_list: AviList = utils::read_struct(&mut file)?;

        if !fcc_equals(&header_list.list, "LIST".as_bytes()) ||
           !fcc_equals(&header_list.list_type, "hdrl".as_bytes()) {

            return Err(AviError::MalformedFile);
        }

        // Returns on error
        macro_rules! read_chunk { () => {
            last_chunk_pos = file.seek(SeekFrom::Current(0)).unwrap();
            chunk = utils::read_struct(&mut file)?;
            last_chunk_size = u32::from_le(chunk.ck_size);
        }};

        read_chunk!();

        if !fcc_equals(&chunk.ck_id, "avih".as_bytes()) {
            return Err(AviError::MalformedFile);
        }

        let avi_header: AviMainHeader = utils::read_struct(&mut file)?;

        // This may be zero; if so, we'll use the stream header's `length` field
        let mut num_images = u32::from_le(avi_header.total_frames) as usize;
        let width = u32::from_le(avi_header.width);
        let height = u32::from_le(avi_header.height);

        if u32::from_le(avi_header.flags) & AVIF_HAS_INDEX == 0 {
            return Err(AviError::IndexNotPresent);
        }

        macro_rules! seek_to_next_list_or_chunk { () => {
            file.seek(SeekFrom::Start(
                last_chunk_pos +
                last_chunk_size as u64 +
                size_of_val(&{let x = chunk.ck_id; x}) as u64 +
                size_of_val(&{let x = chunk.ck_size; x}) as u64)
            )?;
        }};

        seek_to_next_list_or_chunk!();

        // Read the stream list
        list = utils::read_struct(&mut file)?;

        if !fcc_equals(&list.list, "LIST".as_bytes()) ||
           !fcc_equals(&list.list_type, "strl".as_bytes()) {

            return Err(AviError::MalformedFile);
        }

        // Read the stream header
        read_chunk!();

        if !fcc_equals(&chunk.ck_id, "strh".as_bytes()) {
            return Err(AviError::MalformedFile);
        }

        let mut stream_header: AviStreamHeader = utils::read_struct(&mut file)?;

        if !fcc_equals(&stream_header.fcc_type, "vids".as_bytes()) {
            return Err(AviError::MalformedFile);
        }

        if fcc_equals(&stream_header.fcc_handler, "\0\0\0".as_bytes()) {
            // Empty 'fcc_handler' means DIB by default
            stream_header.fcc_handler[0] = 'D' as u8;
            stream_header.fcc_handler[1] = 'I' as u8;
            stream_header.fcc_handler[2] = 'B' as u8;
            stream_header.fcc_handler[3] = ' ' as u8;
        }

        if !fcc_equals(&stream_header.fcc_handler, "DIB ".as_bytes()) &&
           !fcc_equals(&stream_header.fcc_handler, "Y800".as_bytes()) &&
           !fcc_equals(&stream_header.fcc_handler, "Y8  ".as_bytes()) {

            return Err(AviError::UnsupportedFormat);
        }
        let is_dib = fcc_equals(&stream_header.fcc_handler, "DIB ".as_bytes());

        if num_images == 0 { num_images = stream_header.length as usize; }

        // Seek to and read the stream format
        seek_to_next_list_or_chunk!();
        read_chunk!();

        if !fcc_equals(&chunk.ck_id, "strf".as_bytes()) {
            return Err(AviError::MalformedFile);
        }

        let bmp_hdr: bmp::BitmapInfoHeader = utils::read_struct(&mut file)?;

        let palette: Option<Palette> = None;

        if is_dib && u32::from_le(bmp_hdr.compression) != bmp::BI_BITFIELDS && u32::from_le(bmp_hdr.compression) != bmp::BI_RGB ||
            u16::from_le(bmp_hdr.planes) != 1 ||
            u16::from_le(bmp_hdr.bit_count) != 8 && u16::from_le(bmp_hdr.bit_count) != 24 {

            return Err(AviError::UnsupportedFormat);
        }

        let avi_pix_fmt;

        if is_dib && u16::from_le(bmp_hdr.bit_count) == 8 {
            let bmp_palette: bmp::BmpPalette = utils::read_struct(&mut file)?;

            let mut clr_used = u32::from_le(bmp_hdr.clr_used);
            if clr_used == 0 { clr_used = 256; }
            let palette = Some(bmp::convert_bmp_palette(clr_used, &bmp_palette));

            avi_pix_fmt = if bmp::is_mono8_palette(palette.iter().next().unwrap()) { AviPixelFormat::DibMono8 } else { AviPixelFormat::DibPal8 }
        } else if is_dib && u16::from_le(bmp_hdr.bit_count) == 24 {
            avi_pix_fmt = AviPixelFormat::DibRGB8;
        } else {
            avi_pix_fmt = AviPixelFormat::Y800;
        }

        // Jump to the location immediately after `hdrl`
        file.seek(SeekFrom::Start(
            header_list_pos +
            u32::from_le(header_list.list_size) as u64 +
            size_of_val(&header_list.list) as u64 +
            size_of_val(&{let x = header_list.list_size; x}) as u64)
        )?;

        // Skip any additional fragments (e.g. `JUNK` chunks)
        let mut stored_pos: u64;
        loop {
            stored_pos = file.seek(SeekFrom::Current(0)).unwrap();

            let fragment: AviFragment = utils::read_struct(&mut file)?;

            if fcc_equals(&fragment.fcc, "LIST".as_bytes()) {
                let list_type: FourCC = utils::read_struct(&mut file)?;

                // Found a list; if it is the `movi` list, move the file pointer back;
                // the list will be re-read after the current `while` loop
                if fcc_equals(&list_type, "movi".as_bytes()) {
                    file.seek(SeekFrom::Start(stored_pos))?;
                    break;
                } else {
                    // Not the `movi` list; skip it.
                    // Must rewind back by length of the `size` field,
                    // because in a list it is not counted in `size`.
                    file.seek(SeekFrom::Current(-(size_of_val(&{let x = fragment.size; x}) as i64)))?;
                }
            }

            // Skip the current fragment, whatever it is
            file.seek(SeekFrom::Current(u32::from_le(fragment.size) as i64))?;
        }

        list = utils::read_struct(&mut file)?;

        if !fcc_equals(&list.list, "LIST".as_bytes()) ||
           !fcc_equals(&list.list_type, "movi".as_bytes()) {

            return Err(AviError::MalformedFile);
        }

        let frame_chunks_start_ofs = file.seek(SeekFrom::Current(0)).unwrap() - size_of_val(&list.list_type) as u64;

        // Jump to the old-style AVI index
        file.seek(SeekFrom::Current(u32::from_le(list.list_size) as i64 - size_of_val(&{ let x = list.list_size; x}) as i64))?;

        read_chunk!();

        if !fcc_equals(&chunk.ck_id, "idx1".as_bytes()) ||
           (u32::from_le(chunk.ck_size) as usize) < num_images * size_of::<AviOldIndex>() {

            return Err(AviError::MalformedFile);
        }

        // Index may contain bogus entries, this will make it longer than num_images * sizeof(AviOldIndex)
        let index_length = u32::from_le(chunk.ck_size);
        if index_length % size_of::<AviOldIndex>() as u32 != 0 {

            return Err(AviError::MalformedFile);
        }

        // Absolute byte offsets of each frame's contents in the file
        let mut frame_offsets = Vec::<u64>::with_capacity(num_images);

        // We have just checked it is divisible
        let mut avi_old_index = utils::alloc_uninitialized::<AviOldIndex>(index_length as usize / size_of::<AviOldIndex>());
        file.read_exact( unsafe { slice::from_raw_parts_mut(avi_old_index.as_mut_slice().as_ptr() as *mut u8, index_length as usize) })?;

        let mut line_byte_count = width as usize * bytes_per_pixel(avi_to_image_pix_fmt(avi_pix_fmt));
        if is_dib { line_byte_count = upmult!(line_byte_count, 4); }

        let frame_byte_count = line_byte_count * height as usize;

        for entry in &avi_old_index {
            // Ignore bogus entries (they may have "7Fxx" as their ID)
            if fcc_equals(&entry.chunk_id, "00db".as_bytes()) ||
               fcc_equals(&entry.chunk_id, "00dc".as_bytes()) {

                if u32::from_le(entry.frame_size) as usize != frame_byte_count {
                    return Err(AviError::MalformedFile);
                } else {
                    frame_offsets.push(frame_chunks_start_ofs + u32::from_le(entry.offset) as u64);
                }
            }
        }

        // We assumed the frame offsets in the index were relative to the `movi` list; however, they may be actually
        // absolute file offsets. Check and update the offsets array.

        // Try to read the first frame's preamble
        file.seek(SeekFrom::Start(u32::from_le(avi_old_index[0].offset) as u64))?;
        read_chunk!();

        if (fcc_equals(&chunk.ck_id, "00db".as_bytes()) || fcc_equals(&chunk.ck_id, "00dc".as_bytes())) &&
           u32::from_le(chunk.ck_size) as usize == frame_byte_count {

            // Indeed, index frame offsets are absolute; must correct the values
            for ofs in &mut frame_offsets {
                *ofs -= frame_chunks_start_ofs;
            }
        }

        Ok(Box::new(
            AviFile{ file_name: String::from(file_name),
                     file: None,
                     frame_offsets,
                     palette,
                     avi_pix_fmt,
                     num_images,
                     width,
                     height }
        ))
    }
}


impl ImageProvider for AviFile {
    fn get_img(&mut self, idx: usize) -> Result<Image, ImageError> {
        if self.file.is_none() {
            self.file = Some(
                OpenOptions::new()
                    .read(true)
                    .write(false)
                    .open(&self.file_name)?
            );
        }

        let file: &mut File = self.file.iter_mut().next().unwrap();

        file.seek(SeekFrom::Start(self.frame_offsets[idx]))?;

        let chunk: AviChunk = utils::read_struct(file)?;

        let mut src_line_byte_count = self.width as usize * bytes_per_pixel(avi_to_image_pix_fmt(self.avi_pix_fmt));
        let is_dib = is_dib(self.avi_pix_fmt);
        if is_dib { src_line_byte_count = upmult!(src_line_byte_count, 4); }

        if !fcc_equals(&chunk.ck_id, "00db".as_bytes()) && !fcc_equals(&chunk.ck_id, "00dc".as_bytes()) ||
            u32::from_le(chunk.ck_size) as usize != src_line_byte_count * self.height as usize {

            return Err(ImageError::AviError(AviError::InvalidFrame(idx)));
        }

        let mut img = Image::new(self.width, self.height, avi_to_image_pix_fmt(self.avi_pix_fmt), self.palette.clone(), false);

        let mut src_line = utils::alloc_uninitialized::<u8>(src_line_byte_count);

        for y in 0..self.height {
            file.read_exact(&mut src_line)?;

            let bpl = img.get_bytes_per_line();

            // Line order in a DIB is reversed
            let img_line = if is_dib { img.get_line_raw_mut(self.height - y - 1) }
                           else { img.get_line_raw_mut(y) };

            if self.avi_pix_fmt == AviPixelFormat::DibRGB8 {
                // Rearrange channels to RGB order
                for x in 0..self.width as usize {
                    img_line[3*x + 0] = src_line[3*x + 2];
                    img_line[3*x + 1] = src_line[3*x + 1];
                    img_line[3*x + 2] = src_line[3*x + 0];
                }
            }
            else {
                img_line.copy_from_slice(&src_line[0..bpl]);
            }
        }

        Ok(img)
    }


    fn get_img_metadata(&self, _: usize) -> Result<(u32, u32, PixelFormat, Option<Palette>), ImageError> {
        Ok((self.width, self.height, avi_to_image_pix_fmt(self.avi_pix_fmt), self.palette.clone()))
    }


    fn img_count(&self) -> usize {
        self.num_images
    }


    fn deactivate(&mut self) {
        self.file = None;
    }
}
