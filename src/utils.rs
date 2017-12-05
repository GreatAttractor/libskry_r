//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Utilities.
//

use defs::{Point, WHITE_8BIT};
use filters;
use image::{Image, PixelFormat};
use std;
use std::fs::File;
use std::io::{self, Read, Write};
use std::slice;


macro_rules! sqr {
    ($x:expr) => { ($x) * ($x) }
}


/// Rounds `x` up to the closest multiple of `n`.
macro_rules! upmult {
    ($x:expr, $n:expr) => { (($x) + ($n) - 1) / ($n) * ($n) }
}


/// Produces a range of specified length.
macro_rules! range { ($start:expr, $len:expr) => { $start .. $start + $len } }


/// Returns ceil(a/b).
macro_rules! updiv {
    ($a:expr, $b:expr) => { (($a) + ($b) - 1) / ($b) }
}


/// Returns barycentric coordinates `(u, v)` of point `p` in the triangle `(v0, v1, v2)` (`p` can be outside the triangle).
macro_rules! calc_barycentric_coords {
    ($p:expr, $v0:expr, $v1:expr, $v2:expr) => {
        ((($v1.y - $v2.y) as f32 * ($p.x as f32 - $v2.x as f32) + ($v2.x - $v1.x) as f32 * ($p.y as f32  - $v2.y as f32)) as f32 / (($v1.y - $v2.y) * ($v0.x - $v2.x) + ($v2.x - $v1.x) * ($v0.y - $v2.y)) as f32,
         (($v2.y - $v0.y) as f32 * ($p.x as f32 - $v2.x as f32) + ($v0.x - $v2.x) as f32 * ($p.y as f32  - $v2.y as f32)) as f32 / (($v1.y - $v2.y) * ($v0.x - $v2.x) + ($v2.x - $v1.x) * ($v0.y - $v2.y)) as f32)
    }
}


pub fn read_struct<T, R: Read>(read: &mut R) -> io::Result<T> {
    let num_bytes = ::std::mem::size_of::<T>();
    unsafe {
        let mut s = ::std::mem::uninitialized();
        let buffer = slice::from_raw_parts_mut(&mut s as *mut T as *mut u8, num_bytes);
        match read.read_exact(buffer) {
            Ok(()) => Ok(s),
            Err(e) => { ::std::mem::forget(s); Err(e) }
        }
    }
}


pub fn read_vec<T>(file: &mut File, len: usize) -> io::Result<Vec<T>> {
    let mut vec = alloc_uninitialized::<T>(len);
    let num_bytes = ::std::mem::size_of::<T>() * vec.len();
    let buffer = unsafe{ slice::from_raw_parts_mut(vec[..].as_mut_ptr() as *mut u8, num_bytes) };
    try!(file.read_exact(buffer));
    Ok(vec)
}


pub fn write_struct<T, W: Write>(obj: &T, write: &mut W) -> Result<(), io::Error> {
    let num_bytes = ::std::mem::size_of::<T>();
    unsafe {
        let buffer = slice::from_raw_parts(obj as *const T as *const u8, num_bytes);
        write.write_all(buffer)
    }
}


/// Allocates an uninitialized `Vec<T>` having `len` elements.
pub fn alloc_uninitialized<T>(len: usize) -> Vec<T> {
    let mut v = Vec::<T>::with_capacity(len);
    unsafe { v.set_len(len); }
    
    v
}


/// Returns the min. and max. pixel values in a Mono8 image
pub fn find_min_max_brightness(img: &Image) -> (u8, u8) {
    assert!(img.get_pixel_format() == PixelFormat::Mono8);

    let mut bmin: u8 = WHITE_8BIT;
    let mut bmax = 0u8;  
    
    for val in img.get_pixels() {
        if *val < bmin { bmin = *val; }
        if *val > bmax { bmax = *val; }
    }
    
    (bmin, bmax)
}


/// Checks if the specified position `pos` in `img` (Mono8) is appropriate for block matching.
///
/// Uses the distribution of gradient directions around `pos` to decide
/// if the location is safe for block matching. It is not if the image
/// is dominated by a single edge (e.g. the limb of overexposed solar disk,
/// without prominences or resolved spicules). Should block matching be performed
/// in such circumstances, the tracked point would jump along the edge.
///
pub fn assess_gradients_for_block_matching(img: &Image,
                                           pos: Point,
                                           neighborhood_radius: u32) -> bool {

    let block_size = 2*neighborhood_radius + 1;

    let block = img.get_fragment_copy(Point{ x: pos.x - neighborhood_radius as i32,
                                             y: pos.y - neighborhood_radius as i32 },
                                      block_size, block_size, false);

    // Blur to reduce noise impact
    let block_blurred = filters::apply_box_blur(&block, 1, 3);

    let mut line_m1 = block_blurred.get_line_raw(0); // Line at y-1
    let mut line_0  = block_blurred.get_line_raw(1); // Line at y
    let mut line_p1 = block_blurred.get_line_raw(2); // line at y+1

    // Determine the histogram of gradient directions within `block_blurred`

    const NUM_DIRS: usize = 512;

    let mut dirs = [0.0f64; NUM_DIRS]; // Contains sums of gradient lengths

    for y in 1 .. block_size-1 {
        for x in 1 .. (block_size-1) as usize {
            // Calculate gradient using Sobel filter
            let grad_x = 2 * (line_0[x+1] as i32 - line_0[x-1] as i32)
                             + line_m1[x+1] as i32 - line_m1[x-1] as i32
                             + line_p1[x+1] as i32 - line_p1[x-1] as i32;

            let grad_y = 2 * (line_p1[x] as i32 - line_m1[x] as i32)
                             + line_p1[x+1] as i32 - line_m1[x+1] as i32
                             + line_p1[x-1] as i32 - line_m1[x-1] as i32;

            let grad_len = f64::sqrt((sqr!(grad_x) + sqr!(grad_y)) as f64);
            if grad_len > 0.0 {
                let cos_dir = grad_x as f64 / grad_len;
                let mut dir = f64::acos(cos_dir);
                if grad_y < 0 { dir = -dir; }

                let mut index: i32 = NUM_DIRS as i32/2 + (dir * NUM_DIRS as f64 / (2.0 * std::f64::consts::PI)) as i32;

                if index < 0 { index = 0; }
                else if index >= NUM_DIRS as i32 { index = NUM_DIRS as i32 - 1; }

                dirs[index as usize] += grad_len;
            }
        }

        // Move line pointers up
        line_m1 = line_0;
        line_0 = line_p1;
        if y < block_size - 2 {
            line_p1 = block_blurred.get_line_raw(y + 2);
        }
    }

    // Smooth out the histogram to remove spikes (caused by Sobel filter's anisotropy)
    let dirs_smooth = filters::median_filter(&dirs[..], 1);

    // We declare that gradient variability is too low if there are
    // consecutive zeros over more than 1/2 of the histogram and
    // the longest non-zero sequence is shorter than 1/4 of histogram

    let mut zero_count = 0usize;
    let mut nzero_count = 0usize;

    let mut max_zero_count = 0usize;
    let mut max_nzero_count = 0usize;

    for ds in dirs_smooth {
        if ds == 0.0 {
            zero_count += 1;
            if nzero_count > max_nzero_count { max_nzero_count = nzero_count; }
            nzero_count = 0;
        } else {
            if zero_count > max_zero_count { max_zero_count = zero_count; }
            zero_count = 0;
            nzero_count += 1;
        }
    }

    if max_zero_count > NUM_DIRS/3 && max_nzero_count < NUM_DIRS/4 { false } else { true} 
}

/// Changes endianess of 16-bit words.
pub fn swap_words16(img: &mut Image) {
    for val in img.get_pixels_mut::<u16>() {
        *val = u16::swap_bytes(*val);
    }
}


pub fn is_machine_big_endian() -> bool {
    u16::to_be(0x1122u16) == 0x1122u16
}