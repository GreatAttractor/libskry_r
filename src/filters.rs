// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Image filters.
//

use image::{Image, PixelFormat};
use std::cmp::{min, max};
use std::convert::{From};
use std::mem::{swap};
use utils;


//// Result of 3 iterations is quite close to a Gaussian blur
const QUALITY_ESTIMATE_BOX_BLUR_ITERATIONS: usize = 3;


/// Returns a copy of `img` with box blur applied; `img` has to be `Mono8`.
pub fn apply_box_blur(img: &Image, box_radius: u32, iterations: usize) -> Image {
    assert!(img.get_pixel_format() == PixelFormat::Mono8);
    
    let mut blurred_img = Image::new(img.get_width(), img.get_height(),
                                     img.get_pixel_format(), None, false);
    
    box_blur(img.get_pixels::<u8>(), blurred_img.get_pixels_mut::<u8>(),
             img.get_width(), img.get_height(), img.get_width() as usize,
             box_radius, iterations);
    
    blurred_img
}


/// Performs a blurring pass of a range of `length` elements.
///
/// `T` is `u8` or `u32`, `src` points to the range's beginning,
/// `step` is the distance between subsequent elements. 
///
fn box_blur_pass<T>(src: &[T], pix_sum: &mut [u32], box_radius: u32, length: usize, step: usize)
    where T: Copy, u32: From<T> {
                                                                                                     
    // Sum for the first pixel in the current line/row (count the last pixel multiple times if needed)                                   
    pix_sum[0] = (box_radius as u32 + 1) * u32::from(src[0]);                                                   

    let mut i = step;
    while i <= box_radius as usize * step {                                
        pix_sum[0] += u32::from(src[min(i,  (length - 1) * step)]);
        i += step;
    }                                  
                                                                                              
    // Starting region                                                                     
    i = step;
    while i <= min((length - 1) * step, box_radius as usize * step) {
        pix_sum[i] = pix_sum[i - step] - u32::from(src[0]) +
                     u32::from(src[min((length - 1) * step, i + box_radius as usize * step)]);
        i += step;
    }                      
                                                                                              
    if length > box_radius as usize {                                                                                         
        // Middle region                                                                   
        i = (box_radius as usize + 1) * step;
        while i < (length - box_radius as usize) * step {
            pix_sum[i] = pix_sum[i - step] - u32::from(src[i - (box_radius as usize + 1) * step]) +
                         u32::from(src[i + box_radius as usize * step]);
            i += step;
        }                                                     
                                                                                              
        // End region                                                                      
        i = (length - box_radius as usize) * step;
        while i < length * step {
             pix_sum[i] = pix_sum[i - step] -
                          u32::from(src[if i > (box_radius as usize + 1) * step { i - (box_radius as usize + 1) * step } else { 0 }]) +
                          u32::from(src[min(i + box_radius as usize * step, (length - 1) * step)]);
             i += step;          
        }           
    }                                                                                         
}


/// Fills `blurred` with box-blurred contents of `src`.
///
/// Both `src` and `blurred` have `width`*`height` elements (8-bit grayscale).
/// `src_line_stride` is the distance between lines in `src` (which may be a part of a larger image).
/// Line stride in `blurred` equals `width`.
///
fn box_blur(src: &[u8], blurred: &mut [u8],
            width: u32, height: u32, src_line_stride: usize,
            box_radius: u32, iterations: usize) {
    assert!(iterations > 0);
    assert!(box_radius > 0);

    if width == 0 || height == 0 { return; }

    // First the 32-bit unsigned sums of neighborhoods are calculated horizontally
    // and (incrementally) vertically. The max value of a (unsigned) sum is:
    //
    //      (2^8-1) * (box_radius*2 + 1)^2
    //
    // In order for it to fit in 32 bits, box_radius must be below ca. 2^11 - 1.

    assert!((box_radius as u32) < (1u32 << 11) - 1);

    // We need 2 summation buffers to act as source/destination (and then vice versa)
    let mut pix_sum_1 = utils::alloc_uninitialized::<u32>((width * height) as usize);
    let mut pix_sum_2 = utils::alloc_uninitialized::<u32>((width * height) as usize);
    
    let divisor = sqr!(2 * box_radius + 1) as u32;

    // For pixels less than 'box_radius' away from image border, assume
    // the off-image neighborhood consists of copies of the border pixel.

    let mut src_array = &mut pix_sum_1[..];
    let mut dest_array = &mut pix_sum_2[..];


    for n in 0..iterations {
        swap(&mut src_array, &mut dest_array);
        
        // Calculate horizontal neighborhood sums
        if n == 0 {
            // Special case: in iteration 0 the source is the 8-bit 'src'
            let mut s_offs = 0usize; 
            let mut d_offs = 0usize;

            for _ in 0..height {
                box_blur_pass(&src[range!(s_offs, width as usize)],
                              &mut dest_array[range!(d_offs, width as usize)],
                              box_radius, width as usize, 1);
                s_offs += src_line_stride;
                d_offs += width as usize;
            }
        } else {
            let mut offs = 0usize;

            for _ in 0..height {
                box_blur_pass(&src_array[range!(offs, width as usize)],
                              &mut dest_array[range!(offs, width as usize)],
                              box_radius, width as usize, 1);
                offs += width as usize;
            }
        }

        swap(&mut src_array, &mut dest_array);

        // Calculate vertical neighborhood sums
        for offs in 0..width as usize {
            box_blur_pass(&src_array[offs..],
                          &mut dest_array[offs..],
                          box_radius, height as usize, width as usize);
        }

        // Divide to obtain normalized result. We choose not to divide just once
        // after completing all iterations, because the 32-bit intermediate values
        // would overflow in as little as 3 iterations with 8-pixel box radius
        // for an all-white input image. In such case the final sums would be:
        //
        //   255 * ((2*8+1)^2)^3 = 6'155'080'095
        //
        // (where the exponent 3 = number of iterations)

        for i in dest_array.iter_mut() {
            *i /= divisor;
        }
    }

    // 'dest_array' is where the last summation results were stored,
    // now use it as source for producing the final 8-bit image in 'blurred'

    for i in 0 .. (width*height) as usize {
        blurred[i] = dest_array[i] as u8;
    }
}


/// Estimates quality of the specified area (8 bits per pixel).
///
/// Quality is the sum of differences between input image and its blurred version.
/// In other words, sum of values of the high-frequency component.
/// The sum is normalized by dividing by the number of pixels.
///
/// `pixels` starts at the beginning of a `width`x`height` area
/// in an image with `line_stride` distance between lines.
///
pub fn estimate_quality(pixels: &[u8], width: u32, height: u32, line_stride: usize, box_blur_radius: u32) -> f32 {
    let mut blurred = utils::alloc_uninitialized::<u8>((width*height) as usize);
    box_blur(pixels, &mut blurred[..], width, height, line_stride, box_blur_radius, QUALITY_ESTIMATE_BOX_BLUR_ITERATIONS);

    let mut quality = 0.0;

    let mut src_offs = 0usize;
    let mut blur_offs = 0usize;
    for _ in 0..height {
        for x in 0..width {
            quality += i32::abs(pixels[src_offs + x as usize] as i32 - blurred[blur_offs + x as usize] as i32) as f32;
        }
        src_offs += line_stride;
        blur_offs += width as usize;
    }

    quality / ((width*height) as f32)
}


/// Finds `remove_val` in the sorted `array`, replaces it with `new_val` and ensures `array` remains sorted. 
fn shift_sorted_window(array: &mut [f64], remove_val: f64, new_val: f64) {
    // Locate 'remove_val' in 'array'
    let mut curr_idx = array.binary_search_by(|x| x.partial_cmp(&remove_val).unwrap()).unwrap();
    
    // Insert 'new_val' into 'array' and (if needed) move it to keep the array sorted
    array[curr_idx] = new_val;
    while curr_idx <= array.len() - 2 && array[curr_idx] > array[curr_idx + 1] {
        array.swap(curr_idx, curr_idx + 1);
        curr_idx += 1;
    }
    while curr_idx > 0 && array[curr_idx] < array[curr_idx - 1] {
        array.swap(curr_idx - 1, curr_idx);
        curr_idx -= 1;
    }
}

/// Returns contents of `array` after median filtering.
pub fn median_filter(array: &[f64], window_radius: usize) -> Vec<f64> {
    assert!(window_radius < array.len());

    let wnd_len = 2 * window_radius + 1;
    
    // A sorted array
    let mut window = utils::alloc_uninitialized::<f64>(wnd_len);

    // Set initial window contents
    for i in 0..window_radius+1 {
        // upper half
        window[window_radius + i] = array[i];

        // lower half
        if i < window_radius {
            window[i] = array[0];
        }
    }
    
    window.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let mut output = utils::alloc_uninitialized::<f64>(array.len());

    // Replace every 'array' element in 'output' with window's median and shift the window
    for i in 0..array.len() {
        output[i] = window[window_radius];
        shift_sorted_window(&mut window,
                            array[max(i as isize - window_radius as isize, 0) as usize],
                            array[min(i + 1 + window_radius, array.len() - 1)]);
    }

    output
}
