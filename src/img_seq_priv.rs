// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Image provider trait.
//

use image::{Image, ImageError, Palette, PixelFormat};


/// Image provider used by `ImageSequence`.
pub trait ImageProvider {
    fn img_count(&self) -> usize;
    
    fn get_img(&mut self, idx: usize) -> Result<Image, ImageError>;
    
    /// Returns width, height, pixel format, palette. 
    fn get_img_metadata(&self, idx: usize) -> Result<(u32, u32, PixelFormat, Option<Palette>), ImageError>;
        
    fn deactivate(&mut self);
}