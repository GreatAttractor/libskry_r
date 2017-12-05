// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Image provider: list of image files.
//

use image::{FileType, Image, ImageError, Palette, PixelFormat};
use img_seq_priv::{ImageProvider};


pub struct ImageList {
    file_names: Vec<String>
}


impl ImageList {
    pub fn new(file_names: &[&str]) -> Box<ImageProvider> {
        Box::new(
            ImageList {
                file_names: {
                    let mut v = vec![];
                    for fname in file_names {
                        v.push(fname.to_string())
                    }
                    
                    v
                }
            }
        )
    }    
}


impl ImageProvider for ImageList {
    fn get_img(&mut self, idx: usize) -> Result<Image, ImageError> {
        Image::load(&self.file_names[idx], FileType::Auto)
    }
    
    
    fn get_img_metadata(&self, idx: usize) -> Result<(u32, u32, PixelFormat, Option<Palette>), ImageError> {
        Image::get_metadata(&self.file_names[idx], FileType::Auto)
    }
    

    fn img_count(&self) -> usize {
        self.file_names.len()
    }
    
    
    fn deactivate(&mut self) {
        // Do nothing
    }
}