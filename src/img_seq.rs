// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Image sequence.
//

use avi;
use image::{Image, ImageError, Palette, PixelFormat};
use img_list;
use img_seq_priv::ImageProvider;
use ser;


pub enum SeekResult {
    NoMoreImages
}


pub struct ImageSequence {
    img_provider: Box<ImageProvider>,
    
    active_flags: Vec<bool>,
    
    curr_img_idx: usize,
    
    curr_img_idx_within_active_subset: usize,
    
    num_active_imgs: usize,
    
    last_active_idx: usize,
    
    last_loaded_img_idx: usize,
    
    last_loaded_img: Option<Image>
}


impl ImageSequence {
    fn init(img_provider: Box<ImageProvider>) -> ImageSequence {
        let num_images = img_provider.img_count();
        
        ImageSequence {
            img_provider,
            active_flags: vec![true; num_images],
            curr_img_idx: 0,
            curr_img_idx_within_active_subset: 0,
            num_active_imgs: num_images,
            last_active_idx: num_images - 1,
            last_loaded_img_idx: 0,
            last_loaded_img: None
        }
    }
    
    pub fn new_image_list(file_names: &[&str]) -> ImageSequence {
        ImageSequence::init(img_list::ImageList::new(file_names))        
    }
    
    
    pub fn new_avi_video(file_name: &str) -> Result<ImageSequence, avi::AviError> {
        Ok(ImageSequence::init(try!(avi::AviFile::new(file_name))))
    } 


    pub fn new_ser_video(file_name: &str) -> Result<ImageSequence, ser::SerError> {
        Ok(ImageSequence::init(try!(ser::SerFile::new(file_name))))
    } 

    
    fn get_img(&mut self, idx: usize) -> Result<Image, ImageError> {
        if idx == self.last_loaded_img_idx {
            match &self.last_loaded_img {
                &Some(ref img) => return Ok(img.clone()),
                &None => ()
            }
        }
        
        self.last_loaded_img = Some(try!(self.img_provider.get_img(idx)));
        self.last_loaded_img_idx = idx;
        
        Ok(self.last_loaded_img.iter().next().unwrap().clone())
    }


    pub fn get_curr_img_idx(&self) -> usize {
        self.curr_img_idx
    }

    
    pub fn get_curr_img_idx_within_active_subset(&self) -> usize {
        self.curr_img_idx_within_active_subset
    }
    
    
    pub fn get_img_count(&self) -> usize {
        self.img_provider.img_count()
    }

    
    /// Seeks to the first active image
    pub fn seek_start(&mut self) {
        self.curr_img_idx = 0;
        while !self.active_flags[self.curr_img_idx] {
            self.curr_img_idx += 1;
        }
        
        self.curr_img_idx_within_active_subset = 0;
    }

    
    /// Seeks forward to the next active image
    pub fn seek_next(&mut self) -> Result<(), SeekResult> {
        if self.curr_img_idx < self.last_active_idx {
            while !self.active_flags[self.curr_img_idx] {
                self.curr_img_idx += 1;
            }
            self.curr_img_idx += 1;
            self.curr_img_idx_within_active_subset += 1;
            
            Ok(())
        } else {
            Err(SeekResult::NoMoreImages)
        }
    }
    

    pub fn get_curr_img(&mut self) -> Result<Image, ImageError> {
        let idx_to_load = self.curr_img_idx;
        self.get_img(idx_to_load)
    }
    
    
    /// Returns (width, height, pixel format, palette)
    pub fn get_curr_img_metadata(&mut self) -> Result<(u32, u32, PixelFormat, Option<Palette>), ImageError> {
        if self.curr_img_idx == self.last_loaded_img_idx {
            match &self.last_loaded_img {
                &Some(ref img) => return Ok((img.get_width(), img.get_height(), img.get_pixel_format(), img.get_palette().clone())),
                &None => ()
            }
        }

        self.img_provider.get_img_metadata(self.curr_img_idx)
    }
    
    
    pub fn get_img_by_index(&mut self, idx: usize) -> Result<Image, ImageError> {
        self.get_img(idx)
    }
    
    
    /// Should be called when `img_seq` will not be read for some time.
    ///
    /// In case of image lists, the function does nothing. For video files, it closes them.
    /// Video files are opened automatically (and kept open) every time a frame is loaded.
    ///
    pub fn deactivate(&mut self) {
        self.img_provider.deactivate()
    }
    
    
    /// Marks images as active. Element count of `is_active` must equal the number of images in the sequence.
    pub fn set_active_imgs(&mut self, is_active: &[bool]) {
        assert!(is_active.len() == self.active_flags.len());
        self.active_flags.clear();
        self.active_flags.extend_from_slice(is_active);
        
        self.num_active_imgs = 0;
        for i in 0 .. self.img_provider.img_count() {
            if self.active_flags[i] {
                self.last_active_idx = i;
                self.num_active_imgs += 1;
            }
        }
    }
    
    
    pub fn is_img_active(&self, img_idx: usize) -> bool {
        self.active_flags[img_idx]
    }
    
    
    /// Element count of the result equals the number of images in the sequence.
    pub fn get_img_active_flags(&self) -> &[bool] {
        &self.active_flags[..]
    }
    
    
    pub fn get_active_img_count(&self) -> usize {
        self.num_active_imgs
    }
    
    
    /// Translates index in the active images' subset into absolute index.
    pub fn get_absolute_img_idx(&self, active_img_idx: usize) -> usize {
        let mut abs_idx = 0usize;
        let mut active_img_counter = 0usize;
        while abs_idx < self.img_provider.img_count() {
            if active_img_counter == active_img_idx {
                break;
            }
            if self.active_flags[abs_idx] {
                active_img_counter += 1;
            }
            abs_idx += 1;
        }
    
        abs_idx
    }
}
