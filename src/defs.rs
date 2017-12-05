// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Common definitions.
//

use image;
use std;
use std::ops::{Add, AddAssign, Sub};


pub const WHITE_8BIT: u8 = 0xFF;


pub struct PointFlt {
    pub x: f32,
    pub y: f32
}


impl std::fmt::Display for PointFlt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "({:0.1}, {:0.1})", self.x, self.y)
    }
}


#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32
}


impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}


impl Sub for Point {
    type Output = Point;

    fn sub(self, other: Point) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}


impl AddAssign for Point {
    fn add_assign(&mut self, other: Point) {
        *self = Point {
            x: self.x + other.x,
            y: self.y + other.y,
        };
    }
}


impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "({}, {})", self.x, self.y)
    }
}


impl Point {
    pub fn sqr_dist(p1: &Point, p2: &Point) -> i32 {
        sqr!(p1.x - p2.x) + sqr!(p1.y - p2.y)
    }
}


#[derive(Copy, Clone, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32
}


impl Rect {
    pub fn contains_point(&self, p: &Point) -> bool {
        p.x >= self.x && p.x < self.x + self.width as i32 && p.y >= self.y && p.y < self.y + self.height as i32
    }
    
    
    pub fn contains_rect(&self, other: &Rect) -> bool {
        self.contains_point(&Point{ x: other.x, y: other.y }) &&
        self.contains_point(&Point{ x: other.x + other.width as i32 - 1,
                                    y: other.y + other.height as i32 - 1 })
    }
    
    
    pub fn get_pos(&self) -> Point { Point{ x: self.x, y: self.y } }
}


#[derive(Debug)]
pub enum ProcessingError {
    /// There are no more steps in the current processing phase.
    NoMoreSteps,
    
    ImageError(image::ImageError)
}


impl From<image::ImageError> for ProcessingError {
    fn from(err: image::ImageError) -> ProcessingError { ProcessingError::ImageError(err) }
}


/// Represents a single processing phase.
pub trait ProcessingPhase {
    /// Executes one processing step.
    fn step(&mut self) -> Result<(), ProcessingError>;
    

    /// Returns a copy of the image that was processed by the last call to `step()`.
    ///
    /// Can be used to show processing visualization.
    ///   
    fn get_curr_img(&mut self) -> Result<image::Image, image::ImageError>;
}