//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Processing phase: image alignment (video stabilization).
//

use blk_match;
use defs::{Point, ProcessingPhase, ProcessingError, Rect, WHITE_8BIT};
use filters;
use img_seq;
use img_seq::ImageSequence;
use image;
use image::{DemosaicMethod, Image, PixelFormat};
use std::cmp::{max, min};
use utils;


pub struct AnchorConfig {
    /// Anchor points to use for alignment; if `None`, an anchor will be placed automatically.
    ///
    /// Coordinates are relative to the current image's origin.
    ///
    pub initial_anchors: Option<Vec<Point>>,

    /// Radius (in pixels) of anchors' reference blocks
    pub block_radius: u32,

    /// Images are aligned by matching blocks that are offset (horz. and vert.) by up to this radius (in pixels).
    pub search_radius: u32,

    /// Min. image brightness that an anchor can be placed at (values: [0; 1]).
    ///
    /// Value is relative to the image's darkest (0.0) and brightest (1.0) pixels.
    ///
    pub placement_brightness_threshold: f32
}

pub enum AlignmentMethod {
    /// Alignment via block-matching around the specified anchor points
    Anchors(AnchorConfig),

    /// Alignment using the image centroid
    Centroid
}


const QUALITY_EST_BOX_BLUR_RADIUS: u32 = 2;


struct AnchorData {
    /// Current position
    pub pos: Point,

    pub is_valid: bool,

    /// Square image fragment (of the best quality so far) centered (after alignment) on `pos`
    pub ref_block: Image,

    /// Quality of `ref_block`
    pub ref_block_qual: f32
}


/// Set-theoretic intersection of all images after alignment (i.e. the fragment which is visible in all images).
#[derive(Default)]
struct ImgIntersection {
    /// Offset, relative to the first image's origin.
    pub offset: Point,

    /// Coordinates of the bottom right corner (belongs to the intersection), relative to the first image's origin.
    pub bottom_right: Point,

    pub width: u32,
    pub height: u32
}


/// Contains results of processing performed by `ImgAlignmentProc`.
#[derive(Default)]
pub struct ImgAlignmentData {
    /// Images' intersection
    intersection: ImgIntersection,

    /// Image offsets (relative to each image's origin) necessary for them to be aligned.
    ///
    /// Concerns only active images of the sequence.
    ///
    img_offsets: Vec<Point>
}


impl ImgAlignmentData {
    /// Returns offset of images' intersection relative to the first image's origin.
    pub fn get_intersection(&self) -> Rect {
        Rect{ x: self.intersection.offset.x,
              y: self.intersection.offset.y,
              width: self.intersection.width,
              height: self.intersection.height }
    }


    /// Returns offsets (relative to each image's origin) required for images to be aligned.
    pub fn get_image_ofs(&self) -> &[Point] {
        &self.img_offsets[..]
    }
}

struct AnchorsState {
    anchors: Vec<AnchorData>,
    config: AnchorConfig,
    active_anchor_idx: usize,
}

enum State {
    Anchors(AnchorsState),
    /// Contains current centroid position.
    Centroid(Point)
}

/// Performs image alignment (video stabilization).
///
/// Completed alignment results in determining the images' intersection,
/// i.e. the common rectangular area visible in all frames. Due to the likely
/// image drift, this area is usually smaller than the smallest image in the sequence.
///
pub struct ImgAlignmentProc<'a> {
    data_returned: bool,

    is_complete: bool,

    img_seq: &'a mut ImageSequence,

    /// Current image index (within the active images' subset).
    curr_img_idx: usize,

    state: State,

    data: ImgAlignmentData
}


impl<'a> ImgAlignmentProc<'a> {
    /// Returns image alignment data determined during processing. May be called only once.
    pub fn get_data(&mut self) -> ImgAlignmentData {
        assert!(!self.data_returned && self.is_complete());
        self.data_returned = true;
        ::std::mem::replace(&mut self.data, ImgAlignmentData::default())
    }


    /// Initializes image alignment (video stabilization).
    ///
    /// If `align_method` is `AlignmentMethod::Centroid`, all subsequent parameters are ignored.
    /// If `anchors` is empty, anchors will be placed automatically.
    ///
    /// `block_radius`  - radius (in pixels) of square blocks used for matching images
    /// `search_radius` - max offset in pixels (horizontal and vertical) of blocks during matching
    /// `placement_brightness_threshold` - min. image brightness that an anchor can be placed at (values: [0; 1]);
    ///                                    value is relative to the image's darkest (0.0) and brightest (1.0) pixels
    ///
    pub fn init(
        img_seq: &'a mut ImageSequence,
        align_method: AlignmentMethod
    ) -> Result<ImgAlignmentProc<'a>, ProcessingError> {
        assert!(img_seq.get_active_img_count() > 0);

        img_seq.seek_start();
        let first_img = img_seq.get_curr_img()?;

        let img_offsets = Vec::<Point>::with_capacity(img_seq.get_active_img_count());

        let intersection = ImgIntersection{
            offset: Point{ x: 0, y: 0 },
            bottom_right: Point { x: i32::max_value(), y: i32::max_value() },
            width: 0,
            height: 0
        };

        let state;

        match align_method {
            AlignmentMethod::Anchors(anchor_cfg) => {
                assert!(anchor_cfg.block_radius > 0 && anchor_cfg.search_radius > 0);

                let mut anchor_data: Vec<AnchorData> = vec![];
                let mut anchor_positions: Vec<Point> = vec![];

                match &anchor_cfg.initial_anchors {
                    None => anchor_positions.push(ImgAlignmentProc::suggest_anchor_pos(
                        &first_img,
                        anchor_cfg.placement_brightness_threshold,
                        2 * anchor_cfg.block_radius
                    )),

                    Some(positions) => anchor_positions = positions.clone()
                }

                for anchor_pos in anchor_positions {
                    let ref_block = first_img.convert_pix_fmt_of_subimage(
                        PixelFormat::Mono8,
                        Point{
                            x: anchor_pos.x - anchor_cfg.block_radius as i32,
                            y: anchor_pos.y - anchor_cfg.block_radius as i32
                        },
                        2 * anchor_cfg.block_radius,
                        2 * anchor_cfg.block_radius,
                        Some(DemosaicMethod::Simple)
                    );

                    let ref_block_qual = filters::estimate_quality(ref_block.get_pixels(),
                                                                ref_block.get_width(),
                                                                ref_block.get_height(),
                                                                ref_block.get_width() as usize,
                                                                QUALITY_EST_BOX_BLUR_RADIUS);
                    anchor_data.push(
                        AnchorData{ pos: anchor_pos,
                                    is_valid: true,
                                    ref_block,
                                    ref_block_qual }
                    );
                }

                state = State::Anchors(
                    AnchorsState{ anchors: anchor_data, config: anchor_cfg, active_anchor_idx: 0}
                );
            },

            AlignmentMethod::Centroid => state = State::Centroid(first_img.get_centroid(first_img.get_img_rect()))
        }

        Ok(ImgAlignmentProc{
           data_returned: false,
            is_complete: false,
            img_seq,
            curr_img_idx: 0,
            state,
            data: ImgAlignmentData{ intersection, img_offsets }
        })
    }

    pub fn is_complete(&self) -> bool { self.is_complete }

    /// Returns the current number of anchors.
    ///
    /// The return value may increase during processing (when all existing
    /// anchors became invalid and a new one(s) had to be automatically created).
    ///
    pub fn get_anchor_count(&self) -> usize {
        match &self.state {
            State::Anchors(anchors) => anchors.anchors.len(),
            _ => panic!("Alignment mode is not anchors.")
        }
    }

    /// Returns current positions of anchor points.
    pub fn get_anchors(&self) -> Vec<Point> {
        match &self.state {
            State::Anchors(anchors) => anchors.anchors.iter().map(|ref a| a.pos).collect(),
            _ => panic!("Alignment mode is not anchors.")
        }
    }


    pub fn is_anchor_valid(&self, anchor_idx: usize) -> bool {
        match &self.state {
            State::Anchors(anchors) => anchors.anchors[anchor_idx].is_valid,
            _ => panic!("Alignment mode is not anchors.")
        }
    }


    /// Returns the optimal position of a video stabilization anchor in `image`.
    ///
    /// `placement_brightness_threshold` - min. image brightness that an anchor point can be placed at;
    ///                                    value is relative to the image's darkest (0.0) and brightest (1.0) pixels
    ///
    fn suggest_anchor_pos(image: &Image, placement_brightness_threshold: f32, ref_block_size: u32) -> Point {
        let width = image.get_width();
        let height = image.get_height();

        let img8_storage: Box<Image>;
        let mut img8: &Image = image;

        if image.get_pixel_format() != PixelFormat::Mono8 {
            img8_storage = Box::new(image.convert_pix_fmt(PixelFormat::Mono8, Some(DemosaicMethod::Simple)));
            img8 = &img8_storage;
        }

        let (bmin, bmax) = utils::find_min_max_brightness(&img8);

        let mut result = Point{ x: (width / 2) as i32, y: (height / 2) as i32 };
        let mut best_qual = 0.0;

        let num_pixels_in_block = sqr!(ref_block_size) as usize;

        // Consider only the middle 3/4 of `image`
        let mut y = height/8 + ref_block_size/2;
        while y < 7*height/8 - ref_block_size/2 {
            let mut x = width/8 + ref_block_size/2;
            while x < 7*width/8 - ref_block_size {
                let mut num_above_thresh = 0usize;
                let mut num_white = 0usize;

                for ny in range!(y - ref_block_size/2, ref_block_size) {
                    let line = img8.get_line_raw(ny);

                    for nx in range!(x - ref_block_size/2, ref_block_size) {
                        if line[nx as usize] == WHITE_8BIT { num_white += 1; }

                        if line[nx as usize] != WHITE_8BIT && line[nx as usize] >= bmin + (placement_brightness_threshold * (bmax - bmin) as f32) as u8 {
                            num_above_thresh += 1;
                        }
                    }
                }

                if num_above_thresh > num_pixels_in_block / 5 &&
                    // Reject locations at the limb of an overexposed (fully white) disc; the white pixels
                    // would weigh heavily during block matching and the point would tend to jump along the limb
                    num_white < num_pixels_in_block/10 &&
                    utils::assess_gradients_for_block_matching(
                        img8,
                        Point{ x: x as i32, y: y as i32},
                        max(ref_block_size/2, 32)) {

                    let qual = filters::estimate_quality(img8.get_mono8_pixels_from(Point{ x: (x - ref_block_size/2) as i32,
                                                                                           y: (y - ref_block_size/2) as i32 }),
                                                         ref_block_size, ref_block_size, img8.get_width() as usize, 4);

                    if qual > best_qual {
                        best_qual = qual;
                        result = Point{ x: x as i32, y: y as i32 };
                    }
                }

                x += ref_block_size/3;
            }

            y += ref_block_size/3;
        }

        result
    }

    /// Returns the current centroid position
    pub fn get_current_centroid_pos(&self) -> Point {
        match &self.state {
            State::Centroid(centroid) => *centroid,
            _ => panic!("Alignment mode is not centroid.")
        }
    }
}

fn determine_img_offset_using_centroid(current_centroid_pos: Point, img: &Image) -> Point {
    let new_centroid_pos = img.get_centroid(img.get_img_rect());

    Point{
        x: new_centroid_pos.x - current_centroid_pos.x,
        y: new_centroid_pos.y - current_centroid_pos.y
    }
}

fn determine_img_offset_using_anchors(state: &mut AnchorsState, img: &Image) -> Point {
    assert!(img.get_pixel_format() == PixelFormat::Mono8);

    let mut active_anchor_offset = Point::default();

    for (i, anchor) in state.anchors.iter_mut().enumerate() {
        if anchor.is_valid {
            let s_rad = state.config.search_radius;

            let new_pos = blk_match::find_matching_position(
                anchor.pos,
                &anchor.ref_block,
                &img,
                s_rad,
                4
            );

            let blkw = anchor.ref_block.get_width();
            let blkh = anchor.ref_block.get_height();

            if new_pos.x < (blkw + s_rad) as i32 ||
               new_pos.x > (img.get_width() - blkw - s_rad) as i32 ||
               new_pos.y < (blkh + s_rad) as i32 ||
               new_pos.y > (img.get_height() - blkh - s_rad) as i32 {

                anchor.is_valid = false;
                continue;
            }

            let new_qual = filters::estimate_quality(img.get_mono8_pixels_from(new_pos),
                                                     blkw, blkh, img.get_width() as usize, QUALITY_EST_BOX_BLUR_RADIUS);

            if new_qual > anchor.ref_block_qual {
                anchor.ref_block_qual = new_qual;

                // Refresh the reference block using the current image at the block's new position
                img.convert_pix_fmt_of_subimage_into(&mut anchor.ref_block,
                                                     Point{ x: new_pos.x - (blkw/2) as i32, y: new_pos.y - (blkh/2) as i32 },
                                                     Point{ x: 0, y: 0 },
                                                     blkw, blkh,
                                                     Some(DemosaicMethod::Simple));
            }

            if i == state.active_anchor_idx {
                active_anchor_offset.x = new_pos.x - anchor.pos.x;
                active_anchor_offset.y = new_pos.y - anchor.pos.y;
            }

            anchor.pos = new_pos;
        }
    }

    if !state.anchors[state.active_anchor_idx].is_valid {
        // select the next available valid anchor as "active"
        let mut new_active_idx = state.active_anchor_idx + 1;

        while new_active_idx < state.anchors.len() {
            if state.anchors[new_active_idx].is_valid {
                break;
            } else {
                new_active_idx += 1;
            }
        }

        if new_active_idx >= state.anchors.len() {
            // there are no more existing valid anchors; choose and add a new one

            let new_pos = ImgAlignmentProc::suggest_anchor_pos(
                &img, state.config.placement_brightness_threshold, 2 * state.config.search_radius
            );


            let ref_block = img.get_fragment_copy(Point{ x: new_pos.x - state.config.block_radius as i32,
                                                         y: new_pos.y - state.config.block_radius as i32 },
                                                  2 * state.config.block_radius,
                                                  2 * state.config.block_radius,
                                                  false);
            let ref_block_qual = filters::estimate_quality(&ref_block.get_pixels(),
                                                              ref_block.get_width(),
                                                              ref_block.get_height(),
                                                              ref_block.get_width() as usize,
                                                              QUALITY_EST_BOX_BLUR_RADIUS);
            state.anchors.push(
                AnchorData{
                    pos: new_pos,
                    ref_block,
                    ref_block_qual,
                    is_valid: true,
                }
            );

            state.active_anchor_idx = state.anchors.len() - 1;
        }
    }

    active_anchor_offset
}



impl<'a> ProcessingPhase for ImgAlignmentProc<'a> {
    fn get_curr_img(&mut self) -> Result<image::Image, image::ImageError> {
        self.img_seq.get_curr_img()
    }


    fn step(&mut self) -> Result<(), ProcessingError> {
        if self.curr_img_idx == 0 {
            self.data.img_offsets.push(Point{ x: 0, y: 0 });

            let (width, height, _, _) = self.img_seq.get_curr_img_metadata()?;

            self.data.intersection.bottom_right.x = width as i32 - 1;
            self.data.intersection.bottom_right.y = height as i32 - 1;
            self.curr_img_idx += 1;

            Ok(())
        } else {
            match self.img_seq.seek_next() {
                Err(img_seq::SeekResult::NoMoreImages) => {
                    self.data.intersection.width = (self.data.intersection.bottom_right.x - self.data.intersection.offset.x + 1) as u32;
                    self.data.intersection.height = (self.data.intersection.bottom_right.y - self.data.intersection.offset.y + 1) as u32;

                    self.is_complete = true;

                    return Err(ProcessingError::NoMoreSteps);
                },

                _ => { }
            }

            let img = self.img_seq.get_curr_img()?;

            let detected_img_offset;

            match &mut self.state {
                State::Anchors(anchors) => {
                    let img8_storage: Box<Image>;
                    let mut img8 = &img;
                    if img.get_pixel_format() != PixelFormat::Mono8 {
                        img8_storage = Box::new(img.convert_pix_fmt(PixelFormat::Mono8, Some(DemosaicMethod::Simple)));
                        img8 = &img8_storage;
                    }

                    detected_img_offset = determine_img_offset_using_anchors(anchors, img8);
                },

                State::Centroid(centroid_pos) => {
                    detected_img_offset = determine_img_offset_using_centroid(*centroid_pos, &img);
                    *centroid_pos += detected_img_offset;
                }
            }

            // `img_offsets` contain offsets relative to the first frame, so store the current offset incrementally w.r.t. the previous one
            let new_ofs = *self.data.img_offsets.last().unwrap() + detected_img_offset;
            self.data.img_offsets.push(new_ofs);

            self.data.intersection.offset.x = max(self.data.intersection.offset.x, -new_ofs.x);
            self.data.intersection.offset.y = max(self.data.intersection.offset.y, -new_ofs.y);
            self.data.intersection.bottom_right.x = min(self.data.intersection.bottom_right.x, -new_ofs.x + img.get_width() as i32 - 1);
            self.data.intersection.bottom_right.y = min(self.data.intersection.bottom_right.y, -new_ofs.y + img.get_height() as i32 - 1);
            self.curr_img_idx += 1;

            Ok(())
        }
    }
}
