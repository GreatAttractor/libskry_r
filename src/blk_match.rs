// 
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
// 
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Block matching.
//

use defs::{Point, Rect};
use image::{Image, PixelFormat};


const MIN_FRACTION_OF_BLOCK_TO_MATCH: u32 = 4;


/// Returns the sum of squared differences between pixels of `img` and `ref_block`.
///
/// `ref_block`'s center is aligned on `pos` over `img`. The differences are
/// calculated only for the `refblk_rect` portion of `ref_block`.
/// `pos` is relative to `img`; `refblk_rect` is relative to `ref_blk`.
/// 
/// Both `ref_block` and `img` must be `Mono8`. The result is 64-bit, so for
/// 8-bit images it can accommodate a block of 2^(64-2*8) = 2^48 pixels.
///
pub fn calc_sum_of_squared_diffs(img: &Image, ref_block: &Image, pos: &Point, refblk_rect: &Rect) -> u64 {
    assert!(img.get_pixel_format() == PixelFormat::Mono8);
    assert!(ref_block.get_pixel_format() == PixelFormat::Mono8);
    
    assert!(ref_block.get_img_rect().contains_rect(refblk_rect));
    
    let mut result = 0u64;

    let blk_width = ref_block.get_width();
    let blk_height = ref_block.get_height();

    // Example: an 8x6 block, * = pos, . - block's pixels,
    //          refblk_rect = { x: 2, y: 1, w: 5, h: 3 }
    //
    // 
    //         0           pos.x
    //        +----------------------
    //       0|            |
    //        |            | 
    //        |       +--------+ 
    //        |       |........|
    //        |       |..#####.|
    //        |       |..#####.|
    //   pos.y|-------|..##*##.|
    //        |       |........|
    //        |       |........|
    //        |       +--------+
    //        |
    //

    // Loop bounds

    let xstart = pos.x - (blk_width as i32)/2  + refblk_rect.x;
    let ystart = pos.y - (blk_height as i32)/2 + refblk_rect.y;
    
    let xend = xstart + refblk_rect.width as i32;
    let yend = ystart + refblk_rect.height as i32;    

    assert!(xstart >= 0);
    assert!(ystart >= 0);
    
    assert!(xend <= img.get_width() as i32);
    assert!(yend <= img.get_height() as i32);

    // Byte offsets in the pixel arrays
    
    let img_pix = img.get_mono8_pixels_from(Point{ x: xstart, y: ystart});
    let rblk_pix = ref_block.get_mono8_pixels_from(Point{x: refblk_rect.x, y: refblk_rect.y});
    
    let img_stride = img.get_width() as usize;
    let blk_stride = ref_block.get_width() as usize;
    
    let mut img_offs = 0;
    let mut blk_offs = 0;

    for _ in ystart..yend {
        for x in 0..(xend-xstart) {
            unsafe {
                result += sqr!(*img_pix.get_unchecked(img_offs + x as usize) as i32 -
                               *rblk_pix.get_unchecked(blk_offs + x as usize) as i32) as u64;
            }
        }
        img_offs += img_stride;
        blk_offs += blk_stride; 
    }
    
    result
}


/// TODO: add comments
pub fn find_matching_position(ref_pos: Point,
                              ref_block: &Image,
                              image: &Image,
                              search_radius: u32,
                              initial_search_step: u32) -> Point {
    assert!(image.get_pixel_format() == PixelFormat::Mono8);
    assert!(ref_block.get_pixel_format() == PixelFormat::Mono8);

    let blkw = ref_block.get_width();
    let blkh = ref_block.get_height();
    let imgw = image.get_width();
    let imgh = image.get_height();

    // At first, use a coarse step when trying to match `ref_block`
    // with `img` at different positions. Once an approximate matching
    // position is determined, the search continues around it repeatedly
    // using a smaller step, until the step becomes 1.
    let mut search_step = initial_search_step;

    // Range of positions where `ref_block` will be match-tested with `img`.
    // Using signed type is necessary, as the positions may be negative
    // (then the block is appropriately clipped before comparison).
    
    struct SearchRange {
        // Inclusive
        pub xmin: i32,
        pub ymin: i32,
        
        // Exclusive
        pub xmax: i32,
        pub ymax: i32
    };
    
    let mut search_range = SearchRange{ xmin: ref_pos.x - search_radius as i32,
                                        ymin: ref_pos.y - search_radius as i32,
                                        xmax: ref_pos.x + search_radius as i32,
                                        ymax: ref_pos.y + search_radius as i32 };

    let mut best_pos = Point{ x: 0, y: 0 };

    while search_step > 0 {
        // Min. sum of squared differences between pixel values of
        // the reference block and the image at candidate positions
        let mut min_sq_diff_sum = u64::max_value();
        
        // (x, y) = position in `img` for which a block match test is performed
        let mut y = search_range.ymin;
        while y < search_range.ymax {
            let mut x = search_range.xmin;
            while x < search_range.xmax {

                // It is allowed for `ref_block` to not be entirely inside `image`.
                // Before calling `calc_sum_of_squared_diffs()`, find a sub-rectangle
                // `refblk_rect` of `ref_block` which lies within `image`:
                // 
                //   +======== ref_block ========+
                //   |                           |
                //   |   +-------- img ----------|-------
                //   |   |.......................|
                //   |   |..........*............|
                //   |   |.......................|
                //   |   |.......................|
                //   +===========================+
                //       |
                //       |
                //   
                //   *: current search position (x, y); corresponds with the middle
                //      of 'ref_block' during block matching
                // 
                // Dotted area is the 'refblk_rect'. Start coordinates of 'refblk_rect'
                // are relative to the 'ref_block'; if whole 'ref_block' fits in 'image',
                // then refblk_rect = {0, 0, blkw, blkh}.


                let refblk_rect_x = if x >= (blkw as i32)/2 { 0 } else { (blkw as i32)/2 - x };
                let refblk_rect_y = if y >= (blkh as i32)/2 { 0 } else { (blkh as i32)/2 - y };  

                let refblk_rect_xmax: i32 =
                    if x + (blkw as i32)/2 <= imgw as i32 { blkw as i32 } else { blkw as i32 - (x + (blkw as i32)/2 - imgw as i32) };
                let refblk_rect_ymax: i32 =
                    if y + (blkh as i32)/2 <= imgh as i32 { blkh as i32 } else { blkh as i32 - (y + (blkh as i32)/2 - imgh as i32) }; 

                let mut sum_sq_diffs: u64;

                if refblk_rect_x >= refblk_rect_xmax ||
                   refblk_rect_y >= refblk_rect_ymax {
                       
                    // Ref. block completely outside image
                   sum_sq_diffs = u64::max_value();
                } else {
                    let refblk_rect = Rect{ x: refblk_rect_x, y: refblk_rect_y, 
                                            width: (refblk_rect_xmax - refblk_rect_x) as u32,
                                            height: (refblk_rect_ymax - refblk_rect_y) as u32};
                    
                    if refblk_rect.width < blkw / MIN_FRACTION_OF_BLOCK_TO_MATCH ||
                       refblk_rect.height < blkh / MIN_FRACTION_OF_BLOCK_TO_MATCH {
                           
                        // Ref. block too small to compare
                        sum_sq_diffs = u64::max_value();
                    } else {
                        sum_sq_diffs = calc_sum_of_squared_diffs(&image, &ref_block, &Point{ x, y }, &refblk_rect);

                        // The sum must be normalized in order to be comparable with others
                        sum_sq_diffs *= ((blkw as u32) * (blkh as u32)) as u64;
                        sum_sq_diffs /= (refblk_rect.width * refblk_rect.height) as u64;
                    }
                }

                if sum_sq_diffs < min_sq_diff_sum {
                    min_sq_diff_sum = sum_sq_diffs;
                    best_pos = Point{ x, y };
                }

                x += search_step as i32;
            }
            y += search_step as i32;
        }

        search_range.xmin = best_pos.x - search_step as i32;
        search_range.ymin = best_pos.y - search_step as i32;
        search_range.xmax = best_pos.x + search_step as i32;
        search_range.ymax = best_pos.y + search_step as i32;
        
        search_step /= 2;
    }

    best_pos
}
