#![crate_type = "lib"]
#![crate_name = "skry"]

#[macro_use]
mod utils;

pub mod defs;
pub mod filters;
pub mod image;
pub mod img_align;
pub mod img_seq;
pub mod quality;
pub mod ref_pt_align;
pub mod triangulation;
pub mod stacking;

mod avi;
mod blk_match;
mod bmp;
mod img_list;
mod img_seq_priv;
mod ser;
mod tiff;