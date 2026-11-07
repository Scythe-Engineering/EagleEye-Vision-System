//! Re-find an already tracked tag from predicted corners, without searching the image.

use crate::decode::{decode, decode_known, TRACK_FIT};
use crate::detection::Detection;
use crate::families::IndexedFamily;
use crate::geometry::{refine, Quad};
use crate::image::Image;

/// Edge-snap search radius in pixels for predictions the grid fit cannot correct.
const RETRY_RADIUS: f64 = 3.0;

/// Confirm `tag_id` at a predicted quad and grid-fit its corners.
///
/// A prediction within about a quarter cell needs only the grid fit. Otherwise the
/// quad goes through the search path's edge snap, decode and grid fit. Returns None
/// when the tag is not confirmed, so callers can fall back to a search.
pub(crate) fn track(
    prior: &Quad,
    image: &Image<'_>,
    indexed: &IndexedFamily,
    family_index: u32,
    sharpening: f64,
    tag_id: i32,
) -> Option<Detection> {
    decode_known(
        prior,
        image,
        indexed,
        family_index,
        sharpening,
        tag_id,
        TRACK_FIT,
    )
    .or_else(|| {
        let mut quad = *prior;
        let polarity = if indexed.family.reversed { -1 } else { 1 };
        if !refine(&mut quad, image, RETRY_RADIUS, polarity) {
            return None;
        }
        decode(&quad, image, indexed, family_index, sharpening, true)
            .filter(|detection| detection.tag_id == tag_id)
    })
}
