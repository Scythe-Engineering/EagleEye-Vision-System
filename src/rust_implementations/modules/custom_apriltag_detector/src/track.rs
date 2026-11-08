//! Re-find an already tracked tag from predicted corners, without searching the image.

use crate::decode::{decode, decode_known, TRACK_FIT};
use crate::detection::Detection;
use crate::families::IndexedFamily;
use crate::frame::Frame;
use crate::geometry::{refine, Point, Quad};
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

/// Convert eight pupil_apriltags-order pixel-edge corner values into a decode-order quad.
pub(crate) fn decode_quad(corners: &[f64; 8]) -> Quad {
    // Output order traverses unit corners (0,1),(1,1),(1,0),(0,0); decode expects the reverse.
    let mut quad = [Point::default(); 4];
    for (i, point) in quad.iter_mut().enumerate() {
        let k = 3 - i;
        point.x = corners[2 * k] - 0.5;
        point.y = corners[2 * k + 1] - 0.5;
    }
    quad
}

/// Re-find `tag_id` in a window of `frame` around its predicted corners.
///
/// `predicted` holds full-frame OpenCV pixel-centre corners in pupil_apriltags
/// order; so does the result. The window covers the white quiet zone and the edge
/// snap retry. Returns None for a window under 8 pixels or an unconfirmed tag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn track_in_frame(
    frame: &Frame<'_>,
    predicted: &[f64; 8],
    indexed: &IndexedFamily,
    family_index: u32,
    sharpening: f64,
    tag_id: i32,
    scratch: &mut Vec<u8>,
) -> Option<[f64; 8]> {
    if predicted.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (dx, dy) = (predicted[0] - predicted[4], predicted[1] - predicted[5]);
    // Unfused, like the Python tracker's np.linalg.norm that this replaces.
    let edge = (dx * dx + dy * dy).sqrt() / std::f64::consts::SQRT_2;
    let margin = edge / 8.0 * 1.5 + 8.0;
    let axis = |offset: usize| {
        let values = [0, 2, 4, 6].map(|i| predicted[i + offset]);
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let limit = if offset == 0 {
            frame.width
        } else {
            frame.height
        };
        let start = (low - margin).floor().max(0.0) as usize;
        let end = ((high + margin).ceil() + 1.0).clamp(0.0, limit as f64) as usize;
        (start, end)
    };
    let ((x0, x1), (y0, y1)) = (axis(0), axis(1));
    if x1 < x0 + 8 || y1 < y0 + 8 {
        return None;
    }
    let image = frame.gray_window((x0, y0, x1, y1), scratch).ok()?;
    let offset = [x0 as f64, y0 as f64];
    let prior: [f64; 8] = std::array::from_fn(|i| predicted[i] - offset[i % 2] + 0.5);
    let found = track(
        &decode_quad(&prior),
        &image,
        indexed,
        family_index,
        sharpening,
        tag_id,
    )?;
    Some(std::array::from_fn(|i| {
        found.corners[i] - 0.5 + offset[i % 2]
    }))
}
