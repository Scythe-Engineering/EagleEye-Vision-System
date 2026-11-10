use std::cmp::Ordering;

use crate::candidates::{
    prepare, proposals, window_proposals, Proposal, Scratch, FRAGMENT_FRACTION,
};
use crate::decode::{decode, plausible};
use crate::detection::Detection;
use crate::families::{select_family, IndexedFamily};
use crate::geometry::{max_f64, Homography, Quad};
use crate::image::Image;
use crate::workers::Workers;

#[derive(Clone, Copy)]
pub(crate) struct Settings {
    pub(crate) nthreads: usize,
    pub(crate) refine_edges: bool,
    pub(crate) quad_decimate: f64,
    pub(crate) quad_sigma: f64,
    pub(crate) decode_sharpening: f64,
}

impl Settings {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !(1..=64).contains(&self.nthreads)
            || !self.quad_decimate.is_finite()
            || !(1.0..=64.0).contains(&self.quad_decimate)
            || !self.quad_sigma.is_finite()
            || self.quad_sigma.abs() > 16.0
            || !self.decode_sharpening.is_finite()
            || self.decode_sharpening.abs() > 16.0
        {
            return Err("invalid detector configuration".into());
        }
        Ok(())
    }
}

pub(crate) struct Detector {
    pub(crate) settings: Settings,
    pub(crate) family: IndexedFamily,
    pub(crate) results: Vec<Detection>,
    pub(crate) workers: Workers,
    scratch: Scratch,
    row_spans: Vec<(usize, usize)>,
    /// Grown bounding boxes of primary proposals that did not decode this frame.
    failed: Vec<[f64; 4]>,
    /// Windows around small primary proposals that passed the plausibility screen
    /// but did not decode: likely pieces of a tag whose blurred border split.
    fragments: Vec<[f64; 4]>,
}

/// How far one proposal got through refinement and decoding.
#[derive(PartialEq)]
enum Outcome {
    Implausible,
    Failed,
    Decoded,
}

/// Largest failed quad, in pixels, treated as a possible piece of a split tag.
/// Splitting happens where the border is about one decimated pixel wide.
const FRAGMENT_MAX_EXTENT: f64 = 48.0;
/// Growth of a fragment window: a piece can be half its tag, so the rest of
/// the tag lies within half the piece's extent.
const FRAGMENT_GROWTH: f64 = 0.5;

impl Detector {
    pub(crate) fn new(family: &str, settings: Settings) -> Result<Self, String> {
        settings.validate()?;
        let family = select_family(family)?;
        let workers = Workers::new(settings.nthreads)?;
        Ok(Self {
            settings,
            family,
            results: Vec::new(),
            workers,
            scratch: Scratch::new(),
            row_spans: Vec::new(),
            failed: Vec::new(),
            fragments: Vec::new(),
        })
    }

    pub(crate) fn detect(&mut self, mut image: Image<'_>) -> &[Detection] {
        self.results.clear();
        if image.fully_observed() {
            image.source_map = None;
        }
        // Partially observed crops: precompute valid pixel spans once instead of
        // projecting every resampled pixel and bilinear neighbour through the map.
        let mut spans = std::mem::take(&mut self.row_spans);
        let image = if image.compute_row_spans(&mut spans) {
            Image {
                row_spans: Some(&spans),
                ..image
            }
        } else {
            image
        };
        self.detect_prepared(image);
        self.row_spans = spans;
        &self.results
    }

    fn detect_prepared(&mut self, image: Image<'_>) {
        self.failed.clear();

        self.fragments.clear();
        let light = self.family.family.reversed;
        let prepared = prepare(
            &image,
            self.settings.quad_decimate,
            self.settings.quad_sigma,
            &self.workers,
            &mut self.scratch,
        );
        let primary = proposals(
            &prepared,
            &mut self.scratch,
            &self.workers,
            light,
            false,
            &[],
        );
        self.decode_proposals(&image, &primary, false);
        // Recovery runs every frame so one decoded tag cannot hide the rest, but only
        // around primary quads that failed to decode, reusing the primary tile ranges.
        let recovery = proposals(
            &prepared,
            &mut self.scratch,
            &self.workers,
            light,
            true,
            &self.failed,
        );
        self.decode_proposals(&image, &recovery, false);
        // Fragment pass: a higher cut around likely tag pieces reconnects borders
        // that blur split, segmenting only small windows.
        let results = &self.results;
        self.fragments.retain(|region| {
            !results
                .iter()
                .any(|old| inside(region, old.center[0] - 0.5, old.center[1] - 0.5))
        });
        if !self.fragments.is_empty() {
            let found = window_proposals(
                &prepared,
                &mut self.scratch,
                &self.fragments,
                FRAGMENT_FRACTION,
            );
            self.decode_proposals(&image, &found, true);
        }
        self.results.sort_unstable_by(|a, b| {
            a.tag_id
                .cmp(&b.tag_id)
                .then_with(|| {
                    a.center[1]
                        .partial_cmp(&b.center[1])
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| {
                    a.center[0]
                        .partial_cmp(&b.center[0])
                        .unwrap_or(Ordering::Equal)
                })
        });
    }

    /// Whether a proposal sits on a tag already decoded this frame, using the same
    /// centre-distance rule as duplicate suppression.
    fn covers_decoded(&self, quad: &Quad) -> bool {
        let cx = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) * 0.25 + 0.5;
        let cy = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) * 0.25 + 0.5;
        self.results.iter().any(|old| {
            let size = (old.corners[0] - old.corners[4]).hypot(old.corners[1] - old.corners[5]);
            (old.center[0] - cx).hypot(old.center[1] - cy) < max_f64(2.0, size * 0.2)
        })
    }

    /// Decode proposals. Recovery proposals must centre inside a failed region:
    /// a fragment window when `fragment`, otherwise a failed primary quad.
    fn decode_proposals(&mut self, image: &Image<'_>, found: &[Proposal], fragment: bool) {
        for proposal in found {
            if proposal.recovery {
                let regions = if fragment {
                    &self.fragments
                } else {
                    &self.failed
                };
                let (cx, cy) = centre(&proposal.quad);
                if self.covers_decoded(&proposal.quad)
                    || !regions.iter().any(|region| inside(region, cx, cy))
                {
                    continue;
                }
            }
            let outcome = self.decode_one(image, proposal);
            if outcome == Outcome::Decoded || proposal.recovery {
                continue;
            }
            self.failed.push(grown_bounds(&proposal.quad, 0.25));
            // The fragment pass thresholds dark pixels only, like the 0.3 recovery.
            if outcome == Outcome::Failed
                && proposal.color == 0
                && extent(&proposal.quad) <= FRAGMENT_MAX_EXTENT
            {
                self.fragments
                    .push(grown_bounds(&proposal.quad, FRAGMENT_GROWTH));
            }
        }
    }

    /// Refine and decode one proposal.
    fn decode_one(&mut self, image: &Image<'_>, proposal: &Proposal) -> Outcome {
        let mut quad = proposal.quad;
        if i32::from(self.family.family.reversed) != proposal.color
            || !plausible(&quad, image, &self.family)
        {
            return Outcome::Implausible;
        }
        if self.settings.refine_edges
            && !crate::geometry::refine(
                &mut quad,
                image,
                self.settings.quad_decimate + 1.0,
                if proposal.color != 0 { -1 } else { 1 },
            )
        {
            return Outcome::Failed;
        }
        let Some(detection) = decode(
            &quad,
            image,
            &self.family,
            self.settings.decode_sharpening,
            self.settings.refine_edges,
        ) else {
            return Outcome::Failed;
        };
        let size = (detection.corners[0] - detection.corners[2])
            .hypot(detection.corners[1] - detection.corners[3]);
        let duplicate = self.results.iter_mut().find(|old| {
            (old.center[0] - detection.center[0]).hypot(old.center[1] - detection.center[1])
                < max_f64(2.0, size * 0.2)
        });
        match duplicate {
            Some(old) => {
                if !proposal.recovery
                    && (detection.hamming < old.hamming
                        || (detection.hamming == old.hamming
                            && detection.decision_margin > old.decision_margin))
                {
                    *old = detection;
                }
            }
            None => self.results.push(detection),
        }
        Outcome::Decoded
    }
}

fn centre(quad: &Quad) -> (f64, f64) {
    (
        (quad[0].x + quad[1].x + quad[2].x + quad[3].x) * 0.25,
        (quad[0].y + quad[1].y + quad[2].y + quad[3].y) * 0.25,
    )
}

fn inside(&[x0, y0, x1, y1]: &[f64; 4], x: f64, y: f64) -> bool {
    x >= x0 && x <= x1 && y >= y0 && y <= y1
}

/// Larger side of a quad's axis-aligned bounds.
fn extent(quad: &Quad) -> f64 {
    let [x0, y0, x1, y1] = grown_bounds(quad, 0.0);
    (x1 - x0).max(y1 - y0)
}

/// Axis-aligned bounds of a quad, grown on every side by `fraction` of its larger extent.
fn grown_bounds(quad: &Quad, fraction: f64) -> [f64; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for point in quad {
        x0 = x0.min(point.x);
        y0 = y0.min(point.y);
        x1 = x1.max(point.x);
        y1 = y1.max(point.y);
    }
    let pad = fraction * (x1 - x0).max(y1 - y0);
    [x0 - pad, y0 - pad, x1 + pad, y1 + pad]
}

pub(crate) fn image_span(width: usize, height: usize, stride: usize) -> Result<usize, String> {
    if !(2..=32768).contains(&width)
        || !(2..=32768).contains(&height)
        || stride < width
        || stride > i32::MAX as usize
        || width
            .checked_mul(height)
            .is_none_or(|area| area > 268435456)
        || stride.checked_mul(height).is_none()
    {
        return Err("invalid image dimensions, stride or output".into());
    }
    Ok((height - 1) * stride + width)
}

pub(crate) fn mapped_image<'a>(
    pixels: &'a [u8],
    width: usize,
    height: usize,
    stride: usize,
    source_map: Option<Homography>,
    source_shape: Option<(u32, u32)>,
) -> Result<Image<'a>, String> {
    if pixels.len() < image_span(width, height, stride)? {
        return Err("image buffer is shorter than its dimensions and stride".into());
    }
    let (source_height, source_width) = match (source_map, source_shape) {
        (Some(map), Some((height, width))) => {
            if height == 0 || width == 0 {
                return Err("source dimensions must be positive".into());
            }
            if map.iter().any(|value| !value.is_finite()) {
                return Err("source map must be finite".into());
            }
            let determinant = map[0] * (map[4] * map[8] - map[5] * map[7])
                - map[1] * (map[3] * map[8] - map[5] * map[6])
                + map[2] * (map[3] * map[7] - map[4] * map[6]);
            if !determinant.is_finite() || determinant == 0.0 {
                return Err("source map must be nonsingular".into());
            }
            (height, width)
        }
        (None, None) => (0, 0),
        _ => return Err("source dimensions require a map".into()),
    };
    Ok(Image {
        pixels,
        width,
        height,
        stride,
        source_map,
        source_width,
        source_height,
        row_spans: None,
    })
}
