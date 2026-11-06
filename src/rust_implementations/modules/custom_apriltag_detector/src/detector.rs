use std::cmp::Ordering;

use crate::candidates::{prepare, proposals, Proposal, Scratch};
use crate::decode::decode;
use crate::detection::Detection;
use crate::families::{select_families, IndexedFamily};
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
    pub(crate) selected: Vec<IndexedFamily>,
    pub(crate) results: Vec<Detection>,
    workers: Workers,
    scratch: Scratch,
    row_spans: Vec<(usize, usize)>,
    /// Grown bounding boxes of primary proposals that did not decode this frame.
    failed: Vec<[f64; 4]>,
}

impl Detector {
    pub(crate) fn new(names: &str, settings: Settings) -> Result<Self, String> {
        settings.validate()?;
        let selected = select_families(names)?;
        let workers = Workers::new(settings.nthreads)?;
        Ok(Self {
            settings,
            selected,
            results: Vec::new(),
            workers,
            scratch: Scratch::new(),
            row_spans: Vec::new(),
            failed: Vec::new(),
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
        let light = self.selected.iter().any(|family| family.family.reversed);
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
        self.decode_proposals(&image, &primary);
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
        self.decode_proposals(&image, &recovery);
        self.results.sort_unstable_by(|a, b| {
            a.family_index
                .cmp(&b.family_index)
                .then_with(|| a.tag_id.cmp(&b.tag_id))
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

    /// Whether a recovery quad's centre lies inside a primary quad that failed to decode.
    fn inside_failed(&self, quad: &Quad) -> bool {
        let cx = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) * 0.25;
        let cy = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) * 0.25;
        self.failed
            .iter()
            .any(|&[x0, y0, x1, y1]| cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1)
    }

    fn decode_proposals(&mut self, image: &Image<'_>, found: &[Proposal]) {
        for proposal in found {
            if proposal.recovery
                && (self.covers_decoded(&proposal.quad) || !self.inside_failed(&proposal.quad))
            {
                continue;
            }
            if !self.decode_one(image, proposal) && !proposal.recovery {
                self.failed.push(grown_bounds(&proposal.quad, 0.25));
            }
        }
    }

    /// Refine and decode one proposal; returns whether any family decoded it.
    fn decode_one(&mut self, image: &Image<'_>, proposal: &Proposal) -> bool {
        let mut quad = proposal.quad;
        if self.settings.refine_edges
            && !crate::geometry::refine(
                &mut quad,
                image,
                self.settings.quad_decimate + 1.0,
                if proposal.color != 0 { -1 } else { 1 },
            )
        {
            return false;
        }
        let mut decoded = false;
        for (index, family) in self.selected.iter().enumerate() {
            if i32::from(family.family.reversed) != proposal.color {
                continue;
            }
            let Some(detection) = decode(
                &quad,
                image,
                family,
                index as u32,
                self.settings.decode_sharpening,
                self.settings.refine_edges,
            ) else {
                continue;
            };
            decoded = true;
            let mut duplicate = false;
            for old in &mut self.results {
                let distance = (old.center[0] - detection.center[0])
                    .hypot(old.center[1] - detection.center[1]);
                let size = (detection.corners[0] - detection.corners[2])
                    .hypot(detection.corners[1] - detection.corners[3]);
                if old.family_index == detection.family_index && distance < max_f64(2.0, size * 0.2)
                {
                    duplicate = true;
                    if !proposal.recovery
                        && (detection.hamming < old.hamming
                            || (detection.hamming == old.hamming
                                && detection.decision_margin > old.decision_margin))
                    {
                        *old = detection;
                    }
                    break;
                }
            }
            if !duplicate {
                self.results.push(detection);
            }
        }
        decoded
    }
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
