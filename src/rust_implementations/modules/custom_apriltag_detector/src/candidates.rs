//! Area resampling, local thresholding, and ordered run-component proposals.

use crate::geometry::{clamp_f64, fit_quad, hull, max_f64, min_f64, Point, Quad};
use crate::image::Image;
use crate::workers::Workers;

pub(crate) struct Proposal {
    pub(crate) quad: Quad,
    pub(crate) color: i32,
    pub(crate) recovery: bool,
}

#[derive(Clone, Copy)]
struct Span {
    first: usize,
    offset: usize,
    count: usize,
}

struct AreaAxis {
    spans: Vec<Span>,
    weights: Vec<f64>,
}

impl AreaAxis {
    /// Cache each fractional cell's overlaps, in source-pixel order.
    fn new(size: usize, cells: usize, scale: f64) -> Self {
        let mut axis: Self = Self {
            spans: Vec::with_capacity(cells),
            weights: Vec::with_capacity(if cells == 0 { 0 } else { size + cells }),
        };
        for cell in 0..cells {
            let lo: f64 = cell as f64 * scale;
            let hi: f64 = (cell + 1) as f64 * scale;
            let first: usize = lo as usize;
            let end: usize = size.min(hi.ceil() as usize);
            axis.spans.push(Span {
                first,
                offset: axis.weights.len(),
                count: end - first,
            });
            for i in first..end {
                axis.weights
                    .push(min_f64(hi, (i + 1) as f64) - max_f64(lo, i as f64));
            }
        }
        axis
    }
}

/// Resample source pixel areas without changing floating accumulation order.
fn resample(image: &Image<'_>, w: usize, h: usize, workers: &Workers) -> Vec<f64> {
    let sx: f64 = image.width as f64 / w as f64;
    let sy: f64 = image.height as f64 / h as f64;
    let masked: bool = image.source_map.is_some();
    let cached: bool = !masked
        && !((sx == 1.0 && sy == 1.0) || (sx == 2.0 && sy == 2.0) || (sx == 3.0 && sy == 3.0));
    let ax: AreaAxis = AreaAxis::new(image.width, if cached { w } else { 0 }, sx);
    let ay: AreaAxis = AreaAxis::new(image.height, if cached { h } else { 0 }, sy);
    let mut gray: Vec<f64> = vec![0.0; w * h];
    workers.rows(
        &mut gray,
        w,
        w * h > 65536,
        |y: usize, output: &mut [f64]| {
            // Slice only visible pixels: the last source row need not have stride padding.
            if !masked && sx == 1.0 && sy == 1.0 {
                let row: usize = y * image.stride;
                let pixels: &[u8] = &image.pixels[row..row + image.width];
                for (value, pixel) in output.iter_mut().zip(pixels) {
                    *value = *pixel as f64;
                }
                return;
            }
            if !masked && sx == 2.0 && sy == 2.0 {
                let row: usize = 2 * y * image.stride;
                let top: &[u8] = &image.pixels[row..row + image.width];
                let row: usize = row + image.stride;
                let bottom: &[u8] = &image.pixels[row..row + image.width];
                for ((value, top), bottom) in output
                    .iter_mut()
                    .zip(top.as_chunks::<2>().0.iter())
                    .zip(bottom.as_chunks::<2>().0.iter())
                {
                    let sum: u32 =
                        top[0] as u32 + top[1] as u32 + bottom[0] as u32 + bottom[1] as u32;
                    *value = sum as f64 * 0.25;
                }
                return;
            }
            if !masked && sx == 3.0 && sy == 3.0 {
                let row: usize = 3 * y * image.stride;
                let top: &[u8] = &image.pixels[row..row + image.width];
                let row: usize = row + image.stride;
                let middle: &[u8] = &image.pixels[row..row + image.width];
                let row: usize = row + image.stride;
                let bottom: &[u8] = &image.pixels[row..row + image.width];
                for (((value, top), middle), bottom) in output
                    .iter_mut()
                    .zip(top.as_chunks::<3>().0.iter())
                    .zip(middle.as_chunks::<3>().0.iter())
                    .zip(bottom.as_chunks::<3>().0.iter())
                {
                    let sum: u32 = top[0] as u32
                        + top[1] as u32
                        + top[2] as u32
                        + middle[0] as u32
                        + middle[1] as u32
                        + middle[2] as u32
                        + bottom[0] as u32
                        + bottom[1] as u32
                        + bottom[2] as u32;
                    *value = sum as f64 / 9.0;
                }
                return;
            }
            if cached {
                let ys: Span = ay.spans[y];
                let yweights: &[f64] = &ay.weights[ys.offset..ys.offset + ys.count];
                for (output_value, xs) in output.iter_mut().zip(&ax.spans) {
                    let xweights: &[f64] = &ax.weights[xs.offset..xs.offset + xs.count];
                    let mut value: f64 = 0.0;
                    for (j, wy) in yweights.iter().enumerate() {
                        let row: usize = (ys.first + j) * image.stride;
                        let pixels: &[u8] = &image.pixels[row..row + image.width];
                        let pixels: &[u8] = &pixels[xs.first..xs.first + xs.count];
                        for (wx, pixel) in xweights.iter().zip(pixels) {
                            value += wx * wy * *pixel as f64;
                        }
                    }
                    *output_value = value / (sx * sy);
                }
                return;
            }
            for (x, output_value) in output.iter_mut().enumerate() {
                let mut value: f64 = 0.0;
                let mut weight: f64 = 0.0;
                let x0: f64 = x as f64 * sx;
                let x1: f64 = (x + 1) as f64 * sx;
                let y0: f64 = y as f64 * sy;
                let y1: f64 = (y + 1) as f64 * sy;
                for yy in y0 as usize..image.height.min(y1.ceil() as usize) {
                    let wy: f64 = min_f64(y1, (yy + 1) as f64) - max_f64(y0, yy as f64);
                    for xx in x0 as usize..image.width.min(x1.ceil() as usize) {
                        let wx: f64 = min_f64(x1, (xx + 1) as f64) - max_f64(x0, xx as f64);
                        if image.valid(Point {
                            x: xx as f64,
                            y: yy as f64,
                        }) {
                            value += wx * wy * image.pixels[yy * image.stride + xx] as f64;
                            weight += wx * wy;
                        }
                    }
                }
                *output_value = if !masked {
                    value / (sx * sy)
                } else if weight > 0.0
                    && image.valid(Point {
                        x: (x0 + x1) * 0.5 - 0.5,
                        y: (y0 + y1) * 0.5 - 0.5,
                    })
                {
                    value / weight
                } else {
                    f64::NAN
                };
            }
        },
    );
    gray
}

/// Apply the native separable Gaussian or its clamped unsharp counterpart.
fn filter(gray: &mut Vec<f64>, w: usize, h: usize, sigma: f64, masked: bool, workers: &Workers) {
    if sigma == 0.0 {
        return;
    }
    let s: f64 = sigma.abs();
    let radius: i32 = (3.0 * s).ceil().max(1.0) as i32;
    let mut kernel: Vec<f64> = vec![0.0; (2 * radius + 1) as usize];
    let mut sum: f64 = 0.0;
    for i in -radius..=radius {
        let scaled: f64 = i as f64 / s;
        kernel[(i + radius) as usize] = (-0.5 * scaled * scaled).exp();
        sum += kernel[(i + radius) as usize];
    }
    for value in &mut kernel {
        *value /= sum;
    }
    let mut tmp: Vec<f64> = vec![0.0; gray.len()];
    let mut blur: Vec<f64> = vec![0.0; gray.len()];
    workers.rows(
        &mut tmp,
        w,
        w * h > 65536,
        |y: usize, output: &mut [f64]| {
            for (x, value) in output.iter_mut().enumerate() {
                let mut v: f64 = 0.0;
                let mut weight: f64 = 0.0;
                for i in -radius..=radius {
                    let n: f64 = gray[y * w + (x as i32 + i).clamp(0, w as i32 - 1) as usize];
                    if n.is_finite() {
                        v += n * kernel[(i + radius) as usize];
                        weight += kernel[(i + radius) as usize];
                    }
                }
                *value = if !masked {
                    v
                } else if gray[y * w + x].is_finite() && weight > 0.0 {
                    v / weight
                } else {
                    f64::NAN
                };
            }
        },
    );
    workers.rows(
        &mut blur,
        w,
        w * h > 65536,
        |y: usize, output: &mut [f64]| {
            for (x, value) in output.iter_mut().enumerate() {
                let mut v: f64 = 0.0;
                let mut weight: f64 = 0.0;
                for i in -radius..=radius {
                    let n: f64 = tmp[(y as i32 + i).clamp(0, h as i32 - 1) as usize * w + x];
                    if n.is_finite() {
                        v += n * kernel[(i + radius) as usize];
                        weight += kernel[(i + radius) as usize];
                    }
                }
                if masked {
                    v = if gray[y * w + x].is_finite() && weight > 0.0 {
                        v / weight
                    } else {
                        f64::NAN
                    };
                }
                *value = if sigma > 0.0 {
                    v
                } else {
                    clamp_f64(2.0 * gray[y * w + x] - v, 0.0, 255.0)
                };
            }
        },
    );
    *gray = blur;
}

#[derive(Clone, Copy)]
struct Run {
    y: i32,
    x0: i32,
    x1: i32,
    parent: usize,
    color: i32,
}

/// Find a component root with the same path-halving/link direction as native.
fn root(runs: &mut [Run], mut i: usize) -> usize {
    while runs[i].parent != i {
        runs[i].parent = runs[runs[i].parent].parent;
        i = runs[i].parent;
    }
    i
}

/// Extract dark eight-connected and light four/eight-connected runs in native order.
fn components(binary: &[u8], w: usize, h: usize, light: bool) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    let mut previous: Vec<usize> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for color in 0..=i32::from(light) {
        previous.clear();
        for y in 0..h {
            current.clear();
            let mut cursor: usize = 0;
            let mut x: i32 = 0;
            while x < w as i32 {
                while x < w as i32 && binary[y * w + x as usize] & (1 << color) == 0 {
                    x += 1;
                }
                let x0: i32 = x;
                while x < w as i32 && binary[y * w + x as usize] & (1 << color) != 0 {
                    x += 1;
                }
                if x0 == x {
                    continue;
                }
                let id: usize = runs.len();
                runs.push(Run {
                    y: y as i32,
                    x0,
                    x1: x - 1,
                    parent: id,
                    color,
                });
                current.push(id);
                while cursor < previous.len() && runs[previous[cursor]].x1 < x0 - 1 {
                    cursor += 1;
                }
                let mut j: usize = cursor;
                while j < previous.len() && runs[previous[j]].x0 <= x {
                    let old: usize = previous[j];
                    if color == 0 || (runs[old].x1 >= x0 && runs[old].x0 < x) {
                        let a: usize = root(&mut runs, id);
                        let b: usize = root(&mut runs, old);
                        if a != b {
                            runs[b].parent = a;
                        }
                    }
                    j += 1;
                }
            }
            std::mem::swap(&mut previous, &mut current);
        }
    }
    if light {
        let original: usize = runs.len();
        previous.clear();
        current.clear();
        let mut row: i32 = -1;
        let mut cursor: usize = 0;
        for i in 0..original {
            let mut run: Run = runs[i];
            if run.color != 1 {
                continue;
            }
            if run.y != row {
                if run.y == row + 1 {
                    std::mem::swap(&mut previous, &mut current);
                } else {
                    previous.clear();
                }
                current.clear();
                row = run.y;
                cursor = 0;
            }
            let id: usize = runs.len();
            run.parent = id;
            runs.push(run);
            current.push(id);
            while cursor < previous.len() && runs[previous[cursor]].x1 < run.x0 - 1 {
                cursor += 1;
            }
            let mut j: usize = cursor;
            while j < previous.len() && runs[previous[j]].x0 <= run.x1 + 1 {
                let a: usize = root(&mut runs, id);
                let b: usize = root(&mut runs, previous[j]);
                if a != b {
                    runs[b].parent = a;
                }
                j += 1;
            }
        }
    }
    runs
}

/// Generate primary midpoint proposals followed by conservative dark recovery.
pub(crate) fn candidates(
    image: &Image<'_>,
    decimate: f64,
    sigma: f64,
    workers: &Workers,
    light: bool,
    dark_fraction: f64,
) -> Vec<Proposal> {
    let w: usize = ((image.width as f64 / decimate).ceil() as usize).max(1);
    let h: usize = ((image.height as f64 / decimate).ceil() as usize).max(1);
    let sx: f64 = image.width as f64 / w as f64;
    let sy: f64 = image.height as f64 / h as f64;
    let mut gray: Vec<f64> = resample(image, w, h, workers);
    filter(&mut gray, w, h, sigma, image.source_map.is_some(), workers);
    let tw: usize = w.div_ceil(8);
    let th: usize = h.div_ceil(8);
    let mut lo: Vec<f64> = vec![255.0; tw * th];
    let mut hi: Vec<f64> = vec![0.0; tw * th];
    for (y, row) in gray.chunks_exact(w).enumerate() {
        let lows = &mut lo[(y / 8) * tw..(y / 8 + 1) * tw];
        let highs = &mut hi[(y / 8) * tw..(y / 8 + 1) * tw];
        for ((low, high), pixels) in lows.iter_mut().zip(highs).zip(row.chunks(8)) {
            for &v in pixels {
                *low = min_f64(*low, v);
                *high = max_f64(*high, v);
            }
        }
    }
    // The same ordered neighborhood extrema serve every row and both passes.
    let mut ranges: Vec<(f64, f64)> = Vec::with_capacity(tw * th);
    for ty in 0..th {
        for tx in 0..tw {
            let mut low = 255.0;
            let mut high = 0.0;
            for yy in ty.saturating_sub(1)..=(th - 1).min(ty + 1) {
                for xx in tx.saturating_sub(1)..=(tw - 1).min(tx + 1) {
                    let i = yy * tw + xx;
                    low = min_f64(low, lo[i]);
                    high = max_f64(high, hi[i]);
                }
            }
            ranges.push((low, high));
        }
    }
    let mut out: Vec<Proposal> = Vec::new();
    for pass in 0..if dark_fraction > 0.0 { 1 } else { 2 } {
        let fraction: f64 = if dark_fraction > 0.0 {
            dark_fraction
        } else if pass == 0 {
            0.5
        } else {
            0.3
        };
        let pass_light: bool = light && fraction == 0.5;
        let mut binary: Vec<u8> = vec![0; gray.len()];
        workers.rows(
            &mut binary,
            w,
            w * h > 65536,
            |y: usize, output: &mut [u8]| {
                let row = &gray[y * w..(y + 1) * w];
                let tile_ranges = &ranges[(y / 8) * tw..(y / 8 + 1) * tw];
                for ((pixels, output), &(low, high)) in
                    row.chunks(8).zip(output.chunks_mut(8)).zip(tile_ranges)
                {
                    if high - low < 12.0 {
                        continue;
                    }
                    let dark: f64 = low + fraction * (high - low);
                    let bright: f64 = low + 0.5 * (high - low);
                    for (&v, value) in pixels.iter().zip(output) {
                        *value = u8::from(v < dark) | (u8::from(v > bright) << 1);
                    }
                }
            },
        );
        let mut runs: Vec<Run> = components(&binary, w, h, pass_light);
        let mut points: Vec<Vec<Point>> = vec![Vec::new(); runs.len()];
        let mut area: Vec<i32> = vec![0; runs.len()];
        for i in 0..runs.len() {
            let id: usize = root(&mut runs, i);
            let run: Run = runs[i];
            area[id] += run.x1 - run.x0 + 1;
            for y in [run.y as f64 - 0.5, run.y as f64 + 0.5] {
                points[id].push(Point {
                    x: run.x0 as f64 - 0.5,
                    y,
                });
                points[id].push(Point {
                    x: run.x1 as f64 + 0.5,
                    y,
                });
            }
        }
        for i in 0..points.len() {
            if area[i] < 12 || points[i].len() < 12 {
                continue;
            }
            let boundary: Vec<Point> = hull(std::mem::take(&mut points[i]));
            if boundary.len() < 4 {
                continue;
            }
            let Some(mut quad) = fit_quad(&boundary) else {
                continue;
            };
            for point in &mut quad {
                point.x = (point.x + 0.5) * sx - 0.5;
                point.y = (point.y + 0.5) * sy - 0.5;
            }
            let mut top_left: usize = 0;
            for k in 1..4 {
                if quad[k].x + quad[k].y < quad[top_left].x + quad[top_left].y {
                    top_left = k;
                }
            }
            quad.rotate_left(top_left);
            out.push(Proposal {
                quad,
                color: runs[i].color,
                recovery: fraction < 0.5,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct overlap oracle independent of the cached and integer fast paths.
    fn oracle(image: &Image<'_>, w: usize, h: usize) -> Vec<f64> {
        let sx: f64 = image.width as f64 / w as f64;
        let sy: f64 = image.height as f64 / h as f64;
        let mut output: Vec<f64> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let x0: f64 = x as f64 * sx;
                let x1: f64 = (x + 1) as f64 * sx;
                let y0: f64 = y as f64 * sy;
                let y1: f64 = (y + 1) as f64 * sy;
                let mut value: f64 = 0.0;
                let mut weight: f64 = 0.0;
                for yy in y0 as usize..image.height.min(y1.ceil() as usize) {
                    let wy: f64 = min_f64(y1, (yy + 1) as f64) - max_f64(y0, yy as f64);
                    for xx in x0 as usize..image.width.min(x1.ceil() as usize) {
                        let wx: f64 = min_f64(x1, (xx + 1) as f64) - max_f64(x0, xx as f64);
                        if image.valid(Point {
                            x: xx as f64,
                            y: yy as f64,
                        }) {
                            value += wx * wy * image.pixels[yy * image.stride + xx] as f64;
                            weight += wx * wy;
                        }
                    }
                }
                output.push(if image.source_map.is_none() {
                    value / (sx * sy)
                } else if weight > 0.0
                    && image.valid(Point {
                        x: (x0 + x1) * 0.5 - 0.5,
                        y: (y0 + y1) * 0.5 - 0.5,
                    })
                {
                    value / weight
                } else {
                    f64::NAN
                });
            }
        }
        output
    }

    /// Compare NaNs as missing observations and every finite result bit-for-bit.
    fn same(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!(
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                "{a:?} != {b:?}"
            );
        }
    }

    /// Exercise exact scales, fractional cells, unpadded last rows, extrema, and masks.
    #[test]
    fn area_matches_oracle() {
        let workers: Workers = Workers::new(1).unwrap();
        for (width, height, w, h) in [
            (7, 5, 7, 5),
            (14, 10, 7, 5),
            (21, 15, 7, 5),
            (17, 13, 8, 6),
            (7, 5, 3, 2),
            (7, 5, 1, 1),
        ] {
            let stride: usize = width + 5;
            for pattern in 0..3 {
                let pixels: Vec<u8> = (0..stride * (height - 1) + width)
                    .map(|i: usize| match pattern {
                        0 => 0,
                        1 => 255,
                        _ => ((i * 137 + i / stride * 19) % 256) as u8,
                    })
                    .collect();
                let mut image: Image<'_> = Image {
                    pixels: &pixels,
                    width,
                    height,
                    stride,
                    source_map: None,
                    source_width: 0,
                    source_height: 0,
                };
                same(&resample(&image, w, h, &workers), &oracle(&image, w, h));
                image.source_map = Some([1.0, 0.0, -2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0]);
                image.source_width = (width - 3) as u32;
                image.source_height = (height - 2) as u32;
                let expected: Vec<f64> = oracle(&image, w, h);
                same(&resample(&image, w, h, &workers), &expected);
                if w > 1 {
                    assert!(expected.iter().any(|v: &f64| v.is_nan()));
                }
            }
        }
    }

    /// Ensure actual warm-dispatch rows and both filter signs are worker-count invariant.
    #[test]
    fn row_workers_match() {
        let width: usize = 519;
        let height: usize = 517;
        let stride: usize = width + 3;
        let pixels: Vec<u8> = (0..stride * height)
            .map(|i: usize| ((i * 73 + i / stride) % 256) as u8)
            .collect();
        let mut image: Image<'_> = Image {
            pixels: &pixels,
            width,
            height,
            stride,
            source_map: None,
            source_width: 0,
            source_height: 0,
        };
        let serial: Workers = Workers::new(1).unwrap();
        for masked in [false, true] {
            if masked {
                image.source_map = Some([1.0, 0.0, -3.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0]);
                image.source_width = 510;
                image.source_height = 510;
            }
            let expected: Vec<f64> = resample(&image, 260, 259, &serial);
            for threads in [2, 4] {
                let parallel: Workers = Workers::new(threads).unwrap();
                let actual: Vec<f64> = resample(&image, 260, 259, &parallel);
                same(&actual, &expected);
                for sigma in [-0.8, 0.8] {
                    let mut a: Vec<f64> = actual.clone();
                    let mut b: Vec<f64> = expected.clone();
                    filter(&mut a, 260, 259, sigma, masked, &parallel);
                    filter(&mut b, 260, 259, sigma, masked, &serial);
                    same(&a, &b);
                }
            }
        }
    }

    /// Preserve left-edge signed comparisons, link direction, and light alternatives.
    #[test]
    fn diagonal_connectivity_and_root_order() {
        let mut dark: Vec<Run> = components(&[0, 1, 1, 0], 2, 2, false);
        assert_eq!(dark.len(), 2);
        assert_eq!(root(&mut dark, 0), 1);
        assert_eq!(root(&mut dark, 1), 1);
        let mut light: Vec<Run> = components(&[0, 2, 2, 0], 2, 2, true);
        assert_eq!(light.len(), 4);
        assert_eq!(root(&mut light, 0), 0);
        assert_eq!(root(&mut light, 1), 1);
        assert_eq!(root(&mut light, 2), 3);
        assert_eq!(root(&mut light, 3), 3);
        let mut gap: Vec<Run> = components(&[2, 0, 0, 0, 2, 0], 2, 3, true);
        assert_eq!(gap.len(), 4);
        assert_eq!(root(&mut gap, 2), 2);
        assert_eq!(root(&mut gap, 3), 3);
    }

    /// Keep midpoint proposals authoritative and map decimated boundary centers back.
    #[test]
    fn proposal_order_and_center_mapping() {
        let mut pixels: Vec<u8> = vec![255; 32 * 32];
        for y in 8..24 {
            for x in 8..24 {
                pixels[y * 32 + x] = 0;
            }
        }
        let image: Image<'_> = Image {
            pixels: &pixels,
            width: 32,
            height: 32,
            stride: 32,
            source_map: None,
            source_width: 0,
            source_height: 0,
        };
        let workers: Workers = Workers::new(1).unwrap();
        for decimate in [1.0, 2.0] {
            let proposals: Vec<Proposal> = candidates(&image, decimate, 0.0, &workers, false, 0.0);
            assert_eq!(proposals.len(), 2);
            assert!(!proposals[0].recovery);
            assert!(proposals[1].recovery);
            for proposal in &proposals {
                assert_eq!(proposal.color, 0);
                for (point, expected) in
                    proposal
                        .quad
                        .iter()
                        .zip([(7.5, 7.5), (23.5, 7.5), (23.5, 23.5), (7.5, 23.5)])
                {
                    assert!((point.x - expected.0).abs() < 1e-10);
                    assert!((point.y - expected.1).abs() < 1e-10);
                }
            }
            assert_eq!(
                candidates(&image, decimate, 0.0, &workers, false, 0.3).len(),
                1
            );
        }
    }
}
