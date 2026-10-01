//! Area resampling, local thresholding, and ordered run-component proposals.

use crate::geometry::{fit_quad, hull, Point, Quad};
use crate::image::Image;
use crate::workers::Workers;

pub(crate) struct Proposal {
    pub(crate) quad: Quad,
    pub(crate) color: i32,
    pub(crate) recovery: bool,
}

pub(crate) struct Scratch {
    gray: Vec<f32>,
    binary: Vec<u8>,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self {
            gray: Vec::new(),
            binary: Vec::new(),
        }
    }
}

pub(crate) struct PreparedImage {
    width: usize,
    height: usize,
    scale_x: f64,
    scale_y: f64,
}

#[derive(Clone, Copy)]
struct Span {
    first: usize,
    offset: usize,
    count: usize,
}

struct AreaAxis {
    spans: Vec<Span>,
    weights: Vec<f32>,
}

impl AreaAxis {
    fn new(size: usize, cells: usize, scale: f32) -> Self {
        let mut axis = Self {
            spans: Vec::with_capacity(cells),
            weights: Vec::with_capacity(if cells == 0 { 0 } else { size + cells }),
        };
        for cell in 0..cells {
            let lo = cell as f32 * scale;
            let hi = (cell + 1) as f32 * scale;
            let first = lo as usize;
            let end = size.min(hi.ceil() as usize);
            axis.spans.push(Span {
                first,
                offset: axis.weights.len(),
                count: end - first,
            });
            for i in first..end {
                axis.weights
                    .push((hi.min((i + 1) as f32) - lo.max(i as f32)).max(0.0));
            }
        }
        axis
    }
}

fn resample(image: &Image<'_>, width: usize, height: usize, workers: &Workers, gray: &mut [f32]) {
    let scale_x = image.width as f32 / width as f32;
    let scale_y = image.height as f32 / height as f32;
    let masked = image.source_map.is_some();
    let cached = !masked
        && !((scale_x == 1.0 && scale_y == 1.0)
            || (scale_x == 2.0 && scale_y == 2.0)
            || (scale_x == 3.0 && scale_y == 3.0));
    let axis_x = AreaAxis::new(image.width, if cached { width } else { 0 }, scale_x);
    let axis_y = AreaAxis::new(image.height, if cached { height } else { 0 }, scale_y);
    workers.rows(
        gray,
        width,
        width * height > 65536,
        |y: usize, output: &mut [f32]| {
            if !masked && scale_x == 1.0 && scale_y == 1.0 {
                let row = y * image.stride;
                let pixels = &image.pixels[row..row + image.width];
                for (value, pixel) in output.iter_mut().zip(pixels) {
                    *value = *pixel as f32;
                }
                return;
            }
            if !masked && scale_x == 2.0 && scale_y == 2.0 {
                let mut row = 2 * y * image.stride;
                let top = &image.pixels[row..row + image.width];
                row += image.stride;
                let bottom = &image.pixels[row..row + image.width];
                for ((value, top), bottom) in output
                    .iter_mut()
                    .zip(top.as_chunks::<2>().0.iter())
                    .zip(bottom.as_chunks::<2>().0.iter())
                {
                    let sum =
                        top[0] as u32 + top[1] as u32 + bottom[0] as u32 + bottom[1] as u32;
                    *value = sum as f32 * 0.25;
                }
                return;
            }
            if !masked && scale_x == 3.0 && scale_y == 3.0 {
                let mut row = 3 * y * image.stride;
                let top = &image.pixels[row..row + image.width];
                row += image.stride;
                let middle = &image.pixels[row..row + image.width];
                row += image.stride;
                let bottom = &image.pixels[row..row + image.width];
                for (((value, top), middle), bottom) in output
                    .iter_mut()
                    .zip(top.as_chunks::<3>().0.iter())
                    .zip(middle.as_chunks::<3>().0.iter())
                    .zip(bottom.as_chunks::<3>().0.iter())
                {
                    let sum = top[0] as u32
                        + top[1] as u32
                        + top[2] as u32
                        + middle[0] as u32
                        + middle[1] as u32
                        + middle[2] as u32
                        + bottom[0] as u32
                        + bottom[1] as u32
                        + bottom[2] as u32;
                    *value = sum as f32 / 9.0;
                }
                return;
            }
            if cached {
                let y_span = axis_y.spans[y];
                let y_weights = &axis_y.weights[y_span.offset..y_span.offset + y_span.count];
                for (output_value, x_span) in output.iter_mut().zip(&axis_x.spans) {
                    let x_weights = &axis_x.weights[x_span.offset..x_span.offset + x_span.count];
                    let mut value = 0.0;
                    for (j, wy) in y_weights.iter().enumerate() {
                        let row = (y_span.first + j) * image.stride;
                        let pixels = &image.pixels[row + x_span.first..row + x_span.first + x_span.count];
                        for (wx, pixel) in x_weights.iter().zip(pixels) {
                            value += wx * wy * *pixel as f32;
                        }
                    }
                    *output_value = value / (scale_x * scale_y);
                }
                return;
            }
            for (x, output_value) in output.iter_mut().enumerate() {
                let mut value = 0.0;
                let mut weight = 0.0;
                let x0 = x as f32 * scale_x;
                let x1 = (x + 1) as f32 * scale_x;
                let y0 = y as f32 * scale_y;
                let y1 = (y + 1) as f32 * scale_y;
                for yy in y0 as usize..image.height.min(y1.ceil() as usize) {
                    let wy = y1.min((yy + 1) as f32) - y0.max(yy as f32);
                    for xx in x0 as usize..image.width.min(x1.ceil() as usize) {
                        let wx = x1.min((xx + 1) as f32) - x0.max(xx as f32);
                        if image.valid(Point {
                            x: xx as f64,
                            y: yy as f64,
                        }) {
                            value += wx * wy * image.pixels[yy * image.stride + xx] as f32;
                            weight += wx * wy;
                        }
                    }
                }
                *output_value = if !masked {
                    value / (scale_x * scale_y)
                } else if weight > 0.0
                    && image.valid(Point {
                        x: ((x0 + x1) * 0.5 - 0.5) as f64,
                        y: ((y0 + y1) * 0.5 - 0.5) as f64,
                    })
                {
                    value / weight
                } else {
                    f32::NAN
                };
            }
        },
    );
}

fn filter(
    gray: &mut [f32],
    width: usize,
    height: usize,
    sigma: f64,
    masked: bool,
    workers: &Workers,
) {
    if sigma == 0.0 {
        return;
    }
    let sigma_abs = sigma.abs() as f32;
    let radius = (3.0 * sigma_abs).ceil().max(1.0) as i32;
    let mut kernel = vec![0.0f32; (2 * radius + 1) as usize];
    let mut sum = 0.0;
    for i in -radius..=radius {
        let scaled = i as f32 / sigma_abs;
        kernel[(i + radius) as usize] = (-0.5 * scaled * scaled).exp();
        sum += kernel[(i + radius) as usize];
    }
    for value in &mut kernel {
        *value /= sum;
    }
    let mut tmp = vec![0.0f32; gray.len()];
    let mut blur = vec![0.0f32; gray.len()];
    workers.rows(
        &mut tmp,
        width,
        width * height > 65536,
        |y: usize, output: &mut [f32]| {
            for (x, value) in output.iter_mut().enumerate() {
                let mut acc = 0.0;
                let mut weight = 0.0;
                for i in -radius..=radius {
                    let neighbor =
                        gray[y * width + (x as i32 + i).clamp(0, width as i32 - 1) as usize];
                    if neighbor.is_finite() {
                        acc += neighbor * kernel[(i + radius) as usize];
                        weight += kernel[(i + radius) as usize];
                    }
                }
                *value = if !masked {
                    acc
                } else if gray[y * width + x].is_finite() && weight > 0.0 {
                    acc / weight
                } else {
                    f32::NAN
                };
            }
        },
    );
    let tmp = tmp.as_slice();
    workers.rows(
        &mut blur,
        width,
        width * height > 65536,
        |y: usize, output: &mut [f32]| {
            for (x, value) in output.iter_mut().enumerate() {
                let mut acc = 0.0;
                let mut weight = 0.0;
                for i in -radius..=radius {
                    let neighbor =
                        tmp[(y as i32 + i).clamp(0, height as i32 - 1) as usize * width + x];
                    if neighbor.is_finite() {
                        acc += neighbor * kernel[(i + radius) as usize];
                        weight += kernel[(i + radius) as usize];
                    }
                }
                if masked {
                    acc = if gray[y * width + x].is_finite() && weight > 0.0 {
                        acc / weight
                    } else {
                        f32::NAN
                    };
                }
                *value = if sigma > 0.0 {
                    acc
                } else {
                    (2.0 * gray[y * width + x] - acc).clamp(0.0, 255.0)
                };
            }
        },
    );
    gray.copy_from_slice(&blur);
}

#[derive(Clone, Copy)]
struct Run {
    y: i32,
    x0: i32,
    x1: i32,
    parent: usize,
    color: i32,
}

fn root(runs: &mut [Run], mut i: usize) -> usize {
    while runs[i].parent != i {
        runs[i].parent = runs[runs[i].parent].parent;
        i = runs[i].parent;
    }
    i
}

fn components(binary: &[u8], width: usize, height: usize, light: bool) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut previous: Vec<usize> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for color in 0..=i32::from(light) {
        previous.clear();
        for y in 0..height {
            current.clear();
            let mut cursor = 0;
            let mut x = 0;
            while x < width as i32 {
                while x < width as i32 && binary[y * width + x as usize] & (1 << color) == 0 {
                    x += 1;
                }
                let x0 = x;
                while x < width as i32 && binary[y * width + x as usize] & (1 << color) != 0 {
                    x += 1;
                }
                if x0 == x {
                    continue;
                }
                let id = runs.len();
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
                let mut j = cursor;
                while j < previous.len() && runs[previous[j]].x0 <= x {
                    let old = previous[j];
                    if color == 0 || (runs[old].x1 >= x0 && runs[old].x0 < x) {
                        let a = root(&mut runs, id);
                        let b = root(&mut runs, old);
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
        let original = runs.len();
        previous.clear();
        current.clear();
        let mut row = -1;
        let mut cursor = 0;
        for i in 0..original {
            let mut run = runs[i];
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
            let id = runs.len();
            run.parent = id;
            runs.push(run);
            current.push(id);
            while cursor < previous.len() && runs[previous[cursor]].x1 < run.x0 - 1 {
                cursor += 1;
            }
            let mut j = cursor;
            while j < previous.len() && runs[previous[j]].x0 <= run.x1 + 1 {
                let a = root(&mut runs, id);
                let b = root(&mut runs, previous[j]);
                if a != b {
                    runs[b].parent = a;
                }
                j += 1;
            }
        }
    }
    runs
}

pub(crate) fn prepare(
    image: &Image<'_>,
    decimate: f64,
    sigma: f64,
    workers: &Workers,
    scratch: &mut Scratch,
) -> PreparedImage {
    let width = ((image.width as f64 / decimate).ceil() as usize).max(1);
    let height = ((image.height as f64 / decimate).ceil() as usize).max(1);
    scratch.gray.resize(width * height, 0.0);
    resample(image, width, height, workers, &mut scratch.gray);
    filter(
        &mut scratch.gray,
        width,
        height,
        sigma,
        image.source_map.is_some(),
        workers,
    );
    PreparedImage {
        width,
        height,
        scale_x: image.width as f64 / width as f64,
        scale_y: image.height as f64 / height as f64,
    }
}

pub(crate) fn proposals(
    prepared: &PreparedImage,
    scratch: &mut Scratch,
    workers: &Workers,
    light: bool,
    recovery: bool,
) -> Vec<Proposal> {
    let width = prepared.width;
    let height = prepared.height;
    let tile_width = width.div_ceil(8);
    let tile_height = height.div_ceil(8);
    let mut lo = vec![255.0f32; tile_width * tile_height];
    let mut hi = vec![0.0f32; tile_width * tile_height];
    for (y, row) in scratch.gray.chunks_exact(width).enumerate() {
        let lows = &mut lo[(y / 8) * tile_width..(y / 8 + 1) * tile_width];
        let highs = &mut hi[(y / 8) * tile_width..(y / 8 + 1) * tile_width];
        for ((low, high), pixels) in lows.iter_mut().zip(highs).zip(row.chunks(8)) {
            for &value in pixels {
                *low = low.min(value);
                *high = high.max(value);
            }
        }
    }
    let mut ranges = Vec::with_capacity(tile_width * tile_height);
    for tile_y in 0..tile_height {
        for tile_x in 0..tile_width {
            let mut low: f32 = 255.0;
            let mut high: f32 = 0.0;
            for yy in tile_y.saturating_sub(1)..=(tile_height - 1).min(tile_y + 1) {
                for xx in tile_x.saturating_sub(1)..=(tile_width - 1).min(tile_x + 1) {
                    let i = yy * tile_width + xx;
                    low = low.min(lo[i]);
                    high = high.max(hi[i]);
                }
            }
            ranges.push((low, high));
        }
    }
    let fraction = if recovery { 0.3f32 } else { 0.5 };
    let pass_light = light && !recovery;
    scratch.binary.resize(scratch.gray.len(), 0);
    scratch.binary.fill(0);
    {
        let gray = scratch.gray.as_slice();
        let binary = scratch.binary.as_mut_slice();
        workers.rows(
            binary,
            width,
            width * height > 65536,
            |y: usize, output: &mut [u8]| {
                let row = &gray[y * width..(y + 1) * width];
                let tile_ranges = &ranges[(y / 8) * tile_width..(y / 8 + 1) * tile_width];
                for ((pixels, output), &(low, high)) in
                    row.chunks(8).zip(output.chunks_mut(8)).zip(tile_ranges)
                {
                    if high - low < 12.0 {
                        continue;
                    }
                    let dark = low + fraction * (high - low);
                    let bright = low + 0.5 * (high - low);
                    for (&value, out) in pixels.iter().zip(output) {
                        *out = u8::from(value < dark) | (u8::from(value > bright) << 1);
                    }
                }
            },
        );
    }
    let mut runs = components(&scratch.binary, width, height, pass_light);
    let mut points: Vec<Vec<Point>> = vec![Vec::new(); runs.len()];
    let mut area = vec![0; runs.len()];
    for i in 0..runs.len() {
        let id = root(&mut runs, i);
        let run = runs[i];
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
    let mut out = Vec::new();
    for i in 0..points.len() {
        if area[i] < 12 || points[i].len() < 12 {
            continue;
        }
        let boundary = hull(std::mem::take(&mut points[i]));
        if boundary.len() < 4 {
            continue;
        }
        let Some(mut quad) = fit_quad(&boundary) else {
            continue;
        };
        for point in &mut quad {
            point.x = (point.x + 0.5) * prepared.scale_x - 0.5;
            point.y = (point.y + 0.5) * prepared.scale_y - 0.5;
        }
        let mut top_left = 0;
        for k in 1..4 {
            if quad[k].x + quad[k].y < quad[top_left].x + quad[top_left].y {
                top_left = k;
            }
        }
        quad.rotate_left(top_left);
        out.push(Proposal {
            quad,
            color: runs[i].color,
            recovery,
        });
    }
    out
}
