//! Scalar decoding and bounded template grid fit.

use crate::detection::Detection;
use crate::families::IndexedFamily;
use crate::geometry::{dot, homography, min_f64, norm, project, Homography, Point, Quad};
use crate::image::Image;

/// Rotate a unit-square coordinate clockwise by quarter turns.
fn rotate(mut point: Point, rotation: i32) -> Point {
    for _ in 0..rotation {
        point = Point {
            x: 1.0 - point.y,
            y: point.x,
        };
    }
    point
}

/// Grid-fit least-squares unknowns: homography updates plus the edge bias.
const UNKNOWNS: usize = 9;

/// Grid-fit effort: Gauss-Newton iterations and the edge-profile centroid step.
#[derive(Clone, Copy)]
pub(crate) struct GridFit {
    pub(crate) iterations: usize,
    pub(crate) walk_step: f64,
}

/// The detector's grid fit, starting from edge-refined corners.
pub(crate) const FULL_FIT: GridFit = GridFit {
    iterations: 2,
    walk_step: 0.125,
};

/// The tracker's grid fit, starting from last frame's grid-fitted corners. On the
/// temporal replay it matches FULL_FIT's corner error at under half the cost.
pub(crate) const TRACK_FIT: GridFit = GridFit {
    iterations: 1,
    walk_step: 0.25,
};

/// Fit independent template edges, restoring the input on failure.
fn refine_grid(
    h: &mut Homography,
    image: &Image<'_>,
    indexed: &IndexedFamily,
    id: i32,
    rotation: i32,
    fit: GridFit,
) -> bool {
    let family = indexed.family;
    if family.total != family.width + 2 || family.reversed {
        return false;
    }
    let white = |x: i32, y: i32| -> i32 { indexed.white_cell(id, x, y) };
    let initial = *h;
    let mut cell = 1e9;
    for k in 0..4 {
        let a = rotate(
            Point {
                x: f64::from(k == 1 || k == 2),
                y: f64::from(k >= 2),
            },
            rotation,
        );
        let next = (k + 1) % 4;
        let b = rotate(
            Point {
                x: f64::from(next == 1 || next == 2),
                y: f64::from(next >= 2),
            },
            rotation,
        );
        cell = min_f64(
            cell,
            norm(project(h, a.x, a.y) - project(h, b.x, b.y)) / f64::from(family.width),
        );
    }
    if cell < 1.5 {
        return false;
    }
    for _iteration in 0..fit.iterations {
        // Unknowns: eight homography updates, then one shared shift of every edge
        // toward its dark side (blur, gamma or exposure bias), solved and discarded.
        let mut system = [[0.0f64; UNKNOWNS + 1]; UNKNOWNS];
        let mut count = 0;
        for axis in 0..2 {
            for edge in 0..=family.width {
                for along in 0..family.width {
                    let x = if axis != 0 { along } else { edge };
                    let y = if axis != 0 { edge } else { along };
                    let before = white(x - i32::from(axis == 0), y - axis);
                    let after = white(x, y);
                    if before == after {
                        continue;
                    }
                    let phase = 0.5;
                    let width = f64::from(family.width);
                    let uv = rotate(
                        Point {
                            x: (if axis != 0 {
                                f64::from(along) + phase
                            } else {
                                f64::from(edge)
                            }) / width,
                            y: (if axis != 0 {
                                f64::from(edge)
                            } else {
                                f64::from(along) + phase
                            }) / width,
                        },
                        rotation,
                    );
                    let uv2 = rotate(
                        Point {
                            x: (if axis != 0 {
                                f64::from(along) + phase
                            } else {
                                f64::from(edge) + 0.1
                            }) / width,
                            y: (if axis != 0 {
                                f64::from(edge) + 0.1
                            } else {
                                f64::from(along) + phase
                            }) / width,
                        },
                        rotation,
                    );
                    let tangent = rotate(
                        Point {
                            x: (if axis != 0 {
                                f64::from(along) + phase + 0.1
                            } else {
                                f64::from(edge)
                            }) / width,
                            y: (if axis != 0 {
                                f64::from(edge)
                            } else {
                                f64::from(along) + phase + 0.1
                            }) / width,
                        },
                        rotation,
                    );
                    let p = project(h, uv.x, uv.y);
                    let d = project(h, tangent.x, tangent.y) - p;
                    let mut n = Point {
                        x: -d.y / norm(d),
                        y: d.x / norm(d),
                    };
                    if dot(n, project(h, uv2.x, uv2.y) - p) < 0.0 {
                        n = n * -1.0;
                    }
                    let aperture = min_f64(0.75, cell * 0.2);
                    let radius = min_f64(1.5, cell * 0.3);
                    // Taps reach |s| <= radius + 2 (search plus walk), then the aperture.
                    let reach = radius + 2.0 + aperture + 0.01;
                    let interior = image.interior_segment(
                        Point {
                            x: n.x.mul_add(-reach, p.x),
                            y: n.y.mul_add(-reach, p.y),
                        },
                        Point {
                            x: n.x.mul_add(reach, p.x),
                            y: n.y.mul_add(reach, p.y),
                        },
                    );
                    let tap = |u: f64| {
                        let q = Point {
                            x: n.x.mul_add(u, p.x),
                            y: n.y.mul_add(u, p.y),
                        };
                        if interior {
                            image.sample_interior(q)
                        } else {
                            image.sample(q)
                        }
                    };
                    let gradient = |s: f64| -> f64 {
                        f64::from(after - before) * (tap(s + aperture) - tap(s - aperture))
                    };
                    let mut best = 4.0;
                    let mut offset = 0.0;
                    let mut s = -radius;
                    while s <= radius {
                        let g = gradient(s);
                        if g > best {
                            best = g;
                            offset = s;
                        }
                        s += 0.25;
                    }
                    if best <= 4.0 {
                        continue;
                    }
                    let mut mass = best;
                    let mut moment = offset * best;
                    for direction in [-1, 1] {
                        let mut t = fit.walk_step;
                        while t <= 2.0 {
                            let s = offset + f64::from(direction) * t;
                            let g = gradient(s);
                            if !g.is_finite() || g <= 0.0 {
                                break;
                            }
                            mass += g;
                            moment = s.mul_add(g, moment);
                            t += fit.walk_step;
                        }
                    }
                    let residual = moment / mass;
                    let z = h[6].mul_add(uv.x, h[7] * uv.y) + 1.0;
                    let j = [
                        n.x * uv.x / z,
                        n.x * uv.y / z,
                        n.x / z,
                        n.y * uv.x / z,
                        n.y * uv.y / z,
                        n.y / z,
                        -dot(n, p) * uv.x / z,
                        -dot(n, p) * uv.y / z,
                        // A positive bias moves the observed edge into its dark cell.
                        f64::from(before - after),
                    ];
                    for a in 0..UNKNOWNS {
                        for b in 0..UNKNOWNS {
                            system[a][b] = j[a].mul_add(j[b], system[a][b]);
                        }
                        system[a][UNKNOWNS] = j[a].mul_add(residual, system[a][UNKNOWNS]);
                    }
                    count += 1;
                }
            }
        }
        if count < 16 {
            *h = initial;
            return false;
        }
        // Weak prior of no bias keeps the solve stable when edges cannot separate it.
        system[UNKNOWNS - 1][UNKNOWNS - 1] += 1e-3 * f64::from(count);
        for a in 0..UNKNOWNS {
            let mut pivot = a;
            for b in a + 1..UNKNOWNS {
                if system[b][a].abs() > system[pivot][a].abs() {
                    pivot = b;
                }
            }
            for b in a..=UNKNOWNS {
                let value = system[a][b];
                system[a][b] = system[pivot][b];
                system[pivot][b] = value;
            }
            let scale = system[a][a];
            if scale.abs() < 1e-10 {
                *h = initial;
                return false;
            }
            for b in a..=UNKNOWNS {
                system[a][b] /= scale;
            }
            for row in 0..UNKNOWNS {
                if row != a {
                    let factor = system[row][a];
                    for b in a..=UNKNOWNS {
                        system[row][b] -= factor * system[a][b];
                    }
                }
            }
        }
        let mut next = *h;
        for a in 0..8 {
            next[a] += system[a][UNKNOWNS];
        }
        for uv in [
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ] {
            let p = project(&next, uv.x, uv.y);
            if !image.valid(p) || norm(p - project(&initial, uv.x, uv.y)) > cell * 0.25 {
                *h = initial;
                return false;
            }
        }
        *h = next;
    }
    true
}

/// Grid-fit a quad already known to show `id` upright, then confirm it decodes as `id`.
///
/// Skips the rotation search that `decode` runs before its grid fit. Returns None
/// when the grid fit fails, so unrefined corners are never reported.
pub(crate) fn decode_known(
    quad: &Quad,
    image: &Image<'_>,
    indexed: &IndexedFamily,
    family_index: u32,
    sharpening: f64,
    id: i32,
    fit: GridFit,
) -> Option<Detection> {
    let mut h = homography(quad)?;
    if !refine_grid(&mut h, image, indexed, id, 0, fit) {
        return None;
    }
    let fitted = [
        project(&h, 0.0, 0.0),
        project(&h, 1.0, 0.0),
        project(&h, 1.0, 1.0),
        project(&h, 0.0, 1.0),
    ];
    decode(&fitted, image, indexed, family_index, sharpening, false)
        .filter(|detection| detection.tag_id == id)
}

/// Cheap screen of an unrefined proposal before edge refinement and decoding.
///
/// Takes one sample per border cell, per adjacent exterior cell and per payload cell.
/// Accepts when most border cells are darker than most exterior cells (lighter for
/// reversed families) and the payload shows both dark and light cells. Thresholds are
/// looser than `decode` because the quad has not been refined yet.
pub(crate) fn plausible(quad: &Quad, image: &Image<'_>, indexed: &IndexedFamily) -> bool {
    let f = indexed.family;
    let Some(h) = homography(quad) else {
        return false;
    };
    let width = f64::from(f.width);
    let sample = |x: i32, y: i32| -> f64 {
        image.sample(project(
            &h,
            (f64::from(x) + 0.5) / width,
            (f64::from(y) + 0.5) / width,
        ))
    };
    let mut border = [0.0f64; 64];
    let mut exterior = [0.0f64; 64];
    let (mut nb, mut ne) = (0, 0);
    for side in 0..4 {
        for i in 0..f.width {
            let (x, y, ox, oy) = match side {
                1 => (f.width - 1, i, f.width, i),
                2 => (i, f.width - 1, i, f.width),
                3 => (0, i, -1, i),
                _ => (i, 0, i, -1),
            };
            let v = sample(x, y);
            if v.is_finite() && nb < border.len() {
                border[nb] = v;
                nb += 1;
            }
            let v = if indexed.bit_at(ox, oy).is_some() {
                f64::NAN
            } else {
                sample(ox, oy)
            };
            if v.is_finite() && ne < exterior.len() {
                exterior[ne] = v;
                ne += 1;
            }
        }
    }
    let total = (4 * f.width) as usize;
    if nb * 2 < total || ne < 4 {
        return false;
    }
    let (border, exterior) = (&border[..nb], &exterior[..ne]);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (inner, outer) = (mean(border), mean(exterior));
    let (black, white) = if f.reversed {
        (outer, inner)
    } else {
        (inner, outer)
    };
    if white - black < 10.0 {
        return false;
    }
    let threshold = (black + white) * 0.5;
    let is_dark = |v: f64| v < threshold;
    let dark_border = border.iter().filter(|&&v| is_dark(v) != f.reversed).count();
    let light_exterior = exterior
        .iter()
        .filter(|&&v| is_dark(v) == f.reversed)
        .count();
    if dark_border * 2 < nb || light_exterior * 2 < ne {
        return false;
    }
    let (mut dark, mut light) = (0, 0);
    for b in 0..f.nbits {
        let v = sample(f.bit_x[b], f.bit_y[b]);
        if v.is_finite() {
            if is_dark(v) {
                dark += 1;
            } else {
                light += 1;
            }
        }
    }
    dark > 0 && light > 0
}

/// Decode all rotations using observed border evidence and optionally verify a grid fit.
#[allow(clippy::needless_range_loop)] // Keep side and matrix indexing aligned with the frozen oracle.
pub(crate) fn decode(
    quad: &Quad,
    image: &Image<'_>,
    indexed: &IndexedFamily,
    family_index: u32,
    sharpening: f64,
    grid: bool,
) -> Option<Detection> {
    let f = indexed.family;
    let h = homography(quad)?;
    let mut border = Vec::new();
    let mut reference = Vec::new();
    let mut exterior: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let sample = |x: f64, y: f64, r: i32| -> f64 {
        let p = rotate(
            Point {
                x: x / f64::from(f.width),
                y: y / f64::from(f.width),
            },
            r,
        );
        image.sample(project(&h, p.x, p.y))
    };
    let payload = |x: i32, y: i32| -> bool { indexed.bit_at(x, y).is_some() };
    for side in 0..4 {
        for i in 0..f.width {
            let (x, y) = match side {
                1 => (f.width - 1, i),
                2 => (i, f.width - 1),
                3 => (0, i),
                _ => (i, 0),
            };
            let mut v = sample(f64::from(x) + 0.5, f64::from(y) + 0.5, 0);
            if !v.is_finite() {
                return None;
            }
            border.push(v);
            let (ox, oy) = match side {
                0 => (x, y - 1),
                1 => (x + 1, y),
                2 => (x, y + 1),
                _ => (x - 1, y),
            };
            if !payload(ox, oy) {
                v = sample(f64::from(ox) + 0.5, f64::from(oy) + 0.5, 0);
                if !v.is_finite() {
                    // Only real pixels along this side's physical outward ray contribute.
                    let bx = if side == 1 {
                        f64::from(f.width)
                    } else if side == 3 {
                        0.0
                    } else {
                        f64::from(x) + 0.5
                    };
                    let by = if side == 2 {
                        f64::from(f.width)
                    } else if side == 0 {
                        0.0
                    } else {
                        f64::from(y) + 0.5
                    };
                    let edge = project(&h, bx / f64::from(f.width), by / f64::from(f.width));
                    let delta = project(
                        &h,
                        (f64::from(ox) + 0.5) / f64::from(f.width),
                        (f64::from(oy) + 0.5) / f64::from(f.width),
                    ) - edge;
                    if image.sample(edge).is_finite() {
                        let t = image.ray_limit(edge, delta);
                        if t > 0.0 {
                            v = image.sample(edge + delta * (t * 0.999));
                        }
                    }
                }
                if v.is_finite() {
                    reference.push(v);
                    exterior[side].push(v);
                }
            }
        }
    }
    if reference.len() < 4 {
        return None;
    }
    let inner = border.iter().sum::<f64>() / border.len() as f64;
    let outer = reference.iter().sum::<f64>() / reference.len() as f64;
    let black = if f.reversed { outer } else { inner };
    let white = if f.reversed { inner } else { outer };
    if white - black < 15.0 {
        return None;
    }
    let threshold = (black + white) * 0.5;
    let bn = border
        .iter()
        .filter(|&&v| {
            if f.reversed {
                v > threshold
            } else {
                v < threshold
            }
        })
        .count();
    let wn = reference
        .iter()
        .filter(|&&v| {
            if f.reversed {
                v < threshold
            } else {
                v > threshold
            }
        })
        .count();
    if bn < (border.len() as f64 * 0.8) as usize || wn < (reference.len() as f64 * 0.7) as usize {
        return None;
    }
    for side in &exterior {
        let supported = side
            .iter()
            .filter(|&&v| {
                if f.reversed {
                    v < threshold
                } else {
                    v > threshold
                }
            })
            .count();
        if side.is_empty() || supported < (side.len() as f64 * 0.7).ceil() as usize {
            return None;
        }
    }
    // Bits and their sharpening neighbours sit on cell centres (i + 0.5) / width.
    // With a power-of-two width, rotating a centre lands exactly on another
    // centre, so all four rotations read one lazily filled grid of cells
    // -1..=width instead of resampling each rotation.
    let span = (f.width + 2) as usize;
    let shared = f.width > 0 && (f.width as u32).is_power_of_two();
    let mut cells: Vec<Option<f64>> = vec![None; if shared { span * span } else { 0 }];
    let mut cell = |x: i32, y: i32, r: i32| -> f64 {
        let (mut cx, mut cy) = (x, y);
        for _ in 0..r {
            (cx, cy) = (f.width - 1 - cy, cx);
        }
        if !shared || !(-1..=f.width).contains(&cx) || !(-1..=f.width).contains(&cy) {
            return sample(f64::from(x) + 0.5, f64::from(y) + 0.5, r);
        }
        let index = (cy + 1) as usize * span + (cx + 1) as usize;
        *cells[index].get_or_insert_with(|| sample(f64::from(cx) + 0.5, f64::from(cy) + 0.5, 0))
    };
    let mut best = 3;
    let mut id = -1;
    let mut rotation = 0;
    let mut tied = false;
    let mut margin = 0.0;
    for r in 0..4 {
        let mut word = 0u64;
        let mut sum = 0.0;
        let mut valid = true;
        for b in 0..f.nbits {
            let (x, y) = (f.bit_x[b], f.bit_y[b]);
            let mut v = cell(x, y, r);
            if !v.is_finite() {
                valid = false;
                break;
            }
            if sharpening != 0.0 {
                let mut neighbor = 0.0;
                let mut all = true;
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let n = cell(x + dx, y + dy, r);
                    if !n.is_finite() {
                        all = false;
                        break;
                    }
                    neighbor += n;
                }
                if all {
                    v += sharpening * (4.0 * v - neighbor);
                }
            }
            word = (word << 1) | u64::from(v > threshold);
            sum += (v - threshold).abs();
        }
        if !valid {
            continue;
        }
        let (matched_id, distance) = indexed.match_word(word);
        if matched_id < 0 {
            continue;
        }
        if distance < best {
            best = distance;
            id = matched_id;
            rotation = r;
            margin = sum / f.nbits as f64;
            tied = false;
        } else if distance == best {
            tied = true;
        }
    }
    if id < 0 || tied {
        return None;
    }
    if grid {
        let mut refined = h;
        if refine_grid(&mut refined, image, indexed, id, rotation, FULL_FIT) {
            let refined_quad = [
                project(&refined, 0.0, 0.0),
                project(&refined, 1.0, 0.0),
                project(&refined, 1.0, 1.0),
                project(&refined, 0.0, 1.0),
            ];
            if let Some(verified) = decode(
                &refined_quad,
                image,
                indexed,
                family_index,
                sharpening,
                false,
            ) {
                if verified.tag_id == id && verified.hamming <= best {
                    return Some(verified);
                }
            }
        }
    }
    let mut out = Detection {
        family_index,
        tag_id: id,
        hamming: best,
        rotation,
        decision_margin: margin,
        ..Detection::default()
    };
    let corners = [
        Point { x: 0.0, y: 1.0 },
        Point { x: 1.0, y: 1.0 },
        Point { x: 1.0, y: 0.0 },
        Point { x: 0.0, y: 0.0 },
    ];
    let mut canonical = [Point::default(); 4];
    for k in 0..4 {
        let p = rotate(corners[k], rotation);
        canonical[k] = project(&h, p.x, p.y);
        out.corners[2 * k] = canonical[k].x + 0.5;
        out.corners[2 * k + 1] = canonical[k].y + 0.5;
    }
    let center = project(&h, 0.5, 0.5);
    out.center = [center.x + 0.5, center.y + 0.5];
    let unit = [canonical[3], canonical[2], canonical[1], canonical[0]];
    let ch = homography(&unit)?;
    let mut normalized = [
        ch[0] * 0.5,
        ch[1] * 0.5,
        ch[2] + 0.5 * (ch[0] + ch[1]),
        ch[3] * 0.5,
        ch[4] * 0.5,
        ch[5] + 0.5 * (ch[3] + ch[4]),
        ch[6] * 0.5,
        ch[7] * 0.5,
        ch[8] + 0.5 * (ch[6] + ch[7]),
    ];
    for j in 0..3 {
        normalized[j] += 0.5 * normalized[6 + j];
        normalized[3 + j] += 0.5 * normalized[6 + j];
    }
    let scale = normalized[8];
    if !scale.is_finite() || scale.abs() < 1e-9 {
        return None;
    }
    for j in 0..9 {
        out.homography[j] = normalized[j] / scale;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::rotate;
    use crate::geometry::Point;

    #[test]
    fn rotated_cell_centres_are_cell_centres_for_power_of_two_widths() {
        let width = 8;
        let centre = |i: i32| (f64::from(i) + 0.5) / f64::from(width);
        for x in -1..=width {
            for y in -1..=width {
                let (mut cx, mut cy) = (x, y);
                for r in 0..4 {
                    let p = rotate(
                        Point {
                            x: centre(x),
                            y: centre(y),
                        },
                        r,
                    );
                    assert_eq!(
                        (p.x.to_bits(), p.y.to_bits()),
                        (centre(cx).to_bits(), centre(cy).to_bits())
                    );
                    (cx, cy) = (width - 1 - cy, cx);
                }
            }
        }
    }
}
