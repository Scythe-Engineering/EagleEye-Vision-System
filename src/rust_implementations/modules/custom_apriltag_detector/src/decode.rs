//! Independent frozen Eagle Tags scalar decoding and bounded template grid fit.

use crate::detection::Detection;
use crate::families::{Family, IndexedFamily};
use crate::geometry::{dot, homography, min_f64, norm, project, Point, Quad, H};
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

/// Fit independent template edges for two iterations, restoring the input on failure.
#[allow(clippy::needless_range_loop)] // Indexed rows preserve the oracle's pivot/accumulation order.
fn refine_grid(h: &mut H, image: &Image<'_>, family: &Family, id: i32, rotation: i32) -> bool {
    if family.total != family.width + 2 || family.reversed {
        return false;
    }
    for b in 0..family.nbits {
        if family.x[b] < 0
            || family.y[b] < 0
            || family.x[b] >= family.width
            || family.y[b] >= family.width
        {
            return false;
        }
    }
    let white = |x: i32, y: i32| -> i32 {
        for b in 0..family.nbits {
            if family.x[b] == x && family.y[b] == y {
                return ((family.codes[id as usize] >> (family.nbits - 1 - b)) & 1) as i32;
            }
        }
        i32::from(x < 0 || y < 0 || x >= family.width || y >= family.width)
    };
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
    for _iteration in 0..2 {
        let mut system = [[0.0f64; 9]; 8];
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
                    let gradient = |s: f64| -> f64 {
                        f64::from(after - before)
                            * (image.sample(Point {
                                x: n.x.mul_add(s + aperture, p.x),
                                y: n.y.mul_add(s + aperture, p.y),
                            }) - image.sample(Point {
                                x: n.x.mul_add(s - aperture, p.x),
                                y: n.y.mul_add(s - aperture, p.y),
                            }))
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
                        let mut t = 0.125;
                        while t <= 2.0 {
                            let s = offset + f64::from(direction) * t;
                            let g = gradient(s);
                            if !g.is_finite() || g <= 0.0 {
                                break;
                            }
                            mass += g;
                            moment = s.mul_add(g, moment);
                            t += 0.125;
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
                    ];
                    // Frozen C++ normal-equation accumulation: same rows/order; solver unchanged.
                    for a in 0..8 {
                        for b in 0..8 {
                            system[a][b] = j[a].mul_add(j[b], system[a][b]);
                        }
                        system[a][8] = j[a].mul_add(residual, system[a][8]);
                    }
                    count += 1;
                }
            }
        }
        if count < 16 {
            *h = initial;
            return false;
        }
        for a in 0..8 {
            let mut pivot = a;
            for b in a + 1..8 {
                if system[b][a].abs() > system[pivot][a].abs() {
                    pivot = b;
                }
            }
            for b in a..=8 {
                let value = system[a][b];
                system[a][b] = system[pivot][b];
                system[pivot][b] = value;
            }
            let scale = system[a][a];
            if scale.abs() < 1e-10 {
                *h = initial;
                return false;
            }
            for b in a..=8 {
                system[a][b] /= scale;
            }
            for row in 0..8 {
                if row != a {
                    let factor = system[row][a];
                    for b in a..=8 {
                        system[row][b] -= factor * system[a][b];
                    }
                }
            }
        }
        let mut next = *h;
        for a in 0..8 {
            next[a] += system[a][8];
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
    let payload = |x: i32, y: i32| -> bool { (0..f.nbits).any(|b| f.x[b] == x && f.y[b] == y) };
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
    // ponytail: scalar illumination; fit spatial planes only if real clips demand it.
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
            let x = f64::from(f.x[b]) + 0.5;
            let y = f64::from(f.y[b]) + 0.5;
            let mut v = sample(x, y, r);
            if !v.is_finite() {
                valid = false;
                break;
            }
            if sharpening != 0.0 {
                let mut neighbor = 0.0;
                let mut all = true;
                for d in [
                    Point { x: 1.0, y: 0.0 },
                    Point { x: -1.0, y: 0.0 },
                    Point { x: 0.0, y: 1.0 },
                    Point { x: 0.0, y: -1.0 },
                ] {
                    let n = sample(x + d.x, y + d.y, r);
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
        if refine_grid(&mut refined, image, f, id, rotation) {
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
    use super::*;
    use crate::family_data::FAMILIES;

    // Rasterize physical cells, not decoder sample predictions. All evidence is pixels.
    fn render(family: &Family, id: usize, rotation: i32) -> (Vec<u8>, usize, Quad) {
        let cell = 8;
        let origin = 48;
        let size = (origin * 2 + family.width * cell) as usize;
        let mut pixels = vec![0; size * size];
        for y in 0..size {
            for x in 0..size {
                let mut cx = (x as i32 - origin).div_euclid(cell);
                let mut cy = (y as i32 - origin).div_euclid(cell);
                for _ in 0..rotation {
                    (cx, cy) = (cy, family.width - 1 - cx);
                }
                let inside = cx >= 0 && cy >= 0 && cx < family.width && cy < family.width;
                let mut white = if family.reversed { inside } else { !inside };
                for b in 0..family.nbits {
                    if family.x[b] == cx && family.y[b] == cy {
                        white = (family.codes[id] >> (family.nbits - 1 - b)) & 1 != 0;
                        break;
                    }
                }
                pixels[y * size + x] = if white { 230 } else { 20 };
            }
        }
        let lo = f64::from(origin) - 0.5;
        let hi = f64::from(origin + family.width * cell) - 0.5;
        (
            pixels,
            size,
            [
                Point { x: lo, y: lo },
                Point { x: hi, y: lo },
                Point { x: hi, y: hi },
                Point { x: lo, y: hi },
            ],
        )
    }

    #[test]
    fn physical_normal_and_reversed_all_rotations() {
        for (fi, family) in FAMILIES.iter().enumerate() {
            let indexed = IndexedFamily::new(family);
            for rotation in 0..4 {
                let (pixels, size, quad) = render(family, 0, rotation);
                let image = Image {
                    pixels: &pixels,
                    width: size,
                    height: size,
                    stride: size,
                    source_map: None,
                    source_width: 0,
                    source_height: 0,
                };
                for grid in [false, true] {
                    let detection = decode(&quad, &image, &indexed, fi as u32, 0.25, grid)
                        .unwrap_or_else(|| {
                            panic!("{} rotation {rotation} grid {grid}", family.name)
                        });
                    assert_eq!(detection.tag_id, 0);
                    assert_eq!(detection.hamming, 0);
                    assert_eq!(detection.rotation, rotation);
                    assert_eq!(detection.family_index, fi as u32);
                    assert!(detection.decision_margin > 15.0);
                    assert!(detection.homography.iter().all(|v| v.is_finite()));
                    // Normalized homography includes the same native half-pixel shift.
                    for (k, uv) in [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
                        .iter()
                        .enumerate()
                    {
                        let p = project(&detection.homography, uv.0, uv.1);
                        assert!((p.x - detection.corners[2 * k]).abs() < 1e-8);
                        assert!((p.y - detection.corners[2 * k + 1]).abs() < 1e-8);
                    }
                }
            }
        }
    }

    #[test]
    fn degenerate_and_unobserved_evidence_rejected() {
        let family = &FAMILIES[0];
        let indexed = IndexedFamily::new(family);
        let (pixels, size, quad) = render(family, 0, 0);
        let mut image = Image {
            pixels: &pixels,
            width: size,
            height: size,
            stride: size,
            source_map: None,
            source_width: size as u32,
            source_height: size as u32,
        };
        assert!(decode(&[Point::default(); 4], &image, &indexed, 0, 0.0, false).is_none());
        // One missing exterior side cannot borrow reference support from the others.
        image.source_map = Some([1.0, 0.0, -quad[0].x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(decode(&quad, &image, &indexed, 0, 0.0, false).is_none());
        image.source_map = Some([0.0; 9]);
        assert!(decode(&quad, &image, &indexed, 0, 0.0, true).is_none());
        let mut h = homography(&quad).unwrap();
        let initial = h;
        assert!(!refine_grid(&mut h, &image, family, 0, 0));
        assert_eq!(h, initial);
    }
}
