//! PyO3 extension: pinhole projection, per-tag ROIs, and warp-based tag verification for benchmarks.
//!
//! Matches the Python ``FastTemporalCustomAprilTagDetector`` geometry and ``warp_contrast`` gate
//! (no lens distortion; zero-distortion pinhole projection).

use ndarray::{Array2, ArrayView2};
use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type RowMat4 = [[f64; 4]; 4];

const BLENDER_AXES_TO_CV: RowMat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

#[pymodule]
fn apriltag_fast_temporal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastTemporalCustomRustCore>()?;
    Ok(())
}

#[derive(Clone)]
struct TagLayout {
    family: String,
    id: i32,
    corners_world: [[f64; 3]; 4],
}

#[pyclass]
pub struct FastTemporalCustomRustCore {
    intrinsics: [f64; 4],
    tags: Vec<TagLayout>,
    pose_blender_row_major: Option<[f64; 16]>,
    padding_factor: f64,
    max_regions: usize,
    min_region_size_px: i32,
    merge_overlapping_rois: bool,
    warp_canonical_size: i32,
    warp_min_border_delta: f64,
    warp_min_inner_std: f64,
}

#[pymethods]
impl FastTemporalCustomRustCore {
    #[new]
    fn new() -> Self {
        Self {
            intrinsics: [0.0; 4],
            tags: Vec::new(),
            pose_blender_row_major: None,
            padding_factor: 2.0,
            max_regions: 28,
            min_region_size_px: 28,
            merge_overlapping_rois: true,
            warp_canonical_size: 48,
            warp_min_border_delta: 6.0,
            warp_min_inner_std: 7.0,
        }
    }

    fn set_intrinsics(&mut self, fx: f64, fy: f64, cx: f64, cy: f64) {
        self.intrinsics = [fx, fy, cx, cy];
    }

    fn set_layout(
        &mut self,
        tag_families: Vec<String>,
        tag_ids: Vec<i32>,
        corners_world_flat: Vec<f64>,
    ) -> PyResult<()> {
        if tag_families.len() != tag_ids.len() {
            return Err(PyValueError::new_err("tag_families and tag_ids length mismatch"));
        }
        let n = tag_ids.len();
        if corners_world_flat.len() != n * 12 {
            return Err(PyValueError::new_err(
                "corners_world_flat must have 12 floats per tag",
            ));
        }
        let mut tags = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 12;
            let mut c = [[0.0f64; 3]; 4];
            for k in 0..4 {
                c[k][0] = corners_world_flat[base + k * 3];
                c[k][1] = corners_world_flat[base + k * 3 + 1];
                c[k][2] = corners_world_flat[base + k * 3 + 2];
            }
            tags.push(TagLayout {
                family: tag_families[i].clone(),
                id: tag_ids[i],
                corners_world: c,
            });
        }
        self.tags = tags;
        Ok(())
    }

    fn set_pose_blender_row_major(&mut self, camera_matrix_world: Vec<f64>) -> PyResult<()> {
        if camera_matrix_world.len() != 16 {
            return Err(PyValueError::new_err("camera_matrix_world must have 16 floats row-major"));
        }
        let mut m = [0.0f64; 16];
        m.copy_from_slice(&camera_matrix_world[..16]);
        self.pose_blender_row_major = Some(m);
        Ok(())
    }

    fn set_roi_params(
        &mut self,
        padding_factor: f64,
        max_regions: usize,
        min_region_size_px: i32,
        merge_overlapping_rois: bool,
    ) {
        self.padding_factor = padding_factor;
        self.max_regions = max_regions.max(1);
        self.min_region_size_px = min_region_size_px.max(1);
        self.merge_overlapping_rois = merge_overlapping_rois;
    }

    fn set_warp_params(
        &mut self,
        warp_canonical_size: i32,
        warp_min_border_delta: f64,
        warp_min_inner_std: f64,
    ) {
        self.warp_canonical_size = warp_canonical_size.max(8);
        self.warp_min_border_delta = warp_min_border_delta;
        self.warp_min_inner_std = warp_min_inner_std;
    }

    /// Returns ``(tag_family, tag_id, flat_corners_xy)`` for tags that pass warp verification.
    fn process_frame<'py>(
        &self,
        py: Python<'py>,
        gray: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Vec<(String, i32, Vec<f64>)>> {
        let _ = py;
        let pose = self
            .pose_blender_row_major
            .ok_or_else(|| PyValueError::new_err("pose not set"))?;
        let storage = gray.to_owned_array();
        let view = storage.view();
        if view.ndim() != 2 {
            return Err(PyValueError::new_err("gray must be 2D"));
        }
        let height = view.shape()[0];
        let width = view.shape()[1];
        if height < 2 || width < 2 {
            return Err(PyValueError::new_err("image too small"));
        }
        let fx = self.intrinsics[0];
        let fy = self.intrinsics[1];
        let cx = self.intrinsics[2];
        let cy = self.intrinsics[3];
        let cam_from_world = camera_from_world_blender(&pose)?;
        let mut scored: Vec<(f64, String, i32, [[f64; 2]; 4])> = Vec::new();
        let margin = 0.02_f64 * (width.max(height) as f64);
        for tag in &self.tags {
            let mut corners_img = [[0.0f64; 2]; 4];
            let mut ok = true;
            for (k, pw) in tag.corners_world.iter().enumerate() {
                let Some(pi) = project_world_to_pixel(pw, &cam_from_world, fx, fy, cx, cy) else {
                    ok = false;
                    break;
                };
                corners_img[k] = pi;
            }
            if !ok {
                continue;
            }
            let mut inside = 0_i32;
            for c in &corners_img {
                if c[0] >= -margin
                    && c[0] < width as f64 + margin
                    && c[1] >= -margin
                    && c[1] < height as f64 + margin
                {
                    inside += 1;
                }
            }
            if inside < 3 {
                continue;
            }
            let area = quad_area(&corners_img);
            if area < 4.0 {
                continue;
            }
            scored.push((area, tag.family.clone(), tag.id, corners_img));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.max_regions);
        let mut out = Vec::new();
        for (_area, family, id, corners_img) in scored {
            let (rx, ry, rw, rh) = axis_aligned_roi_from_quad(
                &corners_img,
                width,
                height,
                self.padding_factor,
                self.min_region_size_px,
            );
            if rw < 2 || rh < 2 {
                continue;
            }
            let quad_local: [[f64; 2]; 4] = std::array::from_fn(|k| {
                [
                    corners_img[k][0] - rx as f64,
                    corners_img[k][1] - ry as f64,
                ]
            });
            let row_start = ry as usize;
            let row_end = (ry + rh).min(height as i32).max(ry + 1) as usize;
            let col_start = rx as usize;
            let col_end = (rx + rw).min(width as i32).max(rx + 1) as usize;
            if row_end <= row_start || col_end <= col_start {
                continue;
            }
            let crop = view.slice(ndarray::s![row_start..row_end, col_start..col_end]);
            if crop.shape()[0] < 2 || crop.shape()[1] < 2 {
                continue;
            }
            if !verify_warp_border_contrast(
                crop,
                &quad_local,
                self.warp_canonical_size,
                self.warp_min_border_delta,
                self.warp_min_inner_std,
            ) {
                continue;
            }
            let mut flat = Vec::with_capacity(8);
            for c in &corners_img {
                flat.push(c[0]);
                flat.push(c[1]);
            }
            out.push((family, id, flat));
        }
        Ok(out)
    }

    fn merged_roi_coverage_fraction<'py>(
        &self,
        py: Python<'py>,
        gray: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<(f64, f64)> {
        let _ = py;
        let pose = self
            .pose_blender_row_major
            .ok_or_else(|| PyValueError::new_err("pose not set"))?;
        let storage = gray.to_owned_array();
        let view = storage.view();
        let height = view.shape()[0];
        let width = view.shape()[1];
        let fx = self.intrinsics[0];
        let fy = self.intrinsics[1];
        let cx = self.intrinsics[2];
        let cy = self.intrinsics[3];
        let cam_from_world = camera_from_world_blender(&pose)?;
        let margin = 0.02_f64 * (width.max(height) as f64);
        let mut scored: Vec<(f64, [i32; 4])> = Vec::new();
        for tag in &self.tags {
            let mut corners_img = [[0.0f64; 2]; 4];
            let mut ok = true;
            for (k, pw) in tag.corners_world.iter().enumerate() {
                let Some(pi) = project_world_to_pixel(pw, &cam_from_world, fx, fy, cx, cy) else {
                    ok = false;
                    break;
                };
                corners_img[k] = pi;
            }
            if !ok {
                continue;
            }
            let mut inside = 0_i32;
            for c in &corners_img {
                if c[0] >= -margin
                    && c[0] < width as f64 + margin
                    && c[1] >= -margin
                    && c[1] < height as f64 + margin
                {
                    inside += 1;
                }
            }
            if inside < 3 {
                continue;
            }
            let area = quad_area(&corners_img);
            if area < 4.0 {
                continue;
            }
            let (rx, ry, rw, rh) = axis_aligned_roi_from_quad(
                &corners_img,
                width,
                height,
                self.padding_factor,
                self.min_region_size_px,
            );
            if rw >= 2 && rh >= 2 {
                scored.push((area, [rx, ry, rx + rw, ry + rh]));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.max_regions);
        let rois: Vec<[i32; 4]> = scored.into_iter().map(|(_, bbox)| bbox).collect();
        let merged = if self.merge_overlapping_rois {
            merge_axis_aligned_boxes(&rois)
        } else {
            rois.clone()
        };
        let image_area = (width * height).max(1) as f64;
        let roi_area: f64 = merged
            .iter()
            .map(|b| {
                let w = (b[2] - b[0]).max(0) as f64;
                let h = (b[3] - b[1]).max(0) as f64;
                w * h
            })
            .sum();
        let coverage = (roi_area / image_area).min(1.0);
        Ok((coverage, rois.len() as f64))
    }
}

fn mat4_from_row_major16(slice: &[f64; 16]) -> RowMat4 {
    [
        [slice[0], slice[1], slice[2], slice[3]],
        [slice[4], slice[5], slice[6], slice[7]],
        [slice[8], slice[9], slice[10], slice[11]],
        [slice[12], slice[13], slice[14], slice[15]],
    ]
}

fn mat4_mul(a: &RowMat4, b: &RowMat4) -> RowMat4 {
    let mut product = [[0.0f64; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            let mut sum = 0.0f64;
            for k in 0..4 {
                sum += a[row][k] * b[k][col];
            }
            product[row][col] = sum;
        }
    }
    product
}

fn mat4_mul_vec4(matrix: &RowMat4, vector: [f64; 4]) -> [f64; 4] {
    let mut out = [0.0f64; 4];
    for row in 0..4 {
        let mut sum = 0.0f64;
        for col in 0..4 {
            sum += matrix[row][col] * vector[col];
        }
        out[row] = sum;
    }
    out
}

fn mat4_try_inverse(mut matrix: RowMat4) -> Option<RowMat4> {
    let mut inverse: RowMat4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for pivot_column in 0..4 {
        let mut pivot_row = pivot_column;
        let mut best_abs = matrix[pivot_row][pivot_column].abs();
        for row in (pivot_column + 1)..4 {
            let candidate_abs = matrix[row][pivot_column].abs();
            if candidate_abs > best_abs {
                best_abs = candidate_abs;
                pivot_row = row;
            }
        }
        if !best_abs.is_finite() || best_abs < 1e-12 {
            return None;
        }
        if pivot_row != pivot_column {
            matrix.swap(pivot_row, pivot_column);
            inverse.swap(pivot_row, pivot_column);
        }
        let divisor = matrix[pivot_column][pivot_column];
        for column in 0..4 {
            matrix[pivot_column][column] /= divisor;
            inverse[pivot_column][column] /= divisor;
        }
        for row in 0..4 {
            if row == pivot_column {
                continue;
            }
            let factor = matrix[row][pivot_column];
            if factor.abs() < 1e-15 {
                continue;
            }
            for column in 0..4 {
                matrix[row][column] -= factor * matrix[pivot_column][column];
                inverse[row][column] -= factor * inverse[pivot_column][column];
            }
        }
    }
    Some(inverse)
}

fn camera_from_world_blender(camera_matrix_world: &[f64; 16]) -> PyResult<RowMat4> {
    let world_from_blender_camera = mat4_from_row_major16(camera_matrix_world);
    let world_from_cv_camera = mat4_mul(&world_from_blender_camera, &BLENDER_AXES_TO_CV);
    mat4_try_inverse(world_from_cv_camera).ok_or_else(|| {
        PyValueError::new_err("singular world_from_cv_camera transform; cannot invert for projection")
    })
}

fn project_world_to_pixel(
    p_world: &[f64; 3],
    camera_from_world: &RowMat4,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> Option<[f64; 2]> {
    let homogeneous = mat4_mul_vec4(
        camera_from_world,
        [p_world[0], p_world[1], p_world[2], 1.0],
    );
    let depth = homogeneous[2];
    if !depth.is_finite() || depth <= 1e-9 {
        return None;
    }
    let x = homogeneous[0] / depth;
    let y = homogeneous[1] / depth;
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some([fx * x + cx, fy * y + cy])
}

fn quad_area(corners: &[[f64; 2]; 4]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..4 {
        let j = (i + 1) % 4;
        s += corners[i][0] * corners[j][1] - corners[j][0] * corners[i][1];
    }
    0.5 * s.abs()
}

fn axis_aligned_roi_from_quad(
    corners: &[[f64; 2]; 4],
    image_width: usize,
    image_height: usize,
    padding_factor: f64,
    min_side_px: i32,
) -> (i32, i32, i32, i32) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in corners {
        min_x = min_x.min(c[0]);
        min_y = min_y.min(c[1]);
        max_x = max_x.max(c[0]);
        max_y = max_y.max(c[1]);
    }
    let center_x = 0.5 * (min_x + max_x);
    let center_y = 0.5 * (min_y + max_y);
    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    let max_span = span_x.max(span_y);
    let pad = max_span * (padding_factor - 1.0).max(0.0) * 0.5;
    let half_w = (min_side_px as f64 * 0.5).max(span_x * 0.5 + pad);
    let half_h = (min_side_px as f64 * 0.5).max(span_y * 0.5 + pad);
    let mut x1 = (center_x - half_w).floor() as i32;
    let mut y1 = (center_y - half_h).floor() as i32;
    let mut x2 = (center_x + half_w).ceil() as i32;
    let mut y2 = (center_y + half_h).ceil() as i32;
    let iw = image_width as i32;
    let ih = image_height as i32;
    x1 = x1.clamp(0, (iw - 1).max(0));
    y1 = y1.clamp(0, (ih - 1).max(0));
    x2 = x2.max(x1 + 1).min(iw);
    y2 = y2.max(y1 + 1).min(ih);
    (x1, y1, x2 - x1, y2 - y1)
}

fn merge_axis_aligned_boxes(regions: &[[i32; 4]]) -> Vec<[i32; 4]> {
    if regions.is_empty() {
        return Vec::new();
    }
    let mut boxes: Vec<[i32; 4]> = regions.to_vec();
    let mut changed = true;
    while changed {
        changed = false;
        let mut merged: Vec<[i32; 4]> = Vec::new();
        for b in boxes {
            let mut absorbed = false;
            for m in merged.iter_mut() {
                let horizontal_gap = b[0] > m[2] || b[2] < m[0];
                let vertical_gap = b[1] > m[3] || b[3] < m[1];
                if !(horizontal_gap || vertical_gap) {
                    m[0] = m[0].min(b[0]);
                    m[1] = m[1].min(b[1]);
                    m[2] = m[2].max(b[2]);
                    m[3] = m[3].max(b[3]);
                    changed = true;
                    absorbed = true;
                    break;
                }
            }
            if !absorbed {
                merged.push(b);
            }
        }
        boxes = merged;
    }
    boxes
}

fn homography_from_quad_to_square(
    src: &[[f64; 2]; 4],
    size: i32,
) -> Option<[[f64; 3]; 3]> {
    let s = (size - 1) as f64;
    let dst = [
        [0.0, 0.0],
        [s, 0.0],
        [s, s],
        [0.0, s],
    ];
    dlt_homography(src, &dst)
}

fn solve_linear_system_8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    const N: usize = 8;
    for pivot_column in 0..N {
        let mut pivot_row = pivot_column;
        let mut best_abs = a[pivot_row][pivot_column].abs();
        for row in (pivot_column + 1)..N {
            let candidate_abs = a[row][pivot_column].abs();
            if candidate_abs > best_abs {
                best_abs = candidate_abs;
                pivot_row = row;
            }
        }
        if best_abs < 1e-12 {
            return None;
        }
        if pivot_row != pivot_column {
            a.swap(pivot_row, pivot_column);
            b.swap(pivot_row, pivot_column);
        }
        let divisor = a[pivot_column][pivot_column];
        for column in pivot_column..N {
            a[pivot_column][column] /= divisor;
        }
        b[pivot_column] /= divisor;
        for row in 0..N {
            if row == pivot_column {
                continue;
            }
            let factor = a[row][pivot_column];
            if factor.abs() < 1e-15 {
                continue;
            }
            for column in pivot_column..N {
                a[row][column] -= factor * a[pivot_column][column];
            }
            b[row] -= factor * b[pivot_column];
        }
    }
    Some(b)
}

fn determinant_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn invert_3x3_row_major(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = determinant_3x3(&matrix);
    if !det.is_finite() || det.abs() < 1e-14 {
        return None;
    }
    let inv_det = 1.0 / det;
    let cofactor = |r: usize, c: usize| -> f64 {
        let rows: [usize; 2] = if r == 0 { [1, 2] } else if r == 1 { [0, 2] } else { [0, 1] };
        let cols: [usize; 2] = if c == 0 { [1, 2] } else if c == 1 { [0, 2] } else { [0, 1] };
        let minor = matrix[rows[0]][cols[0]] * matrix[rows[1]][cols[1]]
            - matrix[rows[0]][cols[1]] * matrix[rows[1]][cols[0]];
        let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
        sign * minor
    };
    let mut inverse = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            inverse[col][row] = cofactor(row, col) * inv_det;
        }
    }
    Some(inverse)
}

fn mat3_mul_homogeneous(matrix: [[f64; 3]; 3], x: f64, y: f64) -> [f64; 3] {
    let z = 1.0f64;
    [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    ]
}

fn dlt_homography(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<[[f64; 3]; 3]> {
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (x, y) = (src[i][0], src[i][1]);
        let (xp, yp) = (dst[i][0], dst[i][1]);
        let r = i * 2;
        a[r][0] = x;
        a[r][1] = y;
        a[r][2] = 1.0;
        a[r][6] = -xp * x;
        a[r][7] = -xp * y;
        b[r] = xp;
        let r2 = r + 1;
        a[r2][3] = x;
        a[r2][4] = y;
        a[r2][5] = 1.0;
        a[r2][6] = -yp * x;
        a[r2][7] = -yp * y;
        b[r2] = yp;
    }
    let solution = solve_linear_system_8(a, b)?;
    let h00 = solution[0];
    let h01 = solution[1];
    let h02 = solution[2];
    let h10 = solution[3];
    let h11 = solution[4];
    let h12 = solution[5];
    let h20 = solution[6];
    let h21 = solution[7];
    let h22 = 1.0;
    if ![h00, h01, h02, h10, h11, h12, h20, h21, h22]
        .iter()
        .all(|coefficient| coefficient.is_finite())
    {
        return None;
    }
    Some([
        [h00, h01, h02],
        [h10, h11, h12],
        [h20, h21, h22],
    ])
}

fn bilinear_u8(img: ArrayView2<u8>, u: f64, v: f64) -> f64 {
    let h = img.shape()[0] as i32;
    let w = img.shape()[1] as i32;
    if h < 2 || w < 2 {
        return 0.0;
    }
    let uu = u.clamp(0.0, (w - 1) as f64);
    let vv = v.clamp(0.0, (h - 1) as f64);
    let x0 = uu.floor();
    let y0 = vv.floor();
    if !(x0.is_finite() && y0.is_finite()) {
        return 0.0;
    }
    let x0i = x0 as i32;
    let y0i = y0 as i32;
    let x1 = (x0i + 1).min(w - 1);
    let y1 = (y0i + 1).min(h - 1);
    let x0u = x0i.clamp(0, w - 1) as usize;
    let y0u = y0i.clamp(0, h - 1) as usize;
    let x1u = x1.clamp(0, w - 1) as usize;
    let y1u = y1.clamp(0, h - 1) as usize;
    let fx = uu - x0i as f64;
    let fy = vv - y0i as f64;
    let i00 = img[[y0u, x0u]] as f64;
    let i01 = img[[y0u, x1u]] as f64;
    let i10 = img[[y1u, x0u]] as f64;
    let i11 = img[[y1u, x1u]] as f64;
    let a = i00 * (1.0 - fx) + i01 * fx;
    let b = i10 * (1.0 - fx) + i11 * fx;
    a * (1.0 - fy) + b * fy
}

fn warp_perspective_bilinear(
    crop: ArrayView2<u8>,
    quad_local: &[[f64; 2]; 4],
    out_size: i32,
) -> Option<Array2<f64>> {
    let side = out_size as usize;
    let h = homography_from_quad_to_square(quad_local, out_size)?;
    let h_inv = invert_3x3_row_major(h)?;
    let mut warped = Array2::<f64>::zeros((side, side));
    for oy in 0..side {
        for ox in 0..side {
            let src_h = mat3_mul_homogeneous(h_inv, ox as f64, oy as f64);
            if src_h[2].abs() < 1e-12 {
                continue;
            }
            let su = src_h[0] / src_h[2];
            let sv = src_h[1] / src_h[2];
            if !(su.is_finite() && sv.is_finite()) {
                warped[[oy, ox]] = 0.0;
                continue;
            }
            warped[[oy, ox]] = bilinear_u8(crop, su, sv);
        }
    }
    Some(warped)
}

fn verify_warp_border_contrast(
    crop: ArrayView2<u8>,
    quad_local: &[[f64; 2]; 4],
    canonical_size: i32,
    min_border_delta: f64,
    min_inner_std: f64,
) -> bool {
    let side = canonical_size as usize;
    if side < 8 {
        return false;
    }
    let Some(warped) = warp_perspective_bilinear(crop, quad_local, canonical_size) else {
        return false;
    };
    let border_thickness = (canonical_size / 16).max(2) as usize;
    let mut border_vals: Vec<f64> = Vec::new();
    let mut inner_vals: Vec<f64> = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let v = warped[[y, x]];
            let is_border = y < border_thickness
                || y + border_thickness >= side
                || x < border_thickness
                || x + border_thickness >= side;
            if is_border {
                border_vals.push(v);
            } else {
                inner_vals.push(v);
            }
        }
    }
    if border_vals.len() < 8 || inner_vals.len() < 8 {
        return false;
    }
    let border_mean: f64 = border_vals.iter().sum::<f64>() / border_vals.len() as f64;
    let inner_mean: f64 = inner_vals.iter().sum::<f64>() / inner_vals.len() as f64;
    let inner_var: f64 = inner_vals
        .iter()
        .map(|v| {
            let d = v - inner_mean;
            d * d
        })
        .sum::<f64>()
        / inner_vals.len() as f64;
    let inner_std = inner_var.sqrt();
    (border_mean > inner_mean + min_border_delta) && (inner_std > min_inner_std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn quad_area_unit_square() {
        let c = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!((quad_area(&c) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn dlt_and_warp_smoke() {
        let quad_local = [[2.0f64, 2.0], [47.0, 3.0], [46.0, 47.0], [3.0, 46.0]];
        let s = 47.0f64;
        let dst = [[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]];
        let h = dlt_homography(&quad_local, &dst).expect("h");
        let _ = invert_3x3_row_major(h).expect("inv");
        let mut img = Array2::<u8>::zeros((50, 50));
        for y in 5..45 {
            for x in 5..45 {
                img[[y, x]] = 200;
            }
        }
        let v = img.view();
        let _ = verify_warp_border_contrast(v, &quad_local, 48, 1.0, 1.0);
    }
}
