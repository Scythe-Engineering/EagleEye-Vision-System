use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A simple temporal acceleration module
#[pymodule]
fn temporal_acceleration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TemporalAcceleration>()?;
    Ok(())
}

/// Main class for temporal acceleration functionality
#[pyclass]
pub struct TemporalAcceleration {
    camera_matrix: [f32; 9],
    distortion_coefficients: Vec<f32>,
    apriltag_ids: Vec<i32>,
    apriltag_corners: Vec<[f32; 12]>, // 4 corners * 3 coords
    apriltag_centers: Vec<[f32; 3]>,
    padding_factor: f32,
    max_regions: usize,
    min_region_size_px: i32,
    max_detection_distance_m: f32,
    last_pose_world_from_camera: Option<[f32; 16]>,
}

#[pymethods]
impl TemporalAcceleration {
    #[new]
    #[pyo3(signature = (
        camera_matrix,
        distortion_coefficients,
        apriltag_ids,
        apriltag_corners,
        apriltag_centers,
        padding_factor=0.35,
        max_regions=20,
        min_region_size_px=16,
        max_detection_distance_m=0.0
    ))]
    fn new(
        _py: Python,
        camera_matrix: Vec<f32>,
        distortion_coefficients: Vec<f32>,
        apriltag_ids: Vec<i32>,
        apriltag_corners: Vec<f32>,
        apriltag_centers: Vec<f32>,
        padding_factor: f32,
        max_regions: usize,
        min_region_size_px: i32,
        max_detection_distance_m: f32,
    ) -> PyResult<Self> {
        if camera_matrix.len() != 9 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "camera_matrix must have 9 elements (row-major 3x3)",
            ));
        }
        if apriltag_ids.len() * 12 != apriltag_corners.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "apriltag_corners must have 12 floats per tag (4x3)",
            ));
        }
        if apriltag_ids.len() * 3 != apriltag_centers.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "apriltag_centers must have 3 floats per tag",
            ));
        }

        let mut cm: [f32; 9] = [0.0; 9];
        cm.copy_from_slice(&camera_matrix[..9]);

        let mut corners: Vec<[f32; 12]> = Vec::with_capacity(apriltag_ids.len());
        for i in 0..apriltag_ids.len() {
            let start = i * 12;
            let mut arr: [f32; 12] = [0.0; 12];
            arr.copy_from_slice(&apriltag_corners[start..start + 12]);
            corners.push(arr);
        }

        let mut centers: Vec<[f32; 3]> = Vec::with_capacity(apriltag_ids.len());
        for i in 0..apriltag_ids.len() {
            let start = i * 3;
            let mut arr: [f32; 3] = [0.0; 3];
            arr.copy_from_slice(&apriltag_centers[start..start + 3]);
            centers.push(arr);
        }

        Ok(Self {
            camera_matrix: cm,
            distortion_coefficients,
            apriltag_ids,
            apriltag_corners: corners,
            apriltag_centers: centers,
            padding_factor,
            max_regions,
            min_region_size_px,
            max_detection_distance_m: max_detection_distance_m.max(0.0),
            last_pose_world_from_camera: None,
        })
    }

    fn back_propagate_input(&mut self, input_transform: Vec<f32>) -> PyResult<()> {
        if input_transform.len() != 16 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Transform must be 4x4 (16 elements)",
            ));
        }
        if !input_transform.iter().all(|&x| x.is_finite()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Transform contains non-finite values",
            ));
        }
        let mut mat: [f32; 16] = [0.0; 16];
        mat.copy_from_slice(&input_transform[..16]);
        self.last_pose_world_from_camera = Some(mat);
        Ok(())
    }

    /// Return paired padded quadrilaterals and axis-aligned regions, nearest first.
    ///
    /// An empty quadrilateral list with one full-frame region requests an unwarped
    /// fallback crop when no camera pose or projected tag region is available.
    fn process_frame(
        &self,
        width: usize,
        height: usize,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<i32>>)> {
        let mut region_distances: Vec<(f32, [i32; 4], [[f32; 2]; 4])> = Vec::new();

        // Without a pose, leave the full-frame crop unwarped.
        let Some(world_from_camera) = self.last_pose_world_from_camera else {
            let regions = vec![[0, 0, width as i32, height as i32]];
            return Ok((vec![], regions.iter().map(|r| r.to_vec()).collect()));
        };

        let world_to_camera = invert_se3(&world_from_camera);
        let (r_wc, t_wc) = decompose_rt(&world_to_camera);

        let fx = self.camera_matrix[0];
        let fy = self.camera_matrix[4];
        let cx = self.camera_matrix[2];
        let cy = self.camera_matrix[5];

        for i in 0..self.apriltag_ids.len() {
            let corners_world = &self.apriltag_corners[i];
            let center_world = &self.apriltag_centers[i];

            // Compute normal in world
            let e1 = [
                corners_world[3] - corners_world[0],
                corners_world[4] - corners_world[1],
                corners_world[5] - corners_world[2],
            ];
            let e2 = [
                corners_world[6] - corners_world[0],
                corners_world[7] - corners_world[1],
                corners_world[8] - corners_world[2],
            ];
            let normal_world = cross(e1, e2);
            if !is_finite3(&normal_world) {
                continue;
            }
            let normal_camera = mat3_mul_vec3(&r_wc, normal_world);
            if normal_camera[2] >= 0.0 {
                continue;
            }

            // Depth and frustum checks
            let center_camera = vec3_add(mat3_mul_vec3(&r_wc, *center_world), t_wc);
            if center_camera[2] <= 0.01 {
                continue;
            }
            let distance_3d = vec3_norm(center_camera);
            if self.max_detection_distance_m > 0.0
                && distance_3d > self.max_detection_distance_m
            {
                continue;
            }

            // Frustum cull using corners
            if !frustum_cull(&r_wc, t_wc, corners_world, width as i32, height as i32, fx, fy) {
                continue;
            }

            // Project corners with Brown–Conrady distortion
	            let mut img_pts: [[f32; 2]; 4] = [[0.0; 2]; 4];
	            let mut valid_count = 0usize;
	            let (k1, k2, p1, p2, k3) = extract_brown_conrady_coefficients(&self.distortion_coefficients);
            for c in 0..4 {
                let p = [
                    corners_world[c * 3 + 0],
                    corners_world[c * 3 + 1],
                    corners_world[c * 3 + 2],
                ];
                let pc = vec3_add(mat3_mul_vec3(&r_wc, p), t_wc);
                if !pc[2].is_finite() || pc[2] <= 0.0 {
                    continue;
                }
	                let xn = pc[0] / pc[2];
	                let yn = pc[1] / pc[2];
	                let (xd, yd) = distort_brown_conrady(xn, yn, k1, k2, p1, p2, k3);
	                let x = fx * xd + cx;
	                let y = fy * yd + cy;
	                img_pts[c] = [x, y];
	                valid_count += 1;
            }

            if valid_count < 4 || !img_pts.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
                continue;
            }

            // Preserve the projected orientation for perspective-aligned crops.
            if let Some((padded_quad, bbox)) = padded_quad_and_bbox(
                &img_pts,
                width as i32,
                height as i32,
                self.padding_factor,
                self.min_region_size_px,
            ) {
                region_distances.push((distance_3d, bbox, padded_quad));
            }
        }

        // Sort by distance (closest first) and limit to max_regions
        region_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let selected_regions: Vec<([i32; 4], [[f32; 2]; 4])> = region_distances
            .into_iter()
            .take(self.max_regions)
            .map(|(_, bbox, quad)| (bbox, quad))
            .collect();

        if selected_regions.is_empty() {
            let regions = vec![vec![0, 0, width as i32, height as i32]];
            return Ok((vec![], regions));
        }

        let crop_quads = selected_regions
            .iter()
            .map(|(_, quad)| quad.iter().flatten().copied().collect())
            .collect();
        let crop_regions = selected_regions
            .iter()
            .map(|(bbox, _)| bbox.to_vec())
            .collect();

        Ok((crop_quads, crop_regions))
    }

    fn update_config(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(val) = config.get_item("padding_factor")? {
            let extracted_f32 = val.extract::<f32>()?;
            self.padding_factor = extracted_f32.clamp(0.0, 5.0);
        }
        if let Some(val) = config.get_item("max_regions")? {
            let extracted_usize = val.extract::<usize>()?;
            self.max_regions = extracted_usize.max(1);
        }
        if let Some(val) = config.get_item("min_region_size_px")? {
            let extracted_i32 = val.extract::<i32>()?;
            self.min_region_size_px = extracted_i32.max(1);
        }
        if let Some(val) = config.get_item("max_detection_distance_m")? {
            let extracted_f32 = val.extract::<f32>()?;
            self.max_detection_distance_m = extracted_f32.max(0.0);
        }
        Ok(())
    }
}

fn invert_se3(t: &[f32; 16]) -> [f32; 16] {
    // R is top-left 3x3, t is top-right 3x1
    let r = [
        [t[0], t[1], t[2]],
        [t[4], t[5], t[6]],
        [t[8], t[9], t[10]],
    ];
    let r_t = transpose3(r);
    let trans = [t[3], t[7], t[11]];
    let t_inv = mat3_mul_vec3(&r_t, [-trans[0], -trans[1], -trans[2]]);
    let mut out = [0.0f32; 16];
    out[0] = r_t[0][0]; out[1] = r_t[0][1]; out[2] = r_t[0][2]; out[3] = t_inv[0];
    out[4] = r_t[1][0]; out[5] = r_t[1][1]; out[6] = r_t[1][2]; out[7] = t_inv[1];
    out[8] = r_t[2][0]; out[9] = r_t[2][1]; out[10] = r_t[2][2]; out[11] = t_inv[2];
    out[12] = 0.0; out[13] = 0.0; out[14] = 0.0; out[15] = 1.0;
    out
}

fn decompose_rt(t: &[f32; 16]) -> ([[f32; 3]; 3], [f32; 3]) {
    let r = [
        [t[0], t[1], t[2]],
        [t[4], t[5], t[6]],
        [t[8], t[9], t[10]],
    ];
    let trans = [t[3], t[7], t[11]];
    (r, trans)
}

fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }

fn vec3_norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn is_finite3(v: &[f32; 3]) -> bool { v[0].is_finite() && v[1].is_finite() && v[2].is_finite() }

fn frustum_cull(
    r_wc: &[[f32; 3]; 3],
    t_wc: [f32; 3],
    corners_world: &[f32; 12],
    width: i32,
    height: i32,
    fx: f32,
    fy: f32,
) -> bool {
    let mut corners_cam: [[f32; 3]; 4] = [[0.0; 3]; 4];
    for i in 0..4 {
        let p = [
            corners_world[i * 3 + 0],
            corners_world[i * 3 + 1],
            corners_world[i * 3 + 2],
        ];
        let pc = vec3_add(mat3_mul_vec3(r_wc, p), t_wc);
        corners_cam[i] = pc;
    }

    let min_depth = 0.01f32;
    if corners_cam.iter().all(|c| c[2] < min_depth) {
        return false;
    }

    let margin_factor = 0.5f32;
    let fov_x_half = ((width as f32 * 0.5) * (1.0 + margin_factor) / fx).atan();
    let fov_y_half = ((height as f32 * 0.5) * (1.0 + margin_factor) / fy).atan();

    let mut any_in = false;
    for c in corners_cam.iter().filter(|c| c[2] > min_depth) {
        let angle_x = (c[0].abs() / c[2]).atan();
        let angle_y = (c[1].abs() / c[2]).atan();
        if angle_x < fov_x_half && angle_y < fov_y_half {
            any_in = true;
            break;
        }
    }
    any_in
}

/// Expand a projected tag around its center and return its visible axis-aligned bounds.
fn padded_quad_and_bbox(
    points: &[[f32; 2]; 4],
    width: i32,
    height: i32,
    padding_factor: f32,
    min_region_size_px: i32,
) -> Option<([[f32; 2]; 4], [i32; 4])> {
    let center = points.iter().fold([0.0f32; 2], |sum, point| {
        [sum[0] + point[0] * 0.25, sum[1] + point[1] * 0.25]
    });
    let scale = 1.0 + padding_factor;
    let padded = points.map(|point| {
        [
            center[0] + (point[0] - center[0]) * scale,
            center[1] + (point[1] - center[1]) * scale,
        ]
    });

    let min_x = padded.iter().map(|point| point[0]).fold(f32::INFINITY, f32::min);
    let min_y = padded.iter().map(|point| point[1]).fold(f32::INFINITY, f32::min);
    let max_x = padded.iter().map(|point| point[0]).fold(f32::NEG_INFINITY, f32::max);
    let max_y = padded.iter().map(|point| point[1]).fold(f32::NEG_INFINITY, f32::max);
    let projected_span = (max_x - min_x).max(max_y - min_y);
    if projected_span < min_region_size_px as f32 {
        return None;
    }
    let bbox = [
        (min_x.floor() as i32).clamp(0, width),
        (min_y.floor() as i32).clamp(0, height),
        (max_x.ceil() as i32).clamp(0, width),
        (max_y.ceil() as i32).clamp(0, height),
    ];
    if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
        return None;
    }

    Some((padded, bbox))
}

fn extract_brown_conrady_coefficients(coefficients: &Vec<f32>) -> (f32, f32, f32, f32, f32) {
	let k1 = *coefficients.get(0).unwrap_or(&0.0);
	let k2 = *coefficients.get(1).unwrap_or(&0.0);
	let p1 = *coefficients.get(2).unwrap_or(&0.0);
	let p2 = *coefficients.get(3).unwrap_or(&0.0);
	let k3 = *coefficients.get(4).unwrap_or(&0.0);
	(k1, k2, p1, p2, k3)
}

fn distort_brown_conrady(x: f32, y: f32, k1: f32, k2: f32, p1: f32, p2: f32, k3: f32) -> (f32, f32) {
	let r2 = x * x + y * y;
	let r4 = r2 * r2;
	let r6 = r4 * r2;
	let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
	let x_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
	let y_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
	(x * radial + x_tangential, y * radial + y_tangential)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn distort_identity_when_zero_coefficients() {
		let (xd, yd) = distort_brown_conrady(0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0);
		assert!((xd - 0.1).abs() < 1e-6);
		assert!((yd + 0.2).abs() < 1e-6);
	}

	#[test]
	fn distort_radial_only_positive_k1() {
		let x = 0.2f32;
		let y = 0.1f32;
		let k1 = 0.5f32;
		let k2 = 0.0f32;
		let k3 = 0.0f32;
		let p1 = 0.0f32;
		let p2 = 0.0f32;
		let r2 = x * x + y * y;
		let radial = 1.0 + k1 * r2;
		let (xd, yd) = distort_brown_conrady(x, y, k1, k2, p1, p2, k3);
		assert!((xd - x * radial).abs() < 1e-6);
		assert!((yd - y * radial).abs() < 1e-6);
	}

    #[test]
    fn padded_quad_preserves_orientation_and_expands_bounds() {
        let points = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]];
        let (quad, bbox) = padded_quad_and_bbox(&points, 100, 100, 0.5, 16).unwrap();

        assert_eq!(quad, [[5.0, 5.0], [35.0, 5.0], [35.0, 35.0], [5.0, 35.0]]);
        assert_eq!(bbox, [5, 5, 35, 35]);
    }

    #[test]
    fn rotated_quad_keeps_original_minimum_region_semantics() {
        let diagonal = 5.0 * 2.0f32.sqrt();
        let points = [
            [20.0, 20.0 - diagonal],
            [20.0 + diagonal, 20.0],
            [20.0, 20.0 + diagonal],
            [20.0 - diagonal, 20.0],
        ];

        let (_, bbox) = padded_quad_and_bbox(&points, 100, 100, 0.35, 16).unwrap();

        assert_eq!(bbox, [10, 10, 30, 30]);
    }
}
