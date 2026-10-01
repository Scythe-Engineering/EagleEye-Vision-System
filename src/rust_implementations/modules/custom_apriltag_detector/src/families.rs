use crate::family_data::FAMILIES;
use std::collections::HashMap;

pub(crate) struct Family {
    pub(crate) name: &'static str,
    pub(crate) nbits: usize,
    pub(crate) width: i32,
    pub(crate) total: i32,
    pub(crate) reversed: bool,
    pub(crate) codes: &'static [u64],
    pub(crate) bit_x: &'static [i32],
    pub(crate) bit_y: &'static [i32],
}

pub(crate) struct IndexedFamily {
    pub(crate) family: &'static Family,
    index: [HashMap<u32, Vec<usize>>; 3],
    payload_index: Vec<u8>,
}

impl IndexedFamily {
    pub(crate) fn new(family: &'static Family) -> Self {
        let mut index: [HashMap<u32, Vec<usize>>; 3] =
            std::array::from_fn(|_| HashMap::with_capacity(family.codes.len()));
        for (k, partition) in index.iter_mut().enumerate() {
            let lo = k * family.nbits / 3;
            let hi = (k + 1) * family.nbits / 3;
            let mask = (1u64 << (hi - lo)) - 1;
            for (id, &word) in family.codes.iter().enumerate() {
                partition
                    .entry(((word >> lo) & mask) as u32)
                    .or_default()
                    .push(id);
            }
        }
        let cells = (family.width * family.width).max(0) as usize;
        let mut payload_index = vec![u8::MAX; cells];
        for bit in 0..family.nbits {
            let x = family.bit_x[bit];
            let y = family.bit_y[bit];
            if x >= 0 && y >= 0 && x < family.width && y < family.width {
                payload_index[(y * family.width + x) as usize] = bit as u8;
            }
        }
        Self {
            family,
            index,
            payload_index,
        }
    }

    pub(crate) fn bit_at(&self, x: i32, y: i32) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.family.width || y >= self.family.width {
            return None;
        }
        let index = self.payload_index[(y * self.family.width + x) as usize];
        (index != u8::MAX).then_some(index as usize)
    }

    pub(crate) fn white_cell(&self, tag_id: i32, x: i32, y: i32) -> i32 {
        if x < 0 || y < 0 || x >= self.family.width || y >= self.family.width {
            return 1;
        }
        match self.bit_at(x, y) {
            None => 0,
            Some(bit) => {
                ((self.family.codes[tag_id as usize] >> (self.family.nbits - 1 - bit)) & 1) as i32
            }
        }
    }

    pub(crate) fn match_word(&self, word: u64) -> (i32, i32) {
        let mut best = 3;
        let mut id = -1;
        let mut tie = false;
        for k in 0..3 {
            let lo = k * self.family.nbits / 3;
            let hi = (k + 1) * self.family.nbits / 3;
            let key = ((word >> lo) & ((1u64 << (hi - lo)) - 1)) as u32;
            if let Some(candidates) = self.index[k].get(&key) {
                for &candidate in candidates {
                    let distance = (word ^ self.family.codes[candidate]).count_ones() as i32;
                    if distance < best {
                        best = distance;
                        id = candidate as i32;
                        tie = false;
                    } else if distance == best && id != candidate as i32 {
                        tie = true;
                    }
                }
            }
        }
        (if tie { -1 } else { id }, best)
    }
}

pub(crate) fn select_families(names: &str) -> Result<Vec<IndexedFamily>, String> {
    if names.len() > 1024 {
        return Err("family list too long".into());
    }
    let mut selected: Vec<IndexedFamily> = Vec::new();
    for name in names
        .split([',', ' ', '\t', '\r', '\n'])
        .filter(|name| !name.is_empty())
    {
        let family = FAMILIES
            .iter()
            .find(|family| family.name == name)
            .ok_or_else(|| format!("unknown family: {name}"))?;
        if selected.iter().any(|indexed| indexed.family.name == name) {
            return Err(format!("duplicate family: {name}"));
        }
        selected.push(IndexedFamily::new(family));
    }
    if selected.is_empty() {
        return Err("empty family list".into());
    }
    Ok(selected)
}
