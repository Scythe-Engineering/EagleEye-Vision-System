//! Independent frozen Eagle Tags radius-two lookup; licensed data lives in family_data.

use crate::family_data::FAMILIES;
use std::collections::HashMap;

pub(crate) struct Family {
    pub(crate) name: &'static str,
    pub(crate) nbits: usize,
    #[expect(
        dead_code,
        reason = "Retain the licensed family descriptor; lookup uses a fixed radius two"
    )]
    pub(crate) h: usize,
    pub(crate) width: i32,
    pub(crate) total: i32,
    pub(crate) reversed: bool,
    pub(crate) codes: &'static [u64],
    pub(crate) x: &'static [i32],
    pub(crate) y: &'static [i32],
}

pub(crate) struct IndexedFamily {
    pub(crate) family: &'static Family,
    index: [HashMap<u32, Vec<usize>>; 3],
}

impl IndexedFamily {
    /// Index the three disjoint bit partitions of every codeword.
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
        Self { family, index }
    }

    /// Return a unique nearest code within radius two, or (-1, best distance).
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

/// Parse the native comma/ASCII whitespace family list without altering its order.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_families_radius_two() {
        for family in &FAMILIES {
            let indexed = IndexedFamily::new(family);
            for (id, &word) in family.codes.iter().enumerate() {
                assert_eq!(indexed.match_word(word), (id as i32, 0), "{}", family.name);
            }
            // Exhaust every one/two-bit corruption of real codes at both ends.
            for id in [0, family.codes.len() - 1] {
                let word = family.codes[id];
                for a in 0..family.nbits {
                    assert_eq!(indexed.match_word(word ^ (1 << a)), (id as i32, 1));
                    for b in a + 1..family.nbits {
                        assert_eq!(
                            indexed.match_word(word ^ (1 << a) ^ (1 << b)),
                            (id as i32, 2)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ties_and_repeated_partition_hits() {
        static FAMILY: Family = Family {
            name: "test",
            nbits: 6,
            h: 2,
            width: 4,
            total: 6,
            reversed: false,
            codes: &[0, 3],
            x: &[],
            y: &[],
        };
        let indexed = IndexedFamily::new(&FAMILY);
        assert_eq!(indexed.match_word(0), (0, 0));
        assert_eq!(indexed.match_word(1), (-1, 1));
        assert_eq!(indexed.match_word(0b111100), (-1, 3));
    }

    #[test]
    fn native_family_list_errors_and_separators() {
        let names = format!(" ,{}\t\r\n, {} ", FAMILIES[0].name, FAMILIES[1].name);
        let chosen = select_families(&names).unwrap();
        assert_eq!(chosen[0].family.name, FAMILIES[0].name);
        assert_eq!(chosen[1].family.name, FAMILIES[1].name);
        assert!(select_families(" ,\t\r\n").err().unwrap().contains("empty"));
        assert!(select_families("unknown")
            .err()
            .unwrap()
            .contains("unknown family: unknown"));
        assert!(select_families(&format!("{0},{0}", FAMILIES[0].name))
            .err()
            .unwrap()
            .contains("duplicate"));
        assert!(select_families(&" ".repeat(1025))
            .err()
            .unwrap()
            .contains("too long"));
        assert!(select_families(&format!("{}\u{000b}", FAMILIES[0].name)).is_err());
    }
}
