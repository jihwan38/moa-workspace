//! One authoritative E4 cache per sliding layer. Offsets are explicitly E4 bytes.
use crate::{Chip, axes::*, ops_kv_e4_pool as kernels, ops_request_sliding_r64_fp8_scales::Scale8};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) struct Pool {
    pub k: HbmTensor<f8e4m3, Chip, kernels::DataPool>,
    pub v: HbmTensor<f8e4m3, Chip, kernels::DataPool>,
    pub ks: HbmTensor<f32, Chip, kernels::ScalePool>,
    pub vs: HbmTensor<f32, Chip, kernels::ScalePool>,
}
impl Pool {
    pub async fn new(device: &mut Device) -> Result<Self> {
        Ok(Self {
            k: launch(kernels::zero_data_pool, &mut *device).await?,
            v: launch(kernels::zero_data_pool, &mut *device).await?,
            ks: launch(kernels::zero_scale_pool, &mut *device).await?,
            vs: launch(kernels::zero_scale_pool, &mut *device).await?,
        })
    }
    pub async fn read_ring(
        &self,
        device: &mut Device,
        data_base: &HbmTensor<i32, Chip, m![1]>,
        scale_base: &HbmTensor<i32, Chip, m![1]>,
        ring: &mut Ring,
    ) -> std::result::Result<(),furiosa_opt_std::Error> {
        launch(
            kernels::dequant_ring,
            (
                device,
                &self.k,
                &self.ks,
                &self.v,
                &self.vs,
                data_base,
                scale_base,
                &mut ring.k,
                &mut ring.v,
            ),
        )
        .await?;
        Ok(())
    }
}
pub(crate) struct Ring {
    pub k: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    pub v: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
}
impl Ring {
    pub async fn new(device: &mut Device) -> Result<Self> {
        Ok(Self {
            k: HostTensor::<bf16,m![Ts,Ns,Ds]>::zero().to_hbm(&mut device.pdma).await?,
            v: HostTensor::<bf16,m![Ts,Ns,Ds]>::zero().to_hbm(&mut device.pdma).await?,
        })
    }
}
fn data_offset(slot: usize, pos: usize) -> i32 {
    ((slot * Ts::SIZE + pos % Ts::SIZE) * Ns::SIZE * Ds::SIZE) as i32
}
fn scale_offset(slot: usize, pos: usize, head: usize) -> i32 {
    (((slot * Ns::SIZE + head) * Ts::SIZE + pos % Ts::SIZE) * 8 * 4) as i32
}
pub(crate) struct Captured16 {
    pub k: HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    pub v: HbmTensor<f8e4m3, Chip, m![B, Ns, Ds]>,
    pub ks: HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
    pub vs: HbmTensor<f32, Chip, m![B, Ns, Scale8]>,
}
impl Captured16 {
    pub async fn new(device: &mut Device) -> Result<Self> {
        Ok(Self {
            k: HostTensor::<f8e4m3,m![B,Ns,Ds]>::zero().to_hbm(&mut device.pdma).await?,
            v: HostTensor::<f8e4m3,m![B,Ns,Ds]>::zero().to_hbm(&mut device.pdma).await?,
            ks: HostTensor::<f32,m![B,Ns,Scale8]>::zero().to_hbm(&mut device.pdma).await?,
            vs: HostTensor::<f32,m![B,Ns,Scale8]>::zero().to_hbm(&mut device.pdma).await?,
        })
    }
    pub async fn capture(
        &mut self,
        device: &mut Device,
        k: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        v: &mut HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    ) -> Result<()> {
        launch(
            kernels::capture_b16,
            (
                device,
                k,
                v,
                &mut self.k,
                &mut self.v,
                &mut self.ks,
                &mut self.vs,
            ),
        )
        .await?;
        Ok(())
    }
}
pub(crate) struct Captured128 {
    pub k: HbmTensor<f8e4m3, Chip, m![P128, Ns, Ds]>,
    pub v: HbmTensor<f8e4m3, Chip, m![P128, Ns, Ds]>,
    pub ks: HbmTensor<f32, Chip, m![P128, Ns, Scale8]>,
    pub vs: HbmTensor<f32, Chip, m![P128, Ns, Scale8]>,
}
impl Captured128 {
    pub async fn new(device: &mut Device) -> Result<Self> {
        Ok(Self {
            k: HostTensor::<f8e4m3,m![P128,Ns,Ds]>::zero().to_hbm(&mut device.pdma).await?,
            v: HostTensor::<f8e4m3,m![P128,Ns,Ds]>::zero().to_hbm(&mut device.pdma).await?,
            ks: HostTensor::<f32,m![P128,Ns,Scale8]>::zero().to_hbm(&mut device.pdma).await?,
            vs: HostTensor::<f32,m![P128,Ns,Scale8]>::zero().to_hbm(&mut device.pdma).await?,
        })
    }
    pub async fn capture(
        &mut self,
        device: &mut Device,
        k: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
        v: &mut HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    ) -> Result<()> {
        launch(
            kernels::capture_p128,
            (
                device,
                k,
                v,
                &mut self.k,
                &mut self.v,
                &mut self.ks,
                &mut self.vs,
            ),
        )
        .await?;
        Ok(())
    }
}
pub(crate) enum Commit16 {
    N1 {
        data: HbmTensor<i32, Chip, m![B = 1]>,
        scale: HbmTensor<i32, Chip, m![B = 1, Ns]>,
    },
    N2 {
        data: HbmTensor<i32, Chip, m![B = 2]>,
        scale: HbmTensor<i32, Chip, m![B = 2, Ns]>,
    },
    N3 {
        data: HbmTensor<i32, Chip, m![B = 3]>,
        scale: HbmTensor<i32, Chip, m![B = 3, Ns]>,
    },
    N4 {
        data: HbmTensor<i32, Chip, m![B = 4]>,
        scale: HbmTensor<i32, Chip, m![B = 4, Ns]>,
    },
    N5 {
        data: HbmTensor<i32, Chip, m![B = 5]>,
        scale: HbmTensor<i32, Chip, m![B = 5, Ns]>,
    },
    N6 {
        data: HbmTensor<i32, Chip, m![B = 6]>,
        scale: HbmTensor<i32, Chip, m![B = 6, Ns]>,
    },
    N7 {
        data: HbmTensor<i32, Chip, m![B = 7]>,
        scale: HbmTensor<i32, Chip, m![B = 7, Ns]>,
    },
    N8 {
        data: HbmTensor<i32, Chip, m![B = 8]>,
        scale: HbmTensor<i32, Chip, m![B = 8, Ns]>,
    },
    N9 {
        data: HbmTensor<i32, Chip, m![B = 9]>,
        scale: HbmTensor<i32, Chip, m![B = 9, Ns]>,
    },
    N10 {
        data: HbmTensor<i32, Chip, m![B = 10]>,
        scale: HbmTensor<i32, Chip, m![B = 10, Ns]>,
    },
    N11 {
        data: HbmTensor<i32, Chip, m![B = 11]>,
        scale: HbmTensor<i32, Chip, m![B = 11, Ns]>,
    },
    N12 {
        data: HbmTensor<i32, Chip, m![B = 12]>,
        scale: HbmTensor<i32, Chip, m![B = 12, Ns]>,
    },
    N13 {
        data: HbmTensor<i32, Chip, m![B = 13]>,
        scale: HbmTensor<i32, Chip, m![B = 13, Ns]>,
    },
    N14 {
        data: HbmTensor<i32, Chip, m![B = 14]>,
        scale: HbmTensor<i32, Chip, m![B = 14, Ns]>,
    },
    N15 {
        data: HbmTensor<i32, Chip, m![B = 15]>,
        scale: HbmTensor<i32, Chip, m![B = 15, Ns]>,
    },
    N16 {
        data: HbmTensor<i32, Chip, m![B = 16]>,
        scale: HbmTensor<i32, Chip, m![B = 16, Ns]>,
    },
}
impl Commit16 {
    pub async fn new(device: &mut Device, locations: &[(usize, usize)]) -> Result<Self> {
        if locations.is_empty()
            || locations.len() > 16
            || locations.iter().any(|&(slot, _)| slot >= 64)
        {
            return Err("invalid E4 live-prefix locations".into());
        }
        let data: Vec<_> = locations.iter().map(|&(s, p)| data_offset(s, p)).collect();
        let scale: Vec<_> = locations
            .iter()
            .flat_map(|&(s, p)| (0..Ns::SIZE).map(move |h| scale_offset(s, p, h)))
            .collect();
        Ok(match locations.len() {
            1 => Self::N1 {
                data: HostTensor::<i32,m![B=1]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=1,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            2 => Self::N2 {
                data: HostTensor::<i32,m![B=2]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=2,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            3 => Self::N3 {
                data: HostTensor::<i32,m![B=3]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=3,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            4 => Self::N4 {
                data: HostTensor::<i32,m![B=4]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=4,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            5 => Self::N5 {
                data: HostTensor::<i32,m![B=5]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=5,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            6 => Self::N6 {
                data: HostTensor::<i32,m![B=6]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=6,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            7 => Self::N7 {
                data: HostTensor::<i32,m![B=7]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=7,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            8 => Self::N8 {
                data: HostTensor::<i32,m![B=8]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=8,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            9 => Self::N9 {
                data: HostTensor::<i32,m![B=9]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=9,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            10 => Self::N10 {
                data: HostTensor::<i32,m![B=10]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=10,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            11 => Self::N11 {
                data: HostTensor::<i32,m![B=11]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=11,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            12 => Self::N12 {
                data: HostTensor::<i32,m![B=12]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=12,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            13 => Self::N13 {
                data: HostTensor::<i32,m![B=13]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=13,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            14 => Self::N14 {
                data: HostTensor::<i32,m![B=14]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=14,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            15 => Self::N15 {
                data: HostTensor::<i32,m![B=15]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=15,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            16 => Self::N16 {
                data: HostTensor::<i32,m![B=16]>::from_vec(data).to_hbm(&mut device.pdma).await?,
                scale: HostTensor::<i32,m![B=16,Ns]>::from_vec(scale).to_hbm(&mut device.pdma).await?,
            },
            _ => unreachable!(),
        })
    }
    pub async fn prompt(
        device: &mut Device,
        slot: usize,
        start: usize,
        count: usize,
    ) -> Result<Self> {
        Self::new(
            device,
            &(0..count).map(|i| (slot, start + i)).collect::<Vec<_>>(),
        )
        .await
    }
    pub async fn commit(&self, device: &mut Device, x: &Captured16, pool: &mut Pool) -> Result<()> {
        match self {
            Self::N1 { data, scale } => {
                launch(
                    kernels::commit_b16_1,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N2 { data, scale } => {
                launch(
                    kernels::commit_b16_2,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N3 { data, scale } => {
                launch(
                    kernels::commit_b16_3,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N4 { data, scale } => {
                launch(
                    kernels::commit_b16_4,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N5 { data, scale } => {
                launch(
                    kernels::commit_b16_5,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N6 { data, scale } => {
                launch(
                    kernels::commit_b16_6,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N7 { data, scale } => {
                launch(
                    kernels::commit_b16_7,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N8 { data, scale } => {
                launch(
                    kernels::commit_b16_8,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N9 { data, scale } => {
                launch(
                    kernels::commit_b16_9,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N10 { data, scale } => {
                launch(
                    kernels::commit_b16_10,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N11 { data, scale } => {
                launch(
                    kernels::commit_b16_11,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N12 { data, scale } => {
                launch(
                    kernels::commit_b16_12,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N13 { data, scale } => {
                launch(
                    kernels::commit_b16_13,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N14 { data, scale } => {
                launch(
                    kernels::commit_b16_14,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N15 { data, scale } => {
                launch(
                    kernels::commit_b16_15,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
            Self::N16 { data, scale } => {
                launch(
                    kernels::commit_b16_16,
                    (
                        device,
                        &x.k,
                        &x.v,
                        &x.ks,
                        &x.vs,
                        data,
                        scale,
                        &mut pool.k,
                        &mut pool.v,
                        &mut pool.ks,
                        &mut pool.vs,
                    ),
                )
                .await?
            }
        }
        Ok(())
    }
}
pub(crate) struct Commit128 {
    data: HbmTensor<i32, Chip, m![P128]>,
    scale: HbmTensor<i32, Chip, m![P128, Ns]>,
}
impl Commit128 {
    pub async fn new(device: &mut Device, slot: usize, start: usize) -> Result<Self> {
        if slot >= 64 {
            return Err("invalid E4 P128 ring".into());
        }
        Ok(Self {
            data: HostTensor::<i32,m![P128]>::from_vec((0..128).map(|i| data_offset(slot, start + i)))
                .to_hbm(&mut device.pdma)
                .await?,
            scale: HostTensor::<i32,m![P128,Ns]>::from_vec(
                (0..128).flat_map(|i| (0..8).map(move |h| scale_offset(slot, start + i, h))),
            )
            .to_hbm(&mut device.pdma)
            .await?,
        })
    }
    pub async fn commit(
        &self,
        device: &mut Device,
        x: &Captured128,
        pool: &mut Pool,
    ) -> Result<()> {
        launch(
            kernels::commit_p128,
            (
                device,
                &x.k,
                &x.v,
                &x.ks,
                &x.vs,
                &self.data,
                &self.scale,
                &mut pool.k,
                &mut pool.v,
                &mut pool.ks,
                &mut pool.vs,
            ),
        )
        .await?;
        Ok(())
    }
}
pub(crate) struct Step16 {
    pub ring_data: HbmTensor<i32, Chip, m![B]>,
    pub ring_scale: HbmTensor<i32, Chip, m![B]>,
    pub commits: Vec<Commit16>,
}
impl Step16 {
    pub async fn new(
        device: &mut Device,
        sequences: &[super::model::Sequence<'_>],
    ) -> Result<Self> {
        let n = sequences.len();
        if n == 0 || n > 16 {
            return Err("invalid E4 step width".into());
        }
        let locations: Vec<_> = sequences
            .iter()
            .map(|s| (s.blocks.sliding, s.start_position))
            .collect();
        let padded: Vec<_> = (0..16).map(|r| locations[r.min(n - 1)]).collect();
        let mut commits = Vec::new();
        for chunk in locations.chunks(16) {
            commits.push(Commit16::new(device, chunk).await?);
        }
        Ok(Self {
            ring_data: HostTensor::<i32,m![B]>::from_vec(padded.iter().map(|&(s, _)| data_offset(s, 0)))
                .to_hbm(&mut device.pdma)
                .await?,
            ring_scale: HostTensor::<i32,m![B]>::from_vec(padded.iter().map(|&(s, _)| scale_offset(s, 0, 0)))
                .to_hbm(&mut device.pdma)
                .await?,
            commits,
        })
    }
}
pub(crate) struct Step64 {
    pub ring_data: HbmTensor<i32, Chip, m![R64]>,
    pub ring_scale: HbmTensor<i32, Chip, m![R64]>,
    pub commits: Vec<Commit16>,
    pub data: HbmTensor<i32, Chip, m![R64]>,
    pub scale: HbmTensor<i32, Chip, m![R64, Ns]>,
}
impl Step64 {
    pub async fn new(
        device: &mut Device,
        sequences: &[super::model::Sequence<'_>],
    ) -> Result<Self> {
        let n = sequences.len();
        if n == 0 || n > 64 {
            return Err("invalid E4 step width".into());
        }
        let locations: Vec<_> = sequences
            .iter()
            .map(|s| (s.blocks.sliding, s.start_position))
            .collect();
        let padded: Vec<_> = (0..64).map(|r| locations[r.min(n - 1)]).collect();
        let mut commits = Vec::new();
        if n < 64 {
            for chunk in locations.chunks(16) {
                commits.push(Commit16::new(device, chunk).await?);
            }
        }
        Ok(Self {
            ring_data: HostTensor::<i32,m![R64]>::from_vec(padded.iter().map(|&(s, _)| data_offset(s, 0)))
                .to_hbm(&mut device.pdma)
                .await?,
            ring_scale: HostTensor::<i32,m![R64]>::from_vec(padded.iter().map(|&(s, _)| scale_offset(s, 0, 0)))
                .to_hbm(&mut device.pdma)
                .await?,
            commits,
            data: HostTensor::<i32,m![R64]>::from_vec(padded.iter().map(|&(s, p)| data_offset(s, p)))
                .to_hbm(&mut device.pdma)
                .await?,
            scale: HostTensor::<i32,m![R64,Ns]>::from_vec(
                padded
                    .iter()
                    .flat_map(|&(s, p)| (0..8).map(move |h| scale_offset(s, p, h))),
            )
            .to_hbm(&mut device.pdma)
            .await?,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offsets_separate_data_and_head_major_scales() {
        assert_eq!(data_offset(63, 1023), 134215680);
        assert_eq!(data_offset(63, 1024), 132120576);
        assert_eq!(scale_offset(63, 1023, 7), 16777184);
        assert_eq!(scale_offset(1, 1024, 3), 360448);
        assert_eq!(data_offset(0, 0), 0);
    }
}
