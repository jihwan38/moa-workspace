//! Future raw-byte gate only: source preparation does not authorize an ELF/run.
//! Quantization and GEMM are deliberately absent from this ABI/layout proof.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, P512},
    i4_p512_raw_adapter as raw, ops_int4_pack_p512 as k,
};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn signed(n: usize) -> i32 {
    let n = (n & 15) as i32;
    if n >= 8 { n - 16 } else { n }
}
fn code(case: usize, p: usize, c: usize, width: usize) -> i32 {
    match case {
        0 => {
            // Encode all nine row bits in each K960 block, so P/256 cluster
            // swaps and rows sixteen apart cannot evade the byte oracle.
            if c % 960 < 3 {
                return signed(p >> (4 * (c % 960)));
            }
            let mut z = (p as u32).wrapping_mul(0x9e3779b9)
                ^ (c as u32).wrapping_mul(0x85ebca6b)
                ^ 0xc2b2ae35;
            z ^= z >> 16;
            z = z.wrapping_mul(0x7feb352d);
            z ^= z >> 15;
            signed(z as usize)
        }
        1 => -8,
        2 => 7,
        3 => 0,
        4 => code(0, (73 * p + 17) % 512, (37 * c + 19) % width, width),
        _ => panic!("fixed fixture case"),
    }
}
fn canonical_bytes(values: &[i32]) -> Vec<u8> {
    values
        .chunks_exact(2)
        .map(|pair| {
            assert!(pair.iter().all(|v| (-8..=7).contains(v)));
            ((pair[0] as u8) & 15) | (((pair[1] as u8) & 15) << 4)
        })
        .collect()
}
fn blocked_pair_bytes(values: &[i32], width: usize) -> Vec<u8> {
    let mut bytes = vec![0; values.len() / 2];
    for p in 0..512 {
        for c in (0..width).step_by(2) {
            let position =
                (((p / 256 * (width / 960) + c / 960) * 256 + p % 256) * 960 + c % 960) / 2;
            let pair = &values[p * width + c..p * width + c + 2];
            bytes[position] = ((pair[0] as u8) & 15) | (((pair[1] as u8) & 15) << 4);
        }
    }
    bytes
}
fn independently_indexed_words(values: &[i32], width: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() / 2);
    for word_index in 0..values.len() / 8 {
        let block_row = word_index / 120;
        let row_in_half = block_row % 256;
        let group = (block_row / 256) % (width / 960);
        let half = block_row / (256 * (width / 960));
        let row = half * 256 + row_in_half;
        let col = group * 960 + (word_index % 120) * 8;
        let mut word = 0u32;
        for lane in 0..8 {
            word = word
                .checked_add(((values[row * width + col + lane] as u32) & 15) << (4 * lane))
                .unwrap();
        }
        bytes.extend(word.to_le_bytes());
    }
    bytes
}
async fn upload<D: MaterializableScalar, E: M>(
    d: &mut Device,
    bytes: Vec<u8>,
) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(bytes)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn read<D: MaterializableScalar, E: M>(d: &mut Device, t: &HbmTensor<D, Chip, E>) -> Vec<u8> {
    t.to_host::<E>(&mut d.pdma)
        .await
        .unwrap()
        .into_inner()
        .into_buf()
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut d = Device::new(k::pack_input.topology())?;
    let mut total_codes = 0usize;
    let mut total_packed_bytes = 0usize;
    macro_rules! check_shape {($axis:ty,$packed:ty,$i4:ty,$pack:expr,$reader:expr,$raw:expr)=>{{
        let width=<$axis>::SIZE;
        for case in 0..5 {
            let values:Vec<i32>=(0..512*width).map(|i|code(case,i/width,i%width,width)).collect();
            let canonical=canonical_bytes(&values);
            let blocked=blocked_pair_bytes(&values,width);
            assert_eq!(blocked,independently_indexed_words(&values,width),"independent word/nibble/layout oracle");
            // Check every physical signed nibble, including the legacy I4K960
            // UP contract and the same L960 DOWN ordering.
            for p in 0..512 { for c in 0..width {
                let nibble=((p/256*(width/960)+c/960)*256+p%256)*960+c%960;
                let actual=(blocked[nibble/2]>>(4*(nibble%2)))&15;
                assert_eq!(signed(actual as usize),values[p*width+c]);
            }}
            if case==0 {
                let mut seen=[false;16];
                for word in blocked.chunks_exact(4) { seen[(word[3]>>4) as usize]=true; }
                assert!(seen.iter().all(|x|*x),"all sixteen top nibbles");
                for p in 0..512 { for block in 0..width/960 {
                    let tag=(0..3).fold(0usize,|n,j|n|(((values[p*width+block*960+j] as usize)&15)<<(4*j)));
                    assert_eq!(tag,p,"all nine row bits survive in every K960 block");
                }}
            }
            if case==4 { for p in 0..512 { for c in 0..width {
                assert_eq!(values[p*width+c],code(0,(73*p+17)%512,(37*c+19)%width,width));
            }}}
            let input:HbmTensor<i32,Chip,m![P512,$axis]>=HostTensor::<i32,m![P512,$axis]>::from_vec(values.clone()).to_hbm(&mut d.pdma).await?;
            let shifts:HbmTensor<i32,Chip,m![$axis%8]>=HostTensor::<i32,m![$axis%8]>::from_vec((0..8).map(|i|4*i)).to_hbm(&mut d.pdma).await?;
            let mut packed:HbmTensor<i32,Chip,$packed>=HostTensor::<i32,$packed>::from_vec(std::iter::repeat_n(0x13579bdfi32,512*width/8)).to_hbm(&mut d.pdma).await?;
            let typed:HbmTensor<i4,Chip,$i4>=upload(&mut d,blocked.clone()).await;
            let mut candidate:HbmTensor<i4,Chip,m![P512,$axis]>=upload(&mut d,vec![0xad;512*width/2]).await;
            let mut control:HbmTensor<i4,Chip,m![P512,$axis]>=upload(&mut d,vec![0x52;512*width/2]).await;
            launch($pack,(&mut d,&input,&shifts,&mut packed)).await?;
            assert_eq!(read(&mut d,&packed).await,blocked,"whole blocked packed owner");
            // Candidate first, with independently initialized output. The later
            // normal typed launch embeds the identical reader image.
            $raw(&mut d,&packed,&mut candidate).await?;
            assert_eq!(read(&mut d,&candidate).await,canonical,"raw binding/canonical byte oracle");
            launch($reader,(&mut d,&typed,&mut control)).await?;
            assert_eq!(read(&mut d,&control).await,canonical,"typed blocked reader oracle");
            assert_eq!(read(&mut d,&packed).await,blocked,"packed owner immutable after await");
            assert_eq!(read(&mut d,&typed).await,blocked,"typed I4 source immutable");
            let input_bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
            assert_eq!(read(&mut d,&input).await,input_bytes,"ordinary I32 input immutable");
            let shift_bytes:Vec<u8>=(0..8i32).flat_map(|i|(4*i).to_le_bytes()).collect();
            assert_eq!(read(&mut d,&shifts).await,shift_bytes,"shift table immutable");
            println!("PACK_P512_CASE width={width} case={case} codes={} packed_bytes={} whole_owner_exact=true typed_raw_canonical_exact=true",values.len(),blocked.len());
            total_codes+=values.len();total_packed_bytes+=blocked.len();
        }
    }};}
    check_shape!(
        H,
        k::PackedInput,
        k::I4Input,
        k::pack_input,
        k::read_input,
        raw::read_input
    );
    check_shape!(
        L,
        k::PackedHidden,
        k::I4Hidden,
        k::pack_hidden,
        k::read_hidden,
        raw::read_hidden
    );
    println!(
        "PASS_PACK_P512 codes={total_codes} packed_bytes={total_packed_bytes} scope=integer_packing_and_raw_blocked_layout_only no_quantizer_gemm_model_or_speed_claim=true"
    );
    Ok(())
}
