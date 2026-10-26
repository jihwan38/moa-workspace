//! Offset-only extension of the unchanged independent rational/packing oracle.
#[path="int4_grouped_activation_fixture.rs"] mod base;
pub use base::{Expected, CASES, fixture, source};
pub use base::expected as original_expected;
pub use base::table as original_table;

pub fn table()->Vec<i32>{
    let original=base::table();
    (0..128usize).flat_map(|m|{
        let original=&original;
        (0..16usize).flat_map(move |r|(0..8usize).map(move |lane|{
            if lane==0 {original[m*8]} else {(0x5a3c0000u32|((m as u32)<<8)|((r as u32)<<4)|lane as u32)as i32}
        }))
    }).collect()
}
pub fn replica_bases()->Vec<i32>{
    (0..16usize).flat_map(|r|(0..8usize).map(move |lane|{
        if lane==0 {(32*r)as i32} else {(0x6b120000u32|((r as u32)<<8)|lane as u32)as i32}
    })).collect()
}
pub fn expected(bits:&[u16],width:usize)->Expected{
    let mut e=base::expected(bits,width);let groups=width/960;
    // Inverse physical slot traversal independently recovers destination p.
    for(slot,o)in e.offsets.iter_mut().enumerate(){
        let p=(slot/(groups*256))*256+slot%256;
        assert!(*o>=0&&*o<=4064&&*o%32==0);
        let m=*o/32;*o=512*m+32*(p%16)as i32;
        assert!(*o>=0&&*o<=65504&&*o%32==0);
    }e
}
pub fn validate_mappings(){
    let table=table();let original=original_table();let bases=replica_bases();
    assert_eq!(table.len()*4,65536);assert_eq!(bases.len()*4,512);
    for m in 0..128{for r in 0..16{for lane in 0..8{
        let at=(m*16+r)*8+lane;
        let want=if lane==0 {original[m*8]}else{(0x5a3c0000u32|((m as u32)<<8)|((r as u32)<<4)|lane as u32)as i32};
        assert_eq!(table[at],want);
    }}}
    for groups in [4usize,16]{
        let mut seen=vec![false;512*groups];let mut replicas=[0usize;16];
        for c in 0..2{for slice in 0..256{
            let g=slice/(256/groups);let r=slice%(256/groups);
            for j in 0..groups{
                let p=256*c+groups*r+j;
                let named_source=if groups==4 {4*(r%4)+j}else{j};
                assert_eq!(named_source,p%16);assert_eq!(bases[named_source*8],32*(p%16)as i32);
                let slot=(c*groups+g)*256+groups*r+j;
                assert!(!seen[slot]);seen[slot]=true;replicas[p%16]+=1;
                for m in 0..128usize{
                    let offset=512*m+bases[named_source*8]as usize;
                    assert_eq!(offset,((m*16+p%16)*8)*4);
                    assert!(offset%32==0&&offset+32<=table.len()*4);
                    assert_eq!(table[offset/4],original[m*8]);
                    assert_eq!((offset/32)/16,m);assert_eq!((offset/32)%16,p%16);
                }
            }
        }}
        assert!(seen.into_iter().all(|x|x));assert!(replicas.into_iter().all(|n|n==512*groups/16));
    }
}
pub fn verify_permutation(case:usize,width:usize,got:&Expected,original:&Expected){
    let groups=width/960;
    for p in 0..512 {for g in 0..groups {
        let(sp,sg,_)=source(case,p,g,0,groups);
        let dst=((p/256)*groups+g)*256+p%256;let src=((sp/256)*groups+sg)*256+sp%256;
        assert_eq!(got.scales[dst],original.scales[src]);
        // Mantissa follows the data permutation; replica follows DESTINATION row.
        assert_eq!(got.offsets[dst]/512,original.offsets[src]/512);
        assert_eq!(got.offsets[dst]%512,32*(p%16)as i32);
        assert_eq!(original.offsets[src]%512,32*(sp%16)as i32);
        for k in 0..960{let(_,_,sk)=source(case,p,g,k,groups);
            let a=(got.packed[dst*480+k/2]>>(4*(k%2)))&15;
            let b=(original.packed[src*480+sk/2]>>(4*(sk%2)))&15;assert_eq!(a,b);
        }
    }}
}
