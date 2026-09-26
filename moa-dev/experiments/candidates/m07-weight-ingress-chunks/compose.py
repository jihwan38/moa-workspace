#!/usr/bin/env python3
"""Replace only three full-weight HBM loads in the pinned M03 implementation."""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def compose(variant='dma-only-v1'):
    assert variant in {'dma-only-v1', 'compact-upgate-v2', 'stream-assembly-v3'}
    recipe = json.loads((HERE / variant / 'recipe.json').read_text())
    component = recipe['source_components'][0]
    original = subprocess.check_output(
        ['git', 'show', component['commit'] + ':' + component['path']], cwd=ROOT
    ).decode()
    result = original.replace(
        '// M03: C14-B ownership with one L stage, materialized native up/gate and F32 cluster reduction.',
        '// M07: M03 arithmetic and ownership; chunk only the three FP4 HBM ingress transfers.'
    )
    helper = '''// Five disjoint writes fill the original full packed tensor. Decode and dot
// still consume the full 30-row tensor once; no compact staging copy is added.
fn load_upgate_chunks(
    device: &mut Device,
    weight: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, FfnSlice, FfnRow, H]>,
) -> DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> {
    let mut packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> = DmTensor::new();
    static_passes!(i, [0, 1, 2, 3, 4], {
        weight.tile::<m![FfnRow], 6, m![Dummy2, FfnSlice, FfnRow = 6 # 30, H]>(6 * i)
            .to_dm_view(&mut device.tdma,
                packed.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 30, H]>(6 * i));
    });
    packed
}

'''
    marker = 'pub(crate) fn project_up_and_gate('
    assert result.count(marker) == 1
    result = result.replace(marker, helper + marker)
    for name in ['up', 'gate']:
        old = f'''    let {name}_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> =
        {name}_weight_packed.to_dm(&mut device.tdma);'''
        new = f'''    let {name}_weight_packed = load_upgate_chunks(device, {name}_weight_packed);'''
        assert result.count(old) == 1
        result = result.replace(old, new)
    old = '    let packed: DmTensor<f4e2m1, Chip, UpGateCluster, DownColumns, m![H % 60, TileL % 1920]> = weight.to_dm(&mut device.tdma);'
    new = '''    // Keep the original H60 x K1920 consumer layout. Only ingress is chunked.
    let mut packed: DmTensor<f4e2m1, Chip, UpGateCluster, DownColumns, m![H % 60, TileL % 1920]> = DmTensor::new();
    static_passes!(i, [0, 1, 2, 3, 4], {
        weight.tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, Dummy2, TileL]>(12 * i)
            .to_dm_view(&mut device.tdma,
                packed.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60, TileL % 1920]>(12 * i));
    });'''
    assert result.count(old) == 1
    result = result.replace(old, new)
    if variant in {'compact-upgate-v2', 'stream-assembly-v3'}:
        old = '''        weight.tile::<m![FfnRow], 6, m![Dummy2, FfnSlice, FfnRow = 6 # 30, H]>(6 * i)
            .to_dm_view(&mut device.tdma,
                packed.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 30, H]>(6 * i));'''
        new = '''        // A compact destination constrains the HBM descriptor to six rows.
        // Direct matching padded views in v1 emitted a full 30-row load per tile.
        let chunk: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 6, H]> = weight
            .tile::<m![FfnRow], 6, m![Dummy2, FfnSlice, FfnRow = 6 # 30, H]>(6 * i)
            .to_dm(&mut device.tdma);
        chunk.view().to_dm_view(&mut device.tdma,
            packed.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 30, H]>(6 * i));'''
        assert result.count(old) == 1
        result = result.replace(old, new)
        result = result.replace('// still consume the full 30-row tensor once; no compact staging copy is added.',
                                '// still consume the full 30-row tensor once. v2 adds one compact DMA staging copy per chunk.')
    if variant == 'stream-assembly-v3':
        old = '''        chunk.view().to_dm_view(&mut device.tdma,
            packed.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 30, H]>(6 * i));'''
        new = '''        // TDMA assembly widened the compact source beyond its allocation in v2.
        // Copy raw FP4 packets through Sub; 64 nibbles are exactly one 32-byte flit.
        device.sub.begin(chunk.view())
            .fetch::<m![FfnRow = 6, H / 64], m![H % 64]>()
            .collect::<m![FfnRow = 6, H / 64], m![H % 64]>()
            .commit_trim::<m![H % 64]>()
            .commit_view(packed.view_mut().tile::<m![FfnRow], 6, m![FfnRow = 6 #{!} 30, H]>(6 * i));'''
        assert result.count(old) == 1
        result = result.replace(old, new)
        result = result.replace('// still consume the full 30-row tensor once. v2 adds one compact DMA staging copy per chunk.',
                                '// still consume the full 30-row tensor once. v3 assembles compact chunks using raw Sub copies.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', nargs='?', default='dma-only-v1')
    args = parser.parse_args()
    path = HERE / args.variant / 'mlp.rs'
    with path.open('x') as out:
        out.write(compose(args.variant))
