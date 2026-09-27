#!/usr/bin/env python3
"""Revalidate unchanged already-built host after independent gate draft correction.

The original failed package record and builder remain immutable. No compilation.
"""
from package import *

def checked_package(label):
    compile_path, built, control, control_dir, control_record = checked_compile(label)
    path = HERE / f'package-v2-{label}.json'
    value = read(path)
    assert value['status'] == 'ready' and value['host_build_exit_code'] == 0
    assert value['compile_record'] == file_record(compile_path)
    assert value['graph_gate'] == checked_gate(label)
    assert value['control'] == control_record and value['source_inputs'] == built['source_inputs']
    folder = ARTIFACTS / label / 'package'
    assert package_files(folder) == value['package_files']
    host = (folder / 'test_runtime').read_bytes()
    for kernel in KERNELS:
        image_path = ARTIFACTS / label / 'standard-kernels' / f'furiosa_opt_gemma4::ops::{kernel}.bin'
        raw = image_path.read_bytes()
        assert file_record(image_path) == value['image_files'][kernel]
        expected = built['exact_image'] if kernel == TARGET else {k: control['images'][kernel][k] for k in ('bytes', 'sha256')}
        assert digest(raw) == expected
        assert value['images'][kernel] == {**digest(raw), 'host_occurrences': 1}
        assert host.count(raw) == 1
    for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
        assert (folder / name).read_bytes() == (control_dir / name).read_bytes()
    return path, value, folder, control_dir



def recover(label):
    compile_path, built, control, control_dir, control_record = checked_compile(label)
    gate = checked_gate(label)
    failed_path = HERE / f'package-{label}.json'
    old = read(failed_path)
    assert old['status'] == 'failed' and old['host_build_exit_code'] == 0
    assert old['error'] == 'AssertionError()'
    assert old['compile_record'] == file_record(compile_path)
    assert old['source_inputs'] == built['source_inputs'] and old['control'] == control_record
    assert old['builder'] == file_record(HERE / 'package.py')
    value = dict(old)
    value.update(status='ready', graph_gate=gate,
                 failed_attempt=file_record(failed_path),
                 recovery_verifier=file_record(Path(__file__).resolve()),
                 recovery_reason='Original host/all-three-images passed; packer correctly refused draft independent gate passed:false. Gate checker had included separate down480 copies; independent reviewer limited the up-copy test to H480. No source/image/host/input change or rebuild.')
    value.pop('error')
    write(HERE / f'package-v2-{label}.json', value)
    checked_package(label)
    print('Recovered same host/image package after independent gate PASS; no rebuild')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label');parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        path,value,_,_=checked_package(args.label)
        assert value['failed_attempt']==file_record(HERE/f'package-{args.label}.json')
        assert value['recovery_verifier']==file_record(Path(__file__).resolve())
        print('Recovered package/source/control/image gate PASS')
    else:recover(args.label)
