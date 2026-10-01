#!/usr/bin/env python3
"""Finalize an already-built host after descriptor-review coordination timeout.

Preserve the exact original attempt; no compiler, source change or Arena call.
"""
import argparse
from pathlib import Path
from package import (ARTIFACTS, HERE, KERNELS, TARGET, checked_compile,
                     checked_gate, checked_package, clean_label, digest,
                     file_record, package_files, read, write)


def resume(label):
    clean_label(label)
    path = HERE / f'package-{label}.json'
    saved = HERE / f'package-{label}-wait-timeout.json'
    assert not saved.exists(), 'The original wait attempt was already preserved'
    original = path.read_bytes()
    value = read(path)
    assert value['status'] == 'failed' and value['host_build_exit_code'] == 0
    assert value['error'] == "TimeoutError('Independent graph gate not ready after host build')"
    assert value['builder'] == file_record(HERE / 'package.py')
    compile_path, built, control, control_dir, control_record = checked_compile(label)
    assert value['compile_record'] == file_record(compile_path)
    assert value['source_inputs'] == built['source_inputs']
    assert value['control'] == control_record
    artifact = ARTIFACTS / label
    folder = artifact / 'package'
    assert package_files(folder) == value['package_files']
    host = (folder / 'test_runtime').read_bytes()
    for kernel in KERNELS:
        image = artifact / 'standard-kernels' / f'furiosa_opt_gemma4::ops::{kernel}.bin'
        raw = image.read_bytes()
        assert file_record(image) == value['image_files'][kernel]
        expected = built['exact_image'] if kernel == TARGET else {
            k: control['images'][kernel][k] for k in ('bytes', 'sha256')}
        assert digest(raw) == expected and host.count(raw) == 1
        assert value['images'][kernel] == {**digest(raw), 'host_occurrences': 1}
    for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
        assert (folder / name).read_bytes() == (control_dir / name).read_bytes()
    value['graph_gate'] = checked_gate(label)
    for item in value['logs'].values():
        from package import ROOT
        assert file_record(ROOT / item['path']) == item
    path.rename(saved)
    assert saved.read_bytes() == original
    value.update(status='ready', original_wait_attempt=file_record(saved),
                 coordinator=file_record(Path(__file__).resolve()),
                 coordination_resume='Same exact host/images/source/fixtures; descriptor gate now ready. No rebuild or execution.')
    value.pop('error')
    write(path, value)
    checked_package(label)
    print('Package ready; original coordination timeout preserved, no rebuild')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    resume(parser.parse_args().label)
