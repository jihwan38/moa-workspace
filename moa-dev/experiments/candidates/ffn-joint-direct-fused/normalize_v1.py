#!/usr/bin/env python3
"""Normalize frozen v1 stages without touching the existing v2 reports."""
import json,sys
from results import outputs, ROOT, read, write
old = 'e131-joint-direct-fused'
new = 'e131-joint-two-row-v1'
for path, value in outputs('e131-joint-two-row-screen',109430).items():
    if path.parent != ROOT/'docs/experiments':
        continue
    path = path.with_name(path.name.replace(old,new))
    value['label'] = value['label'].replace('E131 ', 'E131-v1 ')
    value['variant'] = 'joint-two-row-v1'
    for field in ('reference','control_after_report'):
        if field in value:
            value[field] = value[field].replace(old,new)
    if '--check' in sys.argv:
        assert read(path) == value, path
    else:
        write(path,value)
print('Frozen two-row stages normalized; four-row reports unchanged')
