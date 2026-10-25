#!/usr/bin/env python3
"""Share identical completed-run ELF storage, retaining every path and byte.

Only immutable built/submitted artifacts named by successful run records qualify.
Modes, owners, sizes and mtime must also agree. Mutable target files are excluded.
Default is inventory only; --apply writes an audit before changing any inode.
"""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def attributes(path):
    s = path.stat(follow_symlinks=False)
    return dict(size=s.st_size, mode=stat.S_IMODE(s.st_mode), uid=s.st_uid,
                gid=s.st_gid, mtime_ns=s.st_mtime_ns, device=s.st_dev,
                inode=s.st_ino, links=s.st_nlink)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    runs = root / 'runs'
    groups = defaultdict(dict)
    for record in runs.glob('*/run.json'):
        data = json.loads(record.read_text())
        if data.get('result') != 'command_succeeded':
            continue
        refs = list(data.get('built_artifacts', {}).values())
        if data.get('submitted_artifact'):
            refs.append(data['submitted_artifact'])
        for ref in refs:
            path = Path(ref.get('path', ''))
            if not path.is_absolute() or not path.is_relative_to(runs):
                continue
            if path.resolve() != path or not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(runs)
            if len(relative.parts) < 3 or relative.parts[1] not in ('built', 'submitted'):
                continue
            info = attributes(path)
            if info['size'] != ref.get('size') or info['links'] != 1:
                continue
            sha = ref.get('sha256', '')
            if len(sha) != 64:
                continue
            key = (sha, info['size'], info['mode'], info['uid'], info['gid'],
                   info['mtime_ns'], info['device'])
            groups[key][str(path)] = dict(path=str(path), original=info,
                                          run_record=str(record))
    selected = []
    for key, rows in groups.items():
        items = sorted(rows.values(), key=lambda x: x['path'])
        if len(items) > 1:
            selected.append(dict(sha256=key[0], bytes_each=key[1], files=items,
                                 potential_bytes=key[1] * (len(items) - 1)))
    selected.sort(key=lambda x: x['potential_bytes'], reverse=True)
    summary = dict(groups=len(selected), paths=sum(len(x['files']) for x in selected),
                   potential_bytes=sum(x['potential_bytes'] for x in selected))
    if not args.apply:
        print(json.dumps(summary, indent=2))
        return

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit = root / 'state' / f'frozen-artifact-dedupe-{stamp}.json'
    report = dict(status='planned', created_at_utc=stamp, inventory=summary,
                  scope='Completed immutable run artifacts only; every path retained',
                  restoration='With sufficient free space, copy each alias to a temporary file using copy2, verify SHA/mode/owner/mtime, then atomically replace that alias to restore a distinct inode.',
                  groups=selected)
    with audit.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')

    for group in selected:
        # Verify the complete group before any replacement in that group.
        for row in group['files']:
            path = Path(row['path'])
            assert attributes(path) == row['original'], path
            assert digest(path) == group['sha256'], path
            assert attributes(path) == row['original'], path
        canonical = Path(group['files'][0]['path'])
        for row in group['files'][1:]:
            path = Path(row['path'])
            assert attributes(path) == row['original'], path
            temporary = path.with_name(path.name + '.dedupe-' + uuid.uuid4().hex)
            os.link(canonical, temporary)
            os.replace(temporary, path)
        assert digest(canonical) == group['sha256'], canonical
        for row in group['files']:
            path = Path(row['path'])
            after = attributes(path)
            assert path.samefile(canonical), path
            for field in ('size', 'mode', 'uid', 'gid', 'mtime_ns', 'device'):
                assert after[field] == row['original'][field], (path, field)
            row['after'] = after
        group['status'] = 'verified_same_bytes_and_attributes_all_paths_retained'
        audit.write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'complete'
    report['completed_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    audit.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(summary, audit=str(audit), status='complete'), indent=2))


if __name__ == '__main__':
    main()
