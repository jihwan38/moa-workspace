"""Read ABI1 fixed-int image metadata; label spans without changing task code.

Layout verified against installed furiosa-opt-abi 0.8.1/src/image.rs.
This is diagnostic metadata editing only, never an adopted/submitted kernel.
"""
import hashlib


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def inspect(data, rename=False):
    assert data[:4] == b"FOPT"
    pos = 4
    result = bytearray(data)

    def uint(n):
        nonlocal pos
        assert pos + n <= len(data)
        value = int.from_bytes(data[pos:pos+n], "little")
        pos += n
        return value

    def vec(read):
        count = uint(8)
        assert count < 100000
        return [read() for _ in range(count)]

    def blob():
        nonlocal pos
        count = uint(8)
        start = pos
        pos += count
        assert pos <= len(data)
        return start, data[start:pos]

    version, chips, pes = uint(4), uint(1), uint(1)
    assert version == 1
    sizes = vec(lambda: vec(lambda: uint(4)))
    weight = None
    if uint(1):
        kind = uint(4)
        _, raw = blob()
        weight = {"kind": kind, **digest(raw)}
    stacks = vec(lambda: (uint(4), uint(8)))
    slots = vec(lambda: (uint(4), uint(8)))
    inputs, outputs = vec(lambda: uint(4)), vec(lambda: uint(4))
    depth = uint(4)

    def span():
        start, raw = blob()
        begin, end = uint(2), uint(2)
        label = f"{begin:x}".rjust(len(raw), "0").encode()
        assert len(label) == len(raw)
        if rename:
            result[start:start+len(raw)] = label
        return {"name": raw.decode(), "label": label.decode(), "begin_marker": begin,
                "end_marker": end, "name_offset": start, "name_bytes": len(raw)}

    spans = vec(lambda: vec(span))
    assert len(spans) == len(sizes)
    header_size = pos
    tasks = []
    for chunk in sizes:
        row = []
        for size in chunk:
            pos = (pos+255)//256*256
            assert pos+size <= len(data)
            assert data[pos:pos+size] == result[pos:pos+size]
            row.append({"offset": pos, **digest(data[pos:pos+size])})
            pos += size
        tasks.append(row)
    assert (pos+255)//256*256 == len(data)
    return bytes(result), {"version": version, "chips": chips, "pes": pes, "header_bytes": header_size,
        "weight": weight, "stacks": stacks, "slots": slots, "inputs": inputs, "outputs": outputs,
        "profile_depth": depth, "spans": spans, "tasks": tasks,
        "original_image": digest(data), "labelled_image": digest(result)}
