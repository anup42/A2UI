import hashlib
import json
import os
import pathlib
import sys
import tarfile

path = pathlib.Path('/group-volume/k.anup/code/A2UI/dataset/data/runs/dataset_muse_glimmer_100k_r8_2/genui.jsonl')
with path.open('rb') as source:
    stat = os.fstat(source.fileno())
    end = stat.st_size
    # Fix a prefix ending at a complete JSONL record before emitting any bytes.
    while end:
        start = max(0, end - 1024 * 1024)
        source.seek(start)
        tail = source.read(end - start)
        offset = tail.rfind(b'\n')
        if offset >= 0:
            end = start + offset + 1
            break
        end = start
    if not end:
        raise ValueError('No complete records')
    print(json.dumps({'snapshot_member':path.parent.name+'/'+path.name,'snapshot_prefix_bytes':end,'file_bytes_at_open':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}),file=sys.stderr,flush=True)
    source.seek(0)
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
        info = tarfile.TarInfo(path.parent.name+'/'+path.name)
        info.size = end
        info.mtime = int(stat.st_mtime)
        info.mode = 0o644
        archive.addfile(info,source)
    after = os.fstat(source.fileno())
    print(json.dumps({'file_bytes_at_end':after.st_size,'snapshot_complete':True,'snapshot_inode_unchanged':after.st_ino==stat.st_ino,'append_during_snapshot':after.st_size-stat.st_size}),file=sys.stderr,flush=True)
