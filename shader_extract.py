#!/usr/bin/env python3
"""Shared shader-container extraction for the ReXGlue titles.

A guest shader is a self describing container: magic, header size (which is
also the offset to the microcode), microcode size, and an offset to the
bind-time patch table. Every title reads that header the same way; what differs
is where the containers live and what each title records in its manifest. This
module holds the parsing plus two ways of finding containers (walking a chunked
blob, scanning for the magic), so a title's own extractor is a thin driver that
says where to look.

Container layout, as read by the creators:
    +0   magic, 0x102A1101 for vertex, 0x102A1100 for pixel
    +4   header size, which is also the offset to the microcode
    +8   microcode size
    +20  offset to the patch table
"""
import os
import re
import struct

# The store lives next to this module; re-exported so drivers get it from here.
from xex_image import XexImage  # noqa: E402

CONTAINER_MAGIC = 0x102A1100
CONTAINER_MAGIC_MASK = ~1 & 0xFFFFFFFF
CONTAINER_OFFSET = 16

# Where each creator copies the container header inside the shader object. The
# patch table field sits 20 bytes into the header.
HEADER_OFFSET_IN_OBJECT = {1: 872, 0: 40}
PATCH_FIELD_IN_HEADER = 20

# The per kind number the extractor assigns starts at 1; entry 0 stays null.
FIRST_INDEX = 1
MAX_INDEX = 0xFF

SHADER_MODEL_RE = re.compile(rb"[vp]s_\d_\d")

# Chunk tags of a static chunked inventory, for the titles that have one. Its
# start address is the title's.
TAG_VERTEX = b"NVSH"
TAG_PIXEL = b"NPSH"

# Texture fetch constants live at device+1024, six dwords (24 bytes) per stage.
FETCH_CONSTANT_BASE = 1024
FETCH_CONSTANT_STRIDE = 24


def microcode_hash(data):
    """FNV-1a 64 over the microcode bytes as they sit in guest memory. The
    runtime computes the same over the shader object's copy."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def decode_patch_table(container, offset):
    """Decode the bind-time patch table, as walked by the D3D runtime's
    SetVertexShader and SetPixelShader.

    Header, relative to the table base (which is container relative):
        +0   value applied to the device dirty mask
        +4   clear mask, ANDed as ~mask into the neighbouring mask dword
        +8   not read by either bind path
        +12  flag; if non-zero the binder sets bit 0x01000000 in the device+32 mask
        +16  byte length of the record stream that follows
        +20  record stream

    The stream is three consecutive sections, each a list of records starting
    with u16 dst_offset, u16 count, and each terminated by a count of zero:

      1. skipped by *both* bind paths; fixed 8 byte records
      2. copy: `count` dwords written straight to device+1024+dst_offset
      3. merge: `count/2` (and_mask, or_value) pairs read-modify-written into
         device+1024+dst_offset

    Sections 2 and 3 are the texture fetch constants, so dst_offset/24 is the
    texture stage. This is the "baked state" half of a shader.
    """
    out = {
        "mask_value": struct.unpack_from(">I", container, offset)[0],
        "mask_clear": struct.unpack_from(">I", container, offset + 4)[0],
        "unread_8": struct.unpack_from(">I", container, offset + 8)[0],
        "flag_12": struct.unpack_from(">I", container, offset + 12)[0],
    }
    stream_size, = struct.unpack_from(">I", container, offset + 16)
    out["stream_size"] = stream_size

    pos = offset + 20
    end = pos + stream_size
    if end > len(container):
        raise ValueError("patch table stream runs past the container")

    def read_record():
        nonlocal pos
        dst, count = struct.unpack_from(">HH", container, pos)
        pos += 4
        return dst, count

    # Section 1: skipped, fixed 8 byte records.
    skipped = []
    while pos < end:
        dst, count = read_record()
        if not count:
            break
        skipped.append({
            "dst_offset": dst,
            "count": count,
            "value": struct.unpack_from(">I", container, pos)[0],
        })
        pos += 4
    out["section1_skipped"] = skipped

    # Section 2: raw dword copies into the fetch constant array.
    copies = []
    while pos < end:
        dst, count = read_record()
        if not count:
            break
        words = list(struct.unpack_from(f">{count}I", container, pos))
        pos += 4 * count
        copies.append({
            "dst_offset": dst,
            "stage": dst // FETCH_CONSTANT_STRIDE,
            "dwords": [f"0x{w:08X}" for w in words],
        })
    out["fetch_constant_copies"] = copies

    # Section 3: masked read-modify-write into the same array.
    merges = []
    while pos < end:
        dst, count = read_record()
        if not count:
            break
        pairs = []
        for _ in range(count // 2):
            and_mask, or_value = struct.unpack_from(">II", container, pos)
            pos += 8
            pairs.append({"and": f"0x{and_mask:08X}", "or": f"0x{or_value:08X}"})
        merges.append({
            "dst_offset": dst,
            "stage": dst // FETCH_CONSTANT_STRIDE,
            "pairs": pairs,
        })
    out["fetch_constant_merges"] = merges
    out["stream_consumed"] = pos - (offset + 20)
    return out


class Shader:
    """One container, parsed. `source` and `chunk_size` are the discovery
    breadcrumbs (which file/list it came from, and the chunk that held it);
    `patch_table` is decoded only when the caller asks."""

    def __init__(self, kind, index, addr, container, source=None, chunk_size=None,
                 with_patch=True):
        self.kind = kind
        self.index = index
        self.addr = addr
        self.source = source
        self.container = container
        self.chunk_size = chunk_size

        magic, header_size, microcode_size = struct.unpack_from(">III", container)
        self.magic = magic
        self.header_size = header_size
        self.microcode_size = microcode_size
        self.patch_table_offset, = struct.unpack_from(
            ">I", container, PATCH_FIELD_IN_HEADER)
        self.object_header_offset = HEADER_OFFSET_IN_OBJECT[magic & 1]
        self.header = container[:header_size]
        self.microcode = container[header_size:header_size + microcode_size]
        self.hash = microcode_hash(self.microcode)

        model = SHADER_MODEL_RE.search(self.header)
        self.model = model.group(0).decode() if model else None

        self.patch_table = None
        if with_patch and self.patch_table_offset:
            self.patch_table = decode_patch_table(container, self.patch_table_offset)

    @property
    def name(self):
        return f"{self.kind}_{self.index:03d}"

    @property
    def container_addr(self):
        return self.addr + CONTAINER_OFFSET


def parse_blob(img, start):
    """Walk a chunk list exactly as the guest's init loop does, stopping where it
    does, and return (shaders, end_address). For a title whose whole inventory
    sits in one static blob."""
    shaders = []
    counters = {"vs": FIRST_INDEX, "ps": FIRST_INDEX}
    addr = start
    while True:
        if not img.contains(addr, 8):
            break
        tag = img.read(addr, 4)
        if tag == TAG_VERTEX:
            kind = "vs"
        elif tag == TAG_PIXEL:
            kind = "ps"
        else:
            break  # the loop's own termination condition

        chunk_size = img.u32(addr + 4)
        if chunk_size <= CONTAINER_OFFSET or not img.contains(addr, chunk_size):
            raise ValueError(f"bad chunk size {chunk_size} at 0x{addr:08X}")

        container = img.read(addr + CONTAINER_OFFSET, chunk_size - CONTAINER_OFFSET)
        magic, = struct.unpack_from(">I", container)
        if magic & CONTAINER_MAGIC_MASK != CONTAINER_MAGIC:
            raise ValueError(
                f"bad container magic 0x{magic:08X} at 0x{addr + CONTAINER_OFFSET:08X}"
            )

        index = counters[kind]
        if index > MAX_INDEX:
            raise ValueError(f"more than {MAX_INDEX} {kind} shaders; table overflow")
        counters[kind] += 1

        shaders.append(Shader(kind, index, addr, container, chunk_size=chunk_size))
        addr += chunk_size
    return shaders, addr


def scan_containers(data, base, aligned):
    """Yield (addr, container) for every self-consistent container magic in
    `data`. For a title that ships its inventory as loose blobs."""
    i = 0
    while True:
        i = data.find(b"\x10\x2a\x11", i)
        if i < 0:
            return
        if not aligned or i % 4 == 0:
            magic, header_size, microcode_size = struct.unpack_from(">III", data, i)
            if (magic & CONTAINER_MAGIC_MASK == CONTAINER_MAGIC
                    and header_size >= 20
                    and microcode_size % 4 == 0
                    and i + header_size + microcode_size <= len(data)):
                yield base + i, data[i:i + header_size + microcode_size]
        i += 1


def collect_unique(sources):
    """Every unique shader across `sources`, a list of (name, iterable of
    (addr, container)) such as scan_containers yields, numbered per kind in
    order. Duplicates (same microcode bytes) are dropped."""
    shaders = []
    seen = set()
    counters = {"vs": FIRST_INDEX, "ps": FIRST_INDEX}
    for source, containers in sources:
        for addr, container in containers:
            magic, header_size = struct.unpack_from(">II", container)
            kind = "vs" if magic & 1 else "ps"
            key = (kind, container[header_size:])
            if key in seen:
                continue
            seen.add(key)

            index = counters[kind]
            if index > MAX_INDEX:
                raise ValueError(f"more than {MAX_INDEX} {kind} shaders; table overflow")
            counters[kind] += 1

            shaders.append(Shader(kind, index, addr, container, source=source,
                                  with_patch=False))
    return shaders


def write_shader_files(shaders, out_dir):
    """The three per-shader artifacts: the whole container, its microcode, and
    its header."""
    os.makedirs(out_dir, exist_ok=True)
    for s in shaders:
        base = os.path.join(out_dir, s.name)
        with open(base + ".bin", "wb") as f:
            f.write(s.container)
        with open(base + ".ucode", "wb") as f:
            f.write(s.microcode)
        with open(base + ".header", "wb") as f:
            f.write(s.header)
