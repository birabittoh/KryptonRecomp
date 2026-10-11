"""Title-agnostic build plumbing for the guest shader generator.

The DXC invocation, the `.S` emitter that `.incbin`s a pack into the executable,
and the write-if-changed helper. These are the same for every title; the pack
and debug-sidecar formats stay with each title's gen-guest-shaders.py, because
their entry layouts have diverged.
"""

import concurrent.futures
import os
import subprocess

# Shader model 6.0 is the floor Plume's D3D12 and Vulkan backends both accept,
# and nothing emitted here needs anything newer.
PROFILE = {"vs": "vs_6_0", "ps": "ps_6_0", "gs": "gs_6_0"}

# Matches Plume's own helper (examples/cmake/modules/PlumeDXC.cmake) so the
# guest shaders and the overlay shaders are compiled the same way. -fvk-invert-y
# on vertex shaders lets the HLSL be written once for the D3D clip space
# convention, with the Vulkan flip handled at compile time.
DXIL_OPTS = ["-Wno-ignored-attributes"]
SPIRV_OPTS = ["-spirv", "-fspv-target-env=vulkan1.0", "-fvk-use-dx-layout"]

# The reflection and debug sections are dead weight here: the pack carries the
# signature tables the host needs, decoded from the guest container rather than
# recovered from the compiled blob.
STRIP_OPTS = ["-Qstrip_reflect", "-Qstrip_debug"]


def write_if_different(path, data):
    """Avoid touching an unchanged file, so DXC can be skipped on rebuild."""
    if os.path.exists(path):
        with open(path, "rb") as handle:
            if handle.read() == data:
                return False
    with open(path, "wb") as handle:
        handle.write(data)
    return True


def write_pack_asm(asm_path, pack_path, symbol):
    """Emit a `.S` that `.incbin`s a pack, so it links straight into the exe.

    Used for both guest_shaders.bin and guest_shaders_debug.bin.

    Symbol names need a leading underscore on Mach-O only; ELF and Windows COFF
    (built here with clang, not cl, so no /Zl-style decoration) use the name as
    written. `.incbin`'s path is read at assemble time, so it has to survive
    unescaped through GAS's string literal: forward slashes side-step the
    backslash-escaping problem entirely and every assembler here accepts them.
    """
    incbin_path = os.path.abspath(pack_path).replace("\\", "/")
    content = f"""\
#if defined(__APPLE__)
#define SYM(name) _##name
.section __TEXT,__const
#elif defined(_WIN32)
#define SYM(name) name
.section .rdata,"dr"
#else
#define SYM(name) name
.section .rodata
#endif

.p2align 4
.global SYM({symbol})
SYM({symbol}):
.incbin "{incbin_path}"
.global SYM({symbol}End)
SYM({symbol}End):
"""
    write_if_different(asm_path, content.encode("utf-8"))
    return True


def compile_one(dxc_command, entry, fmt, out_dir):
    source = entry.source
    output = os.path.join(out_dir, "%s.%s" % (entry.name, fmt))
    # The HLSL source is only rewritten when it changes, so an output that is
    # newer than its source is still valid and DXC can be skipped entirely.
    if os.path.exists(output) and \
            os.path.getmtime(output) >= os.path.getmtime(source):
        with open(output, "rb") as handle:
            return handle.read()

    options = list(DXIL_OPTS if fmt == "dxil" else SPIRV_OPTS)
    if fmt == "dxil":
        options += STRIP_OPTS
    if fmt == "spirv" and entry.kind == "vs":
        options.append("-fvk-invert-y")

    command = list(dxc_command) + options + [
        "-E", "main", "-T", PROFILE[entry.kind], "-Fo", output, source]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output):
        raise SystemExit("dxc failed on %s (%s):\n%s%s"
                         % (entry.name, fmt, result.stdout, result.stderr))
    with open(output, "rb") as handle:
        return handle.read()


def compile_all(dxc_command, entries, formats, out_root, jobs):
    work = []
    for fmt in formats:
        out_dir = os.path.join(out_root, fmt)
        os.makedirs(out_dir, exist_ok=True)
        for entry in entries:
            # An entry a title's generator flagged as untranslated is skipped.
            if getattr(entry, "skipped", None) is None:
                work.append((entry, fmt, out_dir))

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(compile_one, dxc_command, entry, fmt, out_dir):
                (entry, fmt)
            for entry, fmt, out_dir in work
        }
        for future in concurrent.futures.as_completed(futures):
            entry, fmt = futures[future]
            entry.blobs[fmt] = future.result()
