# KryptonRecomp

Offline Xbox 360 Xenos shader translator, shared by the ReXGlue titles.

Each title's executable carries its whole shader inventory compiled in, so
translating microcode to HLSL is a finite offline problem. This repository is
the title-agnostic part of that pipeline; anything that names a game's
addresses, shader slots or pack magic lives in the game's own repository.

## Build time (Python)

* `xex_image.py` decrypts and decompresses a `default.xex` into an image
  addressable by guest address.
* `shader_extract.py` parses shader containers and their patch tables, and
  finds them either by walking a chunked blob (`parse_blob`) or by scanning for
  the container magic (`scan_containers`, `collect_unique`).
* `xenos_ucode.py` decodes Xenos microcode into control flow and ALU/fetch
  instructions.
* `xenos_hlsl.py` emits one HLSL source per shader, plus the point sprite
  geometry shader. `FOOTPRINT_SHADERS` and `REFRACTION_SLOT` are per title
  knobs a driver sets before translating.
* `shader_build.py` runs DXC and writes the `.S` that `.incbin`s a pack into the
  executable.

## Run time (C++)

* `guest_shader_pack.h/.cpp` read the pack linked into the executable.
  Everything that differs between titles' packs is in `PackTraits`.
* `native_renderer_shader_debug_impl.h` implements the F2 shader debugger
  overlay. It is included inside the title's namespace; see its header comment.

## What a title supplies

* `scripts/extract_shaders.py`: where its containers live, and its manifest.
* `scripts/gen-guest-shaders.py`: translation knobs, its renderer-owned extra
  shaders, and the pack and debug sidecar layouts.
* `src/gpu/guest_shaders.cpp`: its `PackTraits` and title-namespaced accessors
  over `krypton::GuestShaderPack`.
* `src/gpu/native_renderer_shader_debug.cpp`: its `DebugPackTraits`, the pack
  symbols, and the include of the overlay implementation.

The build adds this directory to the include path and compiles
`guest_shader_pack.cpp`; the Python drivers put it on `sys.path`.
