// KryptonRecomp: guest shader pack reader shared by the ReXGlue titles.
//
// A "guest shader pack" is the container a title's build-time extractor emits:
// every shader the game can bind, already translated and compiled, linked into
// the executable's read-only data by a generated `.S` that `.incbin`s
// guest_shaders.bin (see shader_build.py's write_pack_asm). Each title that
// ships one wraps this reader in a thin, title-namespaced `guest_shaders.cpp`
// that only supplies its `PackTraits` and forwards the accessors.
//
// The pack format is identical across titles except for a handful of knobs
// (all of them in PackTraits): the magic, which flag bits mean what, the
// literal block size and whether the pool is right aligned inside it, how many
// renderer-owned entries follow the slot tables, and whether there is a
// per-entry hash section. The binary layout is otherwise shared:
//
//   header    magic, version, slot count, input count, key bytes,
//             literal block count, blob bytes
//   entries   2 * slot count records, vertex table first, then pixel, then
//             PackTraits::extra_entries renderer-owned shaders
//   hashes    one u64 per entry (only when PackTraits::has_hash_section)
//   inputs    (usage, usage index) pairs, referenced by offset from an entry
//   keys      interpolator semantic keys, referenced by byte offset
//   literals  fixed size blocks of float4s, referenced by block index.
//             Block 0 is a reserved all-zero sentinel meaning "no pool".
//   blob      the compiled shaders, referenced by offset and size
//
// No Plume types here: the pack is bytes, and the caller turns a blob into a
// RenderShader.

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace krypton {

// D3DDECLUSAGE plus its usage index, one per vertex shader input. This is the
// container's own vertex fetch patch table, which is what the guest matches
// against the bound vertex declaration at draw time; a host input layout is
// built from it instead, which is why vfetch never has to be translated.
struct GuestVertexInput {
  uint8_t usage;
  uint8_t usage_index;
};

// One shader. `dxil` and `spirv` point into the pack's blob section and are
// null when that format was not built for this platform.
struct GuestShader {
  const void* dxil = nullptr;
  uint32_t dxil_size = 0;
  const void* spirv = nullptr;
  uint32_t spirv_size = 0;

  // Pixel shaders: which texture/sampler register slots the shader declares.
  uint32_t texture_mask = 0;

  // Vertex shaders only.
  const GuestVertexInput* inputs = nullptr;
  uint32_t input_count = 0;

  // Interpolator semantic keys, sorted. Both shader types declare their
  // interpolators as TEXCOORD<key>, so the host linker performs the register
  // renumbering the guest does at draw time. D3D12 requires a pixel shader's
  // inputs to be a subset of the vertex shader's outputs, so a pipeline builder
  // should check one set against the other: the hardware synthesised a constant
  // for an unmatched input, and the host will simply refuse to link.
  const uint8_t* interpolator_keys = nullptr;
  uint32_t interpolator_key_count = 0;

  // The compiler's literal constant pool, in host order float4s, belonging in
  // the last registers of this shader's own bank. Never null; a shader without
  // a pool points at zeros.
  //
  // The guest writes these through neither constant setter, so they are not in
  // the register shadows and are overlaid on the bank a draw uploads. Pixel
  // shaders clamp against them (`min(x, c253.w)`, alpha against `c255.x`), so a
  // zero pool zeroes alpha and the alpha test discards the pixel. See
  // PREFIX_CANDIDATES in KryptonRecomp/xenos_ucode.py.
  const uint8_t* literals = nullptr;
  // How many bytes of the pool are live. Only set when
  // PackTraits::literals_tail_aligned is; zero otherwise.
  uint32_t literal_bytes = 0;
  bool has_literals = false;

  bool exports_point_size = false;
  bool has_cube_texture = false;

  // A pixel shader whose sampler 0 taps a resolve destination with screen-space
  // UVs under a non-native render scale: the texels it expects per screen pixel
  // are value * scale apart, and host screen pixels are scale * value apart, so
  // the footprint of the tap must widen by the resolve's scale. The scale is
  // per binding, not per shader; this only says which pixel shaders make the
  // tap at all.
  bool footprint_texture0 = false;

  // A pixel shader's SQ_PROGRAM_CNTL param_gen and SQ_CONTEXT_MISC
  // param_gen_pos. Only set when PackTraits::flag_param_gen is nonzero.
  bool param_gen = false;
  uint32_t param_gen_pos = 0;

  bool valid() const { return dxil_size != 0 || spirv_size != 0; }
};

// Everything one title's pack differs by, in one place, so a single reader can
// serve every title.
struct PackTraits {
  uint32_t magic;
  uint32_t version;
  uint8_t flag_point_size;
  uint8_t flag_has_cube;
  // 0 when the title's pack has no such flag.
  uint8_t flag_param_gen;
  uint8_t flag_footprint;
  // Size of a literal pool block, in bytes.
  uint32_t literal_block_bytes;
  // Set: the pool is right aligned within its block and the entry's tail byte
  // counts its live registers. Clear: the pool is left aligned at a fixed size
  // and the tail byte is param_gen_pos.
  bool literals_tail_aligned;
  // One microcode hash per entry, for titles that resolve a bound shader to its
  // slot by hashing it (see FindSlot).
  bool has_hash_section;
  // Renderer-owned shaders past the two slot tables, in the order the title's
  // generator writes them. Read back with Extra().
  uint32_t extra_entries;
};

// The shader debugger sidecar's identity, checked by
// native_renderer_shader_debug_impl.h before it reads anything.
struct DebugPackTraits {
  uint32_t magic;
  uint32_t version;
  // Entries past the two slot tables; the overlay skips them.
  uint32_t extra_entries;
};

// FNV-1a 64 over the microcode as it sits in guest memory, matching the hash
// the generator stores.
uint64_t HashGuestMicrocode(const uint8_t* data, size_t size);

// The reader for a pack linked into the executable. One instance per title.
class GuestShaderPack {
 public:
  // Parse a pack. Safe to call more than once; only the first call does work.
  // Returns false and logs if the pack is malformed, in which case every lookup
  // returns an invalid shader.
  bool Load(const uint8_t* data, size_t size, const PackTraits& traits);

  // Lookup by slot, 1..255. An out-of-range slot, or one the pack left empty or
  // could not translate, gives an entry whose `valid()` is false.
  const GuestShader& Vertex(uint32_t slot) const;
  const GuestShader& Pixel(uint32_t slot) const;

  // A renderer-owned shader with no guest slot, such as xenos_hlsl.py's
  // point_sprite_gs, by its index in the title's own order. Out of range gives
  // an invalid entry.
  const GuestShader& Extra(uint32_t index) const;

  // How many slots of each table the pack actually filled, for the startup log.
  uint32_t VertexCount() const;
  uint32_t PixelCount() const;

  // The slot whose microcode hashes to `microcode_hash`, or -1 if the pack does
  // not know it (an empty hash section always answers -1). A known slot may
  // still be invalid, when it did not translate.
  int FindSlot(bool pixel, uint64_t microcode_hash) const;

 private:
  std::vector<GuestShader> vertex_;
  std::vector<GuestShader> pixel_;
  std::vector<GuestShader> extra_;
  GuestShader absent_;
  std::unordered_map<uint64_t, uint32_t> vertex_by_hash_;
  std::unordered_map<uint64_t, uint32_t> pixel_by_hash_;
  bool attempted_ = false;
  bool loaded_ = false;
};

}  // namespace krypton
