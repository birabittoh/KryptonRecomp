// Shared guest shader pack reader. See guest_shader_pack.h for the format and
// for what each title overrides via PackTraits.
//
// The bytes come from kGuestShaderPackData/-End, defined by a generated `.S`
// that `.incbin`s the title's guest_shaders.bin into the executable's own
// read-only data, so the pointers handed out below point straight into the
// running process's image rather than a heap buffer.
//
// Every offset is validated against the section it indexes before anything is
// handed out. A pack built by a different revision of the generator should fail
// to load rather than produce plausible garbage that turns into a GPU fault
// later.

#include "guest_shader_pack.h"

#include <cstring>

#include <rex/logging.h>

namespace krypton {
namespace {

#pragma pack(push, 1)
struct PackHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t slots;
  uint32_t input_count;
  uint32_t key_bytes;
  uint32_t literal_blocks;
  uint32_t blob_bytes;
};

struct PackEntry {
  uint32_t dxil_offset;
  uint32_t dxil_size;
  uint32_t spirv_offset;
  uint32_t spirv_size;
  uint32_t texture_mask;
  uint16_t input_offset;  // in pairs, into the input section
  uint8_t input_count;
  uint8_t flags;
  uint16_t key_offset;  // in bytes, into the key section
  uint8_t key_count;
  // param_gen_pos, or the literal pool's live register count when
  // PackTraits::literals_tail_aligned is set.
  uint8_t tail;
  uint16_t literal_block;  // index into the literal section, 0 means none
};
#pragma pack(pop)

static_assert(sizeof(PackHeader) == 28, "pack header layout");
static_assert(sizeof(PackEntry) == 30, "pack entry layout");

// Decode one record. Returns false on any offset that does not fit its section,
// which is the whole validation: a truncated or mismatched pack is rejected here
// rather than at the point a blob is handed to the driver.
bool Decode(const PackEntry& entry, const PackHeader& header, const PackTraits& traits,
            const uint8_t* blob, const GuestVertexInput* inputs, const uint8_t* keys,
            const uint8_t* literals, GuestShader& out) {
  auto in_blob = [&](uint32_t offset, uint32_t size) {
    return size == 0 || (offset <= header.blob_bytes && size <= header.blob_bytes - offset);
  };
  if (!in_blob(entry.dxil_offset, entry.dxil_size) ||
      !in_blob(entry.spirv_offset, entry.spirv_size))
    return false;
  if (entry.input_offset + entry.input_count > header.input_count)
    return false;
  if (entry.key_offset + entry.key_count > header.key_bytes)
    return false;
  if (entry.literal_block >= header.literal_blocks)
    return false;
  if (traits.literals_tail_aligned && entry.tail > 8)
    return false;

  if (entry.dxil_size) {
    out.dxil = blob + entry.dxil_offset;
    out.dxil_size = entry.dxil_size;
  }
  if (entry.spirv_size) {
    out.spirv = blob + entry.spirv_offset;
    out.spirv_size = entry.spirv_size;
  }
  out.texture_mask = entry.texture_mask;
  out.inputs = entry.input_count ? inputs + entry.input_offset : nullptr;
  out.input_count = entry.input_count;
  out.interpolator_keys = entry.key_count ? keys + entry.key_offset : nullptr;
  out.interpolator_key_count = entry.key_count;

  // Always pointed somewhere: block 0 is the all-zero sentinel, so a shader
  // with no pool reads the zeros it would have read on the hardware anyway.
  const size_t block_base = size_t(entry.literal_block) * traits.literal_block_bytes;
  if (traits.literals_tail_aligned) {
    const size_t pool_bytes = size_t(entry.tail) * 16;
    out.literals = literals + block_base + (traits.literal_block_bytes - pool_bytes);
    out.literal_bytes = uint32_t(pool_bytes);
  } else {
    out.literals = literals + block_base;
    out.param_gen_pos = entry.tail;
  }
  out.has_literals = entry.literal_block != 0;

  out.exports_point_size = (entry.flags & traits.flag_point_size) != 0;
  out.has_cube_texture = (entry.flags & traits.flag_has_cube) != 0;
  out.footprint_texture0 = (entry.flags & traits.flag_footprint) != 0;
  if (traits.flag_param_gen != 0)
    out.param_gen = (entry.flags & traits.flag_param_gen) != 0;
  return true;
}

}  // namespace

uint64_t HashGuestMicrocode(const uint8_t* data, size_t size) {
  uint64_t hash = 0xCBF29CE484222325ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 0x100000001B3ull;
  }
  return hash;
}

bool GuestShaderPack::Load(const uint8_t* pack, size_t pack_size, const PackTraits& traits) {
  if (attempted_)
    return loaded_;
  attempted_ = true;

  if (pack_size < sizeof(PackHeader)) {
    REXLOG_ERROR("guest_shaders: embedded pack is too small to be a pack");
    return false;
  }
  PackHeader header;
  std::memcpy(&header, pack, sizeof(header));
  if (header.magic != traits.magic || header.version != traits.version) {
    REXLOG_ERROR("guest_shaders: embedded pack is not a v{} pack (magic {:08X} version {})",
                 traits.version, header.magic, header.version);
    return false;
  }

  // Entries past the two tables hold the renderer's own shaders, which have no
  // guest slot.
  const size_t table_entries = size_t(header.slots) * 2;
  const size_t entry_count = table_entries + traits.extra_entries;
  const size_t entries_bytes = entry_count * sizeof(PackEntry);
  const size_t hashes_bytes = traits.has_hash_section ? entry_count * sizeof(uint64_t) : 0;
  const size_t expected = sizeof(PackHeader) + entries_bytes + hashes_bytes +
                          size_t(header.input_count) * sizeof(GuestVertexInput) +
                          size_t(header.key_bytes) +
                          size_t(header.literal_blocks) * traits.literal_block_bytes +
                          size_t(header.blob_bytes);
  if (pack_size != expected) {
    REXLOG_ERROR("guest_shaders: embedded pack is {} bytes, expected {}", pack_size, expected);
    return false;
  }

  const uint8_t* cursor = pack + sizeof(PackHeader);
  const PackEntry* records = reinterpret_cast<const PackEntry*>(cursor);
  cursor += entries_bytes;
  const uint8_t* hashes = cursor;
  cursor += hashes_bytes;
  const GuestVertexInput* inputs = reinterpret_cast<const GuestVertexInput*>(cursor);
  cursor += size_t(header.input_count) * sizeof(GuestVertexInput);
  const uint8_t* keys = cursor;
  cursor += header.key_bytes;
  const uint8_t* literals = cursor;
  cursor += size_t(header.literal_blocks) * traits.literal_block_bytes;
  const uint8_t* blob = cursor;

  vertex_.assign(header.slots, GuestShader{});
  pixel_.assign(header.slots, GuestShader{});
  extra_.assign(traits.extra_entries, GuestShader{});
  for (size_t i = 0; i < entry_count; ++i) {
    GuestShader& out = i >= table_entries ? extra_[i - table_entries]
                       : i < header.slots ? vertex_[i]
                                          : pixel_[i - header.slots];
    if (!Decode(records[i], header, traits, blob, inputs, keys, literals, out)) {
      REXLOG_ERROR("guest_shaders: entry {} in embedded pack is out of range", i);
      vertex_.clear();
      pixel_.clear();
      extra_.clear();
      return false;
    }
  }

  // Slot 0 is never filled; a zero hash is an empty slot.
  if (traits.has_hash_section) {
    vertex_by_hash_.clear();
    pixel_by_hash_.clear();
    for (size_t i = 1; i < table_entries; ++i) {
      uint64_t hash;
      std::memcpy(&hash, hashes + i * sizeof(uint64_t), sizeof(hash));
      if (hash == 0 || i == header.slots)
        continue;
      if (i < header.slots)
        vertex_by_hash_.emplace(hash, uint32_t(i));
      else
        pixel_by_hash_.emplace(hash, uint32_t(i - header.slots));
    }
  }

  loaded_ = true;
  REXLOG_DEBUG("guest_shaders: loaded {} vertex, {} pixel from embedded pack ({} KiB)",
               VertexCount(), PixelCount(), pack_size / 1024);
  return true;
}

const GuestShader& GuestShaderPack::Vertex(uint32_t slot) const {
  return slot < vertex_.size() ? vertex_[slot] : absent_;
}

const GuestShader& GuestShaderPack::Pixel(uint32_t slot) const {
  return slot < pixel_.size() ? pixel_[slot] : absent_;
}

const GuestShader& GuestShaderPack::Extra(uint32_t index) const {
  return index < extra_.size() ? extra_[index] : absent_;
}

uint32_t GuestShaderPack::VertexCount() const {
  uint32_t count = 0;
  for (const GuestShader& shader : vertex_)
    count += shader.valid() ? 1 : 0;
  return count;
}

uint32_t GuestShaderPack::PixelCount() const {
  uint32_t count = 0;
  for (const GuestShader& shader : pixel_)
    count += shader.valid() ? 1 : 0;
  return count;
}

int GuestShaderPack::FindSlot(bool pixel, uint64_t microcode_hash) const {
  const auto& map = pixel ? pixel_by_hash_ : vertex_by_hash_;
  const auto it = map.find(microcode_hash);
  return it != map.end() ? int(it->second) : -1;
}

}  // namespace krypton
