#pragma once

// Shared implementation of the native renderer's shader debugger overlay.
// See native_renderer_shader_debug.h.
//
// This is included *inside* the title's namespace, from a thin
// native_renderer_shader_debug.cpp that provides the standard includes, the
// extern "C" debug pack symbols and the sidecar's traits:
//
//   namespace mytitle {
//   constexpr krypton::DebugPackTraits kGuestShaderDebugPackTraits{magic, version, extras};
//   #include "native_renderer_shader_debug_impl.h"
//   }
//
// so GuestVertexShader/GuestPixelShader resolve to the title's own and the
// definitions land in that namespace. The title's native_renderer_shader_debug.h
// declares what is defined here. Do not add includes here; the including
// translation unit owns them.
//
// A title may also define KRYPTON_GUEST_SHADER_DEBUG_ON_PUBLISH as a function
// taking (bool pixel, uint32_t slot, uint64_t id), called for each shader handed
// to the dialog.
//
// The overlay is written against the emulated-Xenos command processor, which
// tracks shaders by microcode hash because it discovers them at runtime. Here
// the inventory is the closed set of containers in guest_shaders.bin, addressed
// by guest table slot. ReXApp exposes a ShaderDebuggerOverride hook for this
// case: slots are presented as if they were hashes, and the four questions the
// dialog asks are answered: what exists, what is running, what is slow, and what
// should stop drawing.
//
// The identifier handed to the dialog is the guest table slot, with 0x100 set
// for a pixel shader. The type bit stays below bit 32 because the dialog derives
// an ImGui id with `PushID(static_cast<int>(hash))`, which truncates a higher
// bit and would give vs_007 and ps_007 the same id. The identifier is stable
// across runs, so the names and disable flags the dialog persists in
// shaders.toml stay meaningful: a name written against vs_017 still means
// vs_017. The slot is what every other part of this renderer speaks, so a number
// read out of the overlay can be pasted into a log filter.
//
// Disabling is enforced in the draw path: a draw whose pipeline was built from a
// disabled vertex or pixel shader is dropped and counted like any other drop.
// The emulator substitutes a no-op translation; dropping the draw is what
// answers "which shader draws this?".

namespace {

// Both slot tables are 256 entries, the same bound the pack reader uses.
constexpr uint32_t kSlots = 256;

// One shader's live state. Every field is written on the guest thread and read
// on the UI thread, so all of it is atomic: this is a debug view, and a torn
// counter in it would be a worse bug than the one it is there to find.
struct ShaderState {
  std::atomic<bool> disabled{false};
  std::atomic<uint32_t> active_frame{0};
  std::atomic<uint64_t> draw_count{0};
  std::atomic<uint64_t> profile_ns{0};
  std::atomic<uint64_t> profile_draws{0};
};

ShaderState g_vertex[kSlots];
ShaderState g_pixel[kSlots];

std::atomic<bool> g_profiling{false};
// Bumped once per guest swap. A shader is "active" when it drew in this frame
// or the one before it: the overlay draws at its own rate, so testing only the
// current frame makes every row flicker.
std::atomic<uint32_t> g_frame{1};

// The pixel bit inside the identifier. It has to live below bit 32: the dialog
// pushes an ImGui id with `PushID(static_cast<int>(hash))`, so anything above
// that is truncated away, and a vertex and a pixel shader in the same table
// slot would come out as two visible items with the same id.
constexpr uint64_t kPixelBit = 0x100;

ShaderState* Lookup(bool pixel, uint32_t slot) {
  if (slot >= kSlots)
    return nullptr;
  return pixel ? &g_pixel[slot] : &g_vertex[slot];
}

const GuestShader& PackEntry(bool pixel, uint32_t slot) {
  return pixel ? GuestPixelShader(slot) : GuestVertexShader(slot);
}

// --- The sidecar text pack -------------------------------------------------
//
// guest_shaders_debug.bin carries, per shader, the name the extractor gave it,
// the microcode disassembly and the HLSL the emitter produced. It is a separate
// pack from guest_shaders.bin because it is roughly twice the size and none of
// it is touched unless the debugger is opened, but it is linked into the exe
// the same way; see shader_build.py's write_pack_asm. Its magic, version and
// extra entry count are the title's, from kGuestShaderDebugPackTraits.

#pragma pack(push, 1)
struct DebugHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t slots;
  uint32_t text_bytes;
};

struct DebugEntry {
  uint32_t name_offset, name_size;
  uint32_t ucode_offset, ucode_size;
  uint32_t hlsl_offset, hlsl_size;
};
#pragma pack(pop)

static_assert(sizeof(DebugHeader) == 16, "debug pack header layout");
static_assert(sizeof(DebugEntry) == 24, "debug pack entry layout");

// UI thread only, so no locking: the details provider is the only caller.
bool g_text_attempted = false;
bool g_text_loaded = false;
const DebugEntry* g_text_entries = nullptr;
const uint8_t* g_text_data = nullptr;  // start of the text section
uint32_t g_text_bytes = 0;

bool EnsureTextPack() {
  if (g_text_attempted)
    return g_text_loaded;
  g_text_attempted = true;

  const uint8_t* const pack = kGuestShaderDebugPackData;
  const size_t pack_size = size_t(kGuestShaderDebugPackDataEnd - kGuestShaderDebugPackData);

  if (pack_size < sizeof(DebugHeader)) {
    REXLOG_DEBUG(
        "native_renderer: embedded debug pack is empty, so the shader debugger lists shaders but "
        "shows no source");
    return false;
  }

  DebugHeader header;
  std::memcpy(&header, pack, sizeof(header));
  const krypton::DebugPackTraits& traits = kGuestShaderDebugPackTraits;
  if (header.magic != traits.magic || header.version != traits.version ||
      header.slots != kSlots) {
    REXLOG_WARN("native_renderer: embedded debug pack is not a v{} pack", traits.version);
    return false;
  }

  // The renderer-owned entries past the two tables are not listed by the
  // overlay, but have to be skipped to find the text.
  const size_t entry_count = size_t(header.slots) * 2 + traits.extra_entries;
  const size_t entries_bytes = entry_count * sizeof(DebugEntry);
  const size_t expected = sizeof(DebugHeader) + entries_bytes + size_t(header.text_bytes);
  if (pack_size != expected) {
    REXLOG_WARN("native_renderer: embedded debug pack is {} bytes, expected {}", pack_size,
               expected);
    return false;
  }

  g_text_entries = reinterpret_cast<const DebugEntry*>(pack + sizeof(DebugHeader));
  g_text_data = pack + sizeof(DebugHeader) + entries_bytes;
  g_text_bytes = header.text_bytes;
  g_text_loaded = true;
  return true;
}

std::string ReadText(uint32_t offset, uint32_t size) {
  if (size == 0 || offset > g_text_bytes || size > g_text_bytes - offset)
    return {};
  return std::string(reinterpret_cast<const char*>(g_text_data) + offset, size);
}

// Name, microcode disassembly and HLSL for one shader, or empties when the
// sidecar is missing.
struct ShaderText {
  std::string name;
  std::string ucode;
  std::string hlsl;
};

ShaderText LoadText(bool pixel, uint32_t slot) {
  ShaderText out;
  if (!EnsureTextPack() || slot >= kSlots)
    return out;
  const DebugEntry& entry = g_text_entries[(pixel ? kSlots : 0) + slot];
  out.name = ReadText(entry.name_offset, entry.name_size);
  out.ucode = ReadText(entry.ucode_offset, entry.ucode_size);
  out.hlsl = ReadText(entry.hlsl_offset, entry.hlsl_size);
  return out;
}

}  // namespace

uint64_t GuestShaderDebugId(bool pixel, uint32_t slot) {
  return (pixel ? kPixelBit : 0) | slot;
}

bool GuestShaderDrawDisabled(int vertex_slot, int pixel_slot) {
  if (vertex_slot >= 0 && uint32_t(vertex_slot) < kSlots &&
      g_vertex[vertex_slot].disabled.load(std::memory_order_relaxed))
    return true;
  if (pixel_slot >= 0 && uint32_t(pixel_slot) < kSlots &&
      g_pixel[pixel_slot].disabled.load(std::memory_order_relaxed))
    return true;
  return false;
}

void NoteGuestShaderDraw(int vertex_slot, int pixel_slot, uint64_t elapsed_ns) {
  const uint32_t frame = g_frame.load(std::memory_order_relaxed);
  const bool profiling = elapsed_ns != 0;
  for (int pass = 0; pass < 2; ++pass) {
    const int slot = pass == 0 ? vertex_slot : pixel_slot;
    ShaderState* state = slot >= 0 ? Lookup(pass != 0, uint32_t(slot)) : nullptr;
    if (state == nullptr)
      continue;
    state->active_frame.store(frame, std::memory_order_relaxed);
    state->draw_count.fetch_add(1, std::memory_order_relaxed);
    if (profiling) {
      // The same elapsed time is charged to both stages. A host draw does not
      // separate them, and pretending otherwise would invent a split that the
      // measurement does not contain.
      state->profile_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
      state->profile_draws.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

bool GuestShaderProfilingEnabled() { return g_profiling.load(std::memory_order_relaxed); }

void GuestShaderDebugEndFrame() { g_frame.fetch_add(1, std::memory_order_relaxed); }

std::vector<rex::ui::ShaderDebuggerEntry> GuestShaderSnapshot() {
  std::vector<rex::ui::ShaderDebuggerEntry> out;
  const uint32_t frame = g_frame.load(std::memory_order_relaxed);

  for (uint32_t pass = 0; pass < 2; ++pass) {
    const bool pixel = pass != 0;
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
      const GuestShader& shader = PackEntry(pixel, slot);
      if (!shader.valid())
        continue;  // The game never had a shader in this table slot.
      const ShaderState& state = pixel ? g_pixel[slot] : g_vertex[slot];

      rex::ui::ShaderDebuggerEntry entry;
      entry.ucode_hash = GuestShaderDebugId(pixel, slot);
      entry.type = pixel ? 1u : 0u;
      // The dialog labels this as the shader's size. Nothing at runtime holds
      // the microcode, so the compiled blob's dword count is what is reported;
      // it is the size of the thing that actually runs.
      entry.dword_count = (shader.dxil_size ? shader.dxil_size : shader.spirv_size) / 4;
      entry.disabled = state.disabled.load(std::memory_order_relaxed);
      const uint32_t active = state.active_frame.load(std::memory_order_relaxed);
      entry.active = active != 0 && frame - active <= 1;
      entry.profile_total_ns = state.profile_ns.load(std::memory_order_relaxed);
      entry.profile_draw_count = state.profile_draws.load(std::memory_order_relaxed);
#if defined(KRYPTON_GUEST_SHADER_DEBUG_ON_PUBLISH)
      KRYPTON_GUEST_SHADER_DEBUG_ON_PUBLISH(pixel, slot, entry.ucode_hash);
#endif
      out.push_back(entry);
    }
  }
  return out;
}

rex::ui::ShaderDebuggerDetails GuestShaderDetails(uint64_t id) {
  rex::ui::ShaderDebuggerDetails out;
  const bool pixel = (id & kPixelBit) != 0;
  const uint32_t slot = uint32_t(id & (kPixelBit - 1));
  const GuestShader& shader = PackEntry(pixel, slot);
  // An identifier this renderer never issued decodes to a slot like any other,
  // so the range check has to come before the decode is trusted. See
  // SetGuestShaderDisabled.
  if (id >= kPixelBit * 2 || !shader.valid())
    return out;

  out.found = true;
  const ShaderState& state = pixel ? g_pixel[slot] : g_vertex[slot];
  out.info.ucode_hash = id;
  out.info.type = pixel ? 1u : 0u;
  out.info.dword_count = (shader.dxil_size ? shader.dxil_size : shader.spirv_size) / 4;
  out.info.disabled = state.disabled.load(std::memory_order_relaxed);
  const uint32_t frame = g_frame.load(std::memory_order_relaxed);
  const uint32_t active = state.active_frame.load(std::memory_order_relaxed);
  out.info.active = active != 0 && frame - active <= 1;
  out.info.profile_total_ns = state.profile_ns.load(std::memory_order_relaxed);
  out.info.profile_draw_count = state.profile_draws.load(std::memory_order_relaxed);

  const ShaderText text = LoadText(pixel, slot);

  // A header the dialog shows above the disassembly. Everything in it is
  // something a wrong-looking draw makes you want to check: which slot this is,
  // what the pipeline builder will match its inputs against, which texture
  // slots the texture mirror will visit for it, and whether it carries a
  // literal constant pool.
  std::string preamble;
  preamble += (pixel ? "; pixel shader, guest table slot " : "; vertex shader, guest table slot ") +
              std::to_string(slot);
  if (!text.name.empty())
    preamble += " (" + text.name + ")";
  preamble += "\n; draws so far: " + std::to_string(state.draw_count.load(std::memory_order_relaxed));
  preamble += "\n; literal constant pool: ";
  preamble += shader.has_literals ? "present" : "none";
  if (pixel) {
    preamble += "\n; texture slots declared:";
    if (shader.texture_mask == 0) {
      preamble += " none";
    } else {
      for (uint32_t i = 0; i < 32; ++i) {
        if (shader.texture_mask & (1u << i))
          preamble += " t" + std::to_string(i);
      }
    }
    if (shader.has_cube_texture)
      preamble += "\n; samples a cube map";
  } else {
    preamble += "\n; vertex inputs (D3DDECLUSAGE, index):";
    for (uint32_t i = 0; i < shader.input_count; ++i) {
      preamble += " (" + std::to_string(shader.inputs[i].usage) + "," +
                  std::to_string(shader.inputs[i].usage_index) + ")";
    }
    if (shader.exports_point_size)
      preamble += "\n; exports point size";
  }
  preamble += "\n; interpolator keys:";
  for (uint32_t i = 0; i < shader.interpolator_key_count; ++i)
    preamble += " " + std::to_string(shader.interpolator_keys[i]);
  preamble += "\n\n";

  out.ucode_disassembly =
      preamble + (text.ucode.empty() ? "; no microcode disassembly in the debug pack" : text.ucode);

  // One translation, not one per format: the HLSL is the same source both
  // backends were compiled from, and the blob offered for saving is whichever
  // one this build produced.
  rex::ui::ShaderDebuggerTranslation translation;
  translation.modification = 0;
  translation.is_translated = true;
  translation.is_valid = true;
  translation.host_disassembly =
      text.hlsl.empty() ? "// no HLSL in the debug pack" : text.hlsl;
  const void* blob = shader.dxil ? shader.dxil : shader.spirv;
  const uint32_t blob_size = shader.dxil ? shader.dxil_size : shader.spirv_size;
  if (blob != nullptr) {
    translation.translated_binary.assign(static_cast<const uint8_t*>(blob),
                                         static_cast<const uint8_t*>(blob) + blob_size);
  }
  out.translations.push_back(std::move(translation));
  return out;
}

bool SetGuestShaderDisabled(uint64_t id, bool disabled) {
  // Logged unconditionally: this is a human clicking a checkbox, so it is rare,
  // and a silent rejection here looks exactly like the game ignoring the toggle.
  REXLOG_WARN(
      "native_renderer: shader debugger asked to set disabled={} for id 0x{:016X} (decodes as {} "
      "slot {}, in range {})",
      disabled, id, (id & kPixelBit) != 0 ? "pixel" : "vertex",
      uint32_t(id & (kPixelBit - 1)), id < (kPixelBit * 2));
  // An identifier this renderer never issued is not an error: shaders.toml is
  // shared with the emulated-Xenos backend, whose identifiers are real microcode
  // hashes and mean nothing here.
  ShaderState* state = id < (kPixelBit * 2)
                           ? Lookup((id & kPixelBit) != 0, uint32_t(id & (kPixelBit - 1)))
                           : nullptr;
  if (state == nullptr)
    return false;
  state->disabled.store(disabled, std::memory_order_relaxed);
  return true;
}

void SetGuestShaderProfiling(bool enabled) {
  g_profiling.store(enabled, std::memory_order_relaxed);
}

void ResetGuestShaderProfiling() {
  for (uint32_t slot = 0; slot < kSlots; ++slot) {
    for (ShaderState* state : {&g_vertex[slot], &g_pixel[slot]}) {
      state->profile_ns.store(0, std::memory_order_relaxed);
      state->profile_draws.store(0, std::memory_order_relaxed);
    }
  }
}

void SetGuestShaderBlacklist(const std::vector<uint64_t>& ids) {
  size_t applied = 0;
  for (uint64_t id : ids)
    applied += SetGuestShaderDisabled(id, true) ? 1 : 0;
  if (applied != 0) {
    REXLOG_DEBUG("native_renderer: {} of {} shader(s) in shaders.toml disabled at startup", applied,
                ids.size());
  }
}
