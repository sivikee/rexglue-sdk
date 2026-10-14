#include "spirv_to_glsl.h"

#include <cstring>
#include <exception>
#include <memory>

#include <spirv_glsl.hpp>

#include <rex/logging.h>

namespace rex::graphics::gl {

namespace {
void DumpBlocks(spirv_cross::CompilerGLSL& comp) {
  auto res = comp.get_shader_resources();
  auto dump = [&](const char* kind, spirv_cross::SmallVector<spirv_cross::Resource>& v) {
    for (auto& r : v) {
      const auto& t = comp.get_type(r.base_type_id);
      REXGPU_ERROR("  {} '{}' members={} declared_size={}", kind, r.name, t.member_types.size(),
                   comp.get_declared_struct_size(t));
      for (uint32_t i = 0; i < t.member_types.size(); ++i) {
        const auto& mt = comp.get_type(t.member_types[i]);
        REXGPU_ERROR("    [{}] {} off={} basetype={} vec={} cols={} arr={} stride={}", i,
                     comp.get_member_name(r.base_type_id, i), comp.type_struct_member_offset(t, i),
                     uint32_t(mt.basetype), mt.vecsize, mt.columns, mt.array.size(),
                     mt.array.empty() ? 0u : comp.type_struct_member_array_stride(t, i));
      }
    }
  };
  dump("UBO", res.uniform_buffers);
  dump("SSBO", res.storage_buffers);
}

// Eden (and some drivers) cannot run vertex shaders that load from SSBOs; shared memory is therefore read
// through R32UI texture buffers: xe_shared_memory[k].shared_memory[expr] -> texelFetch(xe_sm_tbo_k, int(expr)).x
void RewriteSharedMemoryAsTbo(std::string& src) {
  const std::string decl_marker = "xe_shared_memory[4];";
  size_t dm = src.find(decl_marker);
  if (dm == std::string::npos) {
    return;
  }
  size_t ds = src.rfind("layout(", dm);
  if (ds == std::string::npos) {
    return;
  }
  std::string decls;
  for (uint32_t i = 0; i < 4; ++i) {
    decls += "layout(binding = " + std::to_string(kSharedMemoryTboUnitBase + i) +
             ") uniform usamplerBuffer xe_sm_tbo_" + std::to_string(i) + ";\n";
  }
  src.replace(ds, dm + decl_marker.size() - ds, decls);
  const std::string use = "xe_shared_memory[";
  size_t pos = 0;
  while ((pos = src.find(use, pos)) != std::string::npos) {
    size_t k_begin = pos + use.size();
    size_t k_end = src.find(']', k_begin);
    const std::string member = "].shared_memory[";
    if (k_end == std::string::npos || src.compare(k_end, member.size(), member) != 0) {
      pos = k_begin;
      continue;
    }
    std::string k = src.substr(k_begin, k_end - k_begin);
    size_t e_begin = k_end + member.size();
    size_t depth = 1, e = e_begin;
    while (e < src.size() && depth) {
      if (src[e] == '[') ++depth;
      else if (src[e] == ']') --depth;
      ++e;
    }
    std::string expr = src.substr(e_begin, e - 1 - e_begin);
    std::string rep = "texelFetch(xe_sm_tbo_" + k + ", int(" + expr + ")).x";
    src.replace(pos, e - pos, rep);
    pos += rep.size();
  }
}

}  // namespace

bool ConvertSpirvToGlsl(const std::vector<uint8_t>& spirv, uint32_t texture_unit_base,
                        uint32_t texture_count, GlslShader& out, std::string& error) {
  std::unique_ptr<spirv_cross::CompilerGLSL> comp_ptr;
  try {
    std::vector<uint32_t> words(spirv.size() / 4);
    std::memcpy(words.data(), spirv.data(), words.size() * 4);
    comp_ptr = std::make_unique<spirv_cross::CompilerGLSL>(std::move(words));
    spirv_cross::CompilerGLSL& comp = *comp_ptr;
    spirv_cross::CompilerGLSL::Options o;
    o.version = 430;
    o.es = false;
    o.vulkan_semantics = false;
    o.separate_shader_objects = true;
    o.enable_420pack_extension = false;
    o.emit_push_constant_as_uniform_buffer = true;
    comp.set_common_options(o);
    auto res = comp.get_shader_resources();
    auto remap = [&](spirv_cross::SmallVector<spirv_cross::Resource>& v) {
      for (auto& r : v) {
        uint32_t set = comp.get_decoration(r.id, spv::DecorationDescriptorSet);
        uint32_t bind = comp.get_decoration(r.id, spv::DecorationBinding);
        comp.unset_decoration(r.id, spv::DecorationDescriptorSet);
        comp.set_decoration(r.id, spv::DecorationBinding, set * 8 + bind);
      }
    };
    out.samplers.clear();
    if (!res.separate_images.empty()) {
      comp.build_dummy_sampler_for_combined_images();
      comp.build_combined_image_samplers();
      uint32_t index = 0;
      for (auto& c : comp.get_combined_image_samplers()) {
        CombinedSampler cs;
        cs.unit = texture_unit_base + index++;
        cs.texture_binding = int32_t(comp.get_decoration(c.image_id, spv::DecorationBinding));
        if (comp.has_decoration(c.sampler_id, spv::DecorationBinding)) {
          uint32_t sampler_binding = comp.get_decoration(c.sampler_id, spv::DecorationBinding);
          cs.sampler_binding = int32_t(sampler_binding) - int32_t(texture_count);
        } else {
          cs.sampler_binding = -1;
        }
        out.samplers.push_back(cs);
        comp.set_decoration(c.combined_id, spv::DecorationBinding, cs.unit);
        comp.unset_decoration(c.combined_id, spv::DecorationDescriptorSet);
      }
      res.separate_images.clear();
      res.separate_samplers.clear();
    }
    for (auto& pc : res.push_constant_buffers) {
      comp.set_decoration(pc.id, spv::DecorationBinding, kUboBindingPushConstants);
    }
    remap(res.uniform_buffers);
    remap(res.storage_buffers);
    remap(res.sampled_images);
    remap(res.storage_images);
    out.source = comp.compile();
    RewriteSharedMemoryAsTbo(out.source);
    if (out.source.find("gl_CullDistance") != std::string::npos) {
      size_t nl = out.source.find('\n');
      out.source.insert(nl + 1, "#extension GL_ARB_cull_distance : require\n");
    }
    return true;
  } catch (const std::exception& e) {
    error = e.what();
    static int dumps = 0;
    if (comp_ptr && dumps++ < 2) {
      DumpBlocks(*comp_ptr);
    }
    return false;
  }
}

}  // namespace rex::graphics::gl
