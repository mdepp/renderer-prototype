#include "Model.hpp"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <stdexcept>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <gsl/span>
#include "Buffer.hpp"
#include "texture.hpp"
#include <filesystem>

using buffer::Buffer;
using texture::load_texture;

namespace model {
    auto _load_texture(const aiScene* scene, const aiMesh* in_mesh, const aiTextureType texture_type, const std::filesystem::path& model_dir) {
        bool has_texture = false;
        std::shared_ptr<Buffer<glm::vec3>> texture;

        const auto* material = scene->mMaterials[in_mesh->mMaterialIndex];
        const auto texture_count = material->GetTextureCount(texture_type);
        if (texture_count > 1) {
            spdlog::warn("Material has {} {} textures, but only 0 or 1 is supported", static_cast<int>(texture_count), static_cast<int>(texture_type));
        }
        has_texture = (texture_count != 0);
        if (has_texture) {
            aiString texture_path_ai;
            material->GetTexture(aiTextureType_DIFFUSE, 0, &texture_path_ai);
            auto texture_path = std::filesystem::path(texture_path_ai.C_Str());
            if (texture_path.is_relative()) {
                texture_path = model_dir / texture_path;
            }
            texture = std::make_shared<Buffer<glm::vec3>>(load_texture(texture_path));
        }
        return std::make_tuple(has_texture, texture);
    }

    std::vector<TexturedPhongMesh> load_meshes(const std::filesystem::path& filepath) {
        spdlog::info("Loading model {}", filepath.string());
        auto meshes = std::vector<TexturedPhongMesh>();

        auto importer = Assimp::Importer();
        const auto postprocess_flags = aiProcess_GenNormals
                | aiProcess_Triangulate
                | aiProcess_JoinIdenticalVertices;
        const auto* scene = importer.ReadFile(filepath, postprocess_flags);
        if (scene == nullptr) {
            throw std::runtime_error(importer.GetErrorString());
        }

        for(size_t mesh_index=0; mesh_index < scene->mNumMeshes; ++mesh_index) {
            const auto* in_mesh = scene->mMeshes[mesh_index];

            TexturedPhongMesh out_mesh;

            assert(in_mesh->HasPositions());
            assert(in_mesh->HasNormals());
            assert(in_mesh->HasFaces());
            const gsl::span vertices(in_mesh->mVertices, in_mesh->mNumVertices);
            const gsl::span normals(in_mesh->mNormals, in_mesh->mNumVertices);
            const gsl::span faces(in_mesh->mFaces, in_mesh->mNumFaces);

            for (const auto& face : faces) {
                assert(face.mNumIndices == 3);
                out_mesh.indices.insert(out_mesh.indices.end(), face.mIndices, face.mIndices+face.mNumIndices);
            }
            std::transform(std::cbegin(vertices), std::cend(vertices), std::back_inserter(out_mesh.positions),
                           [](const auto& vertex) { return glm::vec3(vertex.x, vertex.y, vertex.z); });
            std::transform(std::cbegin(normals), std::cend(normals), std::back_inserter(out_mesh.normals),
                           [](const auto& normal) { return glm::vec3(normal.x, normal.y, normal.z); });
            if (in_mesh->HasVertexColors(0)) {
                const gsl::span colours(in_mesh->mColors[0], in_mesh->mNumVertices);
                std::transform(std::cbegin(colours), std::cend(colours), std::back_inserter(out_mesh.diffuse_colour),
                               [](const auto& colour) { return glm::vec3(colour.r, colour.g, colour.b); });
            } else {
                std::transform(std::cbegin(vertices), std::cend(vertices), std::back_inserter(out_mesh.diffuse_colour),
                               [](const auto& colour) { return glm::vec3(1.f); });
            }

            std::tie(out_mesh.has_diffuse_texture, out_mesh.diffuse_texture) =
                    _load_texture(scene, in_mesh, aiTextureType_DIFFUSE, filepath.parent_path());

            if (in_mesh->HasTextureCoords(0)) {
                out_mesh.has_tex_coords = true;
                const gsl::span texcoords(in_mesh->mTextureCoords[0], in_mesh->mNumVertices);
                std::transform(std::cbegin(texcoords), std::cend(texcoords), std::back_inserter(out_mesh.texcoords),
                              [](const auto& coord) { return glm::vec3(coord.x, coord.y, coord.z); });
            }

            meshes.push_back(std::move(out_mesh));
        }
        spdlog::info("Finished loading model {}", filepath.string());
        return meshes;
    }
}