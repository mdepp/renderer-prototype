#ifndef RENDERER_PROTOTYPE_MODEL_HPP
#define RENDERER_PROTOTYPE_MODEL_HPP

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <memory>
#include <filesystem>
#include "Buffer.hpp"


namespace model {
    class TexturedPhongMesh {
    public:
        TexturedPhongMesh() : has_tex_coords(false), has_diffuse_texture(false) {}

        // Mandatory
        std::vector<size_t> indices;
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;

        // Optional or switched

        bool has_tex_coords;
        std::vector<glm::vec2> texcoords;

        bool has_diffuse_texture;
        std::shared_ptr<buffer::Buffer<glm::vec3>> diffuse_texture;
        std::vector<glm::vec3> diffuse_colour;
    };

    std::vector<TexturedPhongMesh> load_meshes(const std::filesystem::path& filepath);
}

#endif //RENDERER_PROTOTYPE_MODEL_HPP
