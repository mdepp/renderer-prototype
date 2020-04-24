#ifndef RENDERER_PROTOTYPE_TEXTURE_HPP
#define RENDERER_PROTOTYPE_TEXTURE_HPP

#include <glm/glm.hpp>
#include "Buffer.hpp"
#include <filesystem>
#include <string>

namespace texture {
    /**
     * Loads an image from a file into an rgb buffer
     * @param filename Filename of the image to load. Any format that can be handled by IMG_Load works.
     * @return A buffer containing the rgb image pixels. `(0,0)` is bottom left and `(width-1,height-1)` is top right.
     *     Each pixel has the range `[0,1]^3`.
     * @throws SDLException if image loading fails or if the image cannot be converted into the necessary format.
     */
    buffer::Buffer<glm::vec3> load_texture(const std::filesystem::path& filepath);

    void save_texture(const buffer::Buffer<glm::vec3>& texture, const std::string& filename);
}

#endif //RENDERER_PROTOTYPE_TEXTURE_HPP
