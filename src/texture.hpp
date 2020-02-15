#ifndef RENDERER_PROTOTYPE_TEXTURE_HPP
#define RENDERER_PROTOTYPE_TEXTURE_HPP

#include <glm/glm.hpp>
#include "Buffer.hpp"
#include "util.hpp"
#include <string>
#include <SDL2/SDL_image.h>

namespace texture {
    /**
     * Loads an image from a file into an rgb buffer
     * @param filename Filename of the image to load. Any format that can be handled by IMG_Load works.
     * @return A buffer containing the rgb image pixels. `(0,0)` is bottom left and `(width-1,height-1)` is top right.
     *     Each pixel has the range `[0,1]^3`.
     * @throws SDLException if image loading fails or if the image cannot be converted into the necessary format.
     */
    buffer::Buffer<glm::vec3> load_texture(const std::string& filename) {
        auto loaded_surface = IMG_Load(filename.c_str());
        if (!loaded_surface) {
            throw util::SDLException("Failed to load texture");
        }
        auto converted_surface = SDL_ConvertSurfaceFormat(loaded_surface, SDL_PIXELFORMAT_RGB888, 0);
        if (!converted_surface) {
            throw util::SDLException("Failed to convert texture");
        }
        SDL_FreeSurface(loaded_surface);

        constexpr uint32_t BITS_PER_CHANNEL = 8;
        constexpr uint32_t CHANNEL_MASK = 0xFF;

        buffer::Buffer<glm::vec3> buffer(converted_surface->w, converted_surface->h);
        auto pixels = static_cast<uint32_t *>(converted_surface->pixels);
        for (size_t y=0; y<converted_surface->h; ++y) {
            for(size_t x=0; x<converted_surface->w; ++x) {
                auto pixel = pixels[x+(converted_surface->h-1-y)*converted_surface->w];
                auto r = (pixel >> (2*BITS_PER_CHANNEL)) & CHANNEL_MASK;
                auto g = (pixel >> BITS_PER_CHANNEL) & CHANNEL_MASK;
                auto b = pixel & CHANNEL_MASK;
                buffer.at(glm::ivec2(x, y)) = glm::vec3((float)r/0xFF, (float)g/0xFF, (float)b/0xFF);
            }
        }
        SDL_FreeSurface(converted_surface);
        return buffer;
    }
}

#endif //RENDERER_PROTOTYPE_TEXTURE_HPP
