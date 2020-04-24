#include "texture.hpp"
#include <SDL2/SDL_image.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include "util.hpp"

namespace texture {

    buffer::Buffer<glm::vec3> load_texture(const std::filesystem::path &filepath) {
        spdlog::info("Loading texture {}", filepath.string());
        auto loaded_surface = IMG_Load(filepath.c_str());
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
        for (int y=0; y<converted_surface->h; ++y) {
            for(int x=0; x<converted_surface->w; ++x) {
                auto pixel = pixels[x+(converted_surface->h-1-y)*converted_surface->w];
                auto r = (pixel >> (2*BITS_PER_CHANNEL)) & CHANNEL_MASK;
                auto g = (pixel >> BITS_PER_CHANNEL) & CHANNEL_MASK;
                auto b = pixel & CHANNEL_MASK;
                buffer.at(glm::ivec2(x, y)) = glm::vec3((float)r/0xFF, (float)g/0xFF, (float)b/0xFF);
            }
        }
        SDL_FreeSurface(converted_surface);
        spdlog::info("Finished loading texture {}", filepath.string());
        return buffer;
    }

    void save_texture(const buffer::Buffer<glm::vec3> &texture, const std::string &filename) {
        buffer::Buffer<uint32_t> flipped_texture(texture.width(), texture.height());
        texture.for_each_pixel([&](const glm::ivec2& position) {
            auto colour = texture.at(position);
            auto r = static_cast<uint32_t>(colour.r * 0xFF);
            auto g = static_cast<uint32_t>(colour.g * 0xFF);
            auto b = static_cast<uint32_t>(colour.b * 0xFF);
            auto rgb = (r << 16) + (g << 8) + b;
            flipped_texture.at(glm::ivec2(position.x, texture.height() - 1 - position.y)) = rgb;
        });
        auto surface = SDL_CreateRGBSurfaceFrom((void *) flipped_texture.data(),
                                                flipped_texture.width(),
                                                flipped_texture.height(),
                                                sizeof(uint32_t) * 8,
                                                flipped_texture.width() * sizeof(uint32_t),
                                                0xFF0000,
                                                0x00FF00,
                                                0x0000FF,
                                                0);
        if (!surface) {
            throw util::SDLException("Cannot create RGB surface from buffer");
        }
        IMG_SaveJPG(surface, filename.c_str(), 85);
    }

}