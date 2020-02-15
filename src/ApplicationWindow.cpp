#include "ApplicationWindow.hpp"
#include "util.hpp"
#include <glm/glm.hpp>

app::ApplicationWindow::ApplicationWindow(const size_t width, const size_t height)
        : m_width(width), m_height(height), m_window(nullptr), m_renderer(nullptr), m_screen_buffer(width, height) {
    SDL_CreateWindowAndRenderer(width, height, SDL_WINDOW_SHOWN, &m_window, &m_renderer);
    if (!m_window || !m_renderer) {
        throw util::SDLException("Failed to create SDL window and renderer");
    }
    m_screen_texture = SDL_CreateTexture(m_renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, m_width, m_height);
    if (!m_screen_texture) {
        throw util::SDLException("Failed to create buffer texture");
    }
}

void app::ApplicationWindow::draw(const buffer::Buffer<glm::vec3> &texture) {
    SDL_PixelFormat pixel_format;
    texture.for_each_pixel([&](const glm::ivec2& position) {
        const auto colour = texture.at(position);
        const uint32_t r = colour.r * 255;
        const uint32_t g = colour.g * 255;
        const uint32_t b = colour.b * 255;
        auto val = (r << 16) + (g << 8) + b;
        m_screen_buffer.at(glm::ivec2(position.x, m_height-1-position.y)) = val;
    });
    SDL_UpdateTexture(m_screen_texture, nullptr, m_screen_buffer.data(), sizeof(uint32_t)*m_width);
    SDL_RenderCopy(m_renderer, m_screen_texture, nullptr, nullptr);
    SDL_RenderPresent(m_renderer);
}
