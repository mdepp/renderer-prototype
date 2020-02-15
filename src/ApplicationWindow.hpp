#ifndef RENDERER_PROTOTYPE_APPLICATIONWINDOW_HPP
#define RENDERER_PROTOTYPE_APPLICATIONWINDOW_HPP

#include <SDL.h>
#include "Buffer.hpp"

namespace app {

    class ApplicationWindow {
    public:
        explicit ApplicationWindow(size_t width, size_t height);
        ApplicationWindow(const ApplicationWindow&) = delete;
        ApplicationWindow& operator = (const ApplicationWindow&) = delete;

        void draw(const buffer::Buffer<glm::vec3>& texture);

        bool poll_events() {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) return false;
            }
            return true;
        }

        virtual ~ApplicationWindow() {
            SDL_DestroyTexture(m_screen_texture);
            SDL_DestroyRenderer(m_renderer);
            SDL_DestroyWindow(m_window);
        }

    private:
        const size_t m_width;
        const size_t m_height;
        SDL_Window* m_window;
        SDL_Renderer* m_renderer;
        buffer::Buffer<uint32_t> m_screen_buffer;
        SDL_Texture* m_screen_texture;
    };

}


#endif //RENDERER_PROTOTYPE_APPLICATIONWINDOW_HPP
