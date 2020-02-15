//
// Created by mitchell on 2020-02-14.
//

#include "util.hpp"
#include <SDL2/SDL.h>

util::SDLException::SDLException(const std::string &message) {
    m_message = message + std::string(": ") + SDL_GetError();
}

const char *util::SDLException::what() const noexcept {
    return m_message.c_str();
}
