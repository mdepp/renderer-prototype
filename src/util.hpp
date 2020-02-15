#ifndef RENDERER_PROTOTYPE_UTIL_HPP
#define RENDERER_PROTOTYPE_UTIL_HPP

#include <stdexcept>
#include <string>

namespace util {
    /**
     * Exception for SDL errors. Automatically adds the contents of SDL_GetError() to the message.
     */
    class SDLException : public std::exception {
    public:
        explicit SDLException(const std::string &message);

        const char *what() const noexcept override;

    private:
        std::string m_message;
    };
}

#endif //RENDERER_PROTOTYPE_UTIL_HPP
