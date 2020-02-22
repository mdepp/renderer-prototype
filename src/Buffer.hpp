#ifndef RENDERER_PROTOTYPE_BUFFER_HPP
#define RENDERER_PROTOTYPE_BUFFER_HPP


#include <cstdio>
#include <glm/glm.hpp>
#include <vector>


namespace buffer {
/**
 * A two-dimensional array. Memory is stored in row-major order, so if iterating through it should iterate by y, then
 * by x.
 * @tparam T Type of data to store in the array (e.g. vectors).
 */
    template<typename T>
    class Buffer {
    public:
        /**
         * @brief Creates and clears the buffer.
         *
         * @param width Width of the buffer
         * @param height Height of the buffer
         * @param clear_value Value to clear buffer to.
         */
        explicit Buffer(const size_t width, const size_t height, const T clear_value = T())
                : m_width(width), m_height(height), m_clear_value(clear_value) {
            assert(m_width > 0);
            assert(m_height > 0);
            m_items.resize(width * height);
            clear();
        }


        /**
         * @brief Returns the buffer value at the given xy position.
         *
         * @param position xy coordinates of the value. Should be an integer in [0,width-1]x[0,height-1].
         * @return Value of the buffer at the given position.
         */
        [[nodiscard]] const T &at(const glm::ivec2 &position) const {
            assert(position.x >= 0);
            assert(position.x < m_width);
            assert(position.y >= 0);
            assert(position.y < m_height);
            return m_items[m_width * position.y + position.x];
        }

        /**
         * @brief Returns the buffer value at the given xy position.
         *
         * @param position xy coordinates of the value. Should be an integer in [0,width-1]x[0,height-1].
         * @return Value of the buffer at the given position. This can be modified.
         */
        T &at(const glm::ivec2 &position) {
            assert(position.x >= 0);
            assert(position.x < m_width);
            assert(position.y >= 0);
            assert(position.y < m_height);
            return m_items[m_width * position.y + position.x];
        }

        /**
         * @brief Clears the buffer back to the default clear value.
         */
        void clear() {
            std::fill(m_items.begin(), m_items.end(), m_clear_value);
        }

        /**
         * @brief Executes a function once per pixel in the buffer.
         *
         * @tparam Functor
         * @param functor Function to execute. Should have a signature something like `void(glm::ivec2 position)`,
         *     where `position` is the coordinates of some pixel. This function is called once for every pixel in the
         *     buffer.
         */
        template<typename Functor>
        void for_each_pixel(Functor &&functor) const {
            glm::ivec2 position;
            for (position.y = 0; position.y < m_height; ++position.y) {
                for (position.x = 0; position.x < m_width; ++position.x) {
                    functor(position);
                }
            }
        }

        /**
         * @brief Accesses raw buffer data.
         *
         * @return The underlying data of the buffer. Elements are stored contiguously in row-major order.
         */
        const T* data() const {
            return m_items.data();
        }

        /**
         * @brief Accesses buffer width.
         *
         * @return The width in pixels of the buffer.
         */
        auto width() const {
            return m_width;
        }

        /**
         * @brief Accesses buffer height.
         *
         * @return The height in pixels of the buffer.
         */
        auto height() const {
            return m_height;
        }

    private:
        const int m_width;
        const int m_height;
        const T m_clear_value;
        std::vector <T> m_items;
    };
}



#endif //RENDERER_PROTOTYPE_BUFFER_HPP
