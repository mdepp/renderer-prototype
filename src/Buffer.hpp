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

        template<typename Functor>
        void for_each_pixel(Functor &&functor) const {
            glm::ivec2 position;
            for (position.y = 0; position.y < m_height; ++position.y) {
                for (position.x = 0; position.x < m_width; ++position.x) {
                    functor(position);
                }
            }
        }

        const T* data() const {
            return m_items.data();
        }

    private:
        const size_t m_width;
        const size_t m_height;
        const T m_clear_value;
        std::vector <T> m_items;
    };
}



#endif //RENDERER_PROTOTYPE_BUFFER_HPP
