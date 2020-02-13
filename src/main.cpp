/**
 * @file Main source file
 */
#include <iostream>
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <array>
#include <memory>
#include <numeric>


void apply_matrix(const std::vector<glm::vec3>& vertices, const glm::mat4& matrix, std::vector<glm::vec3>& result) {
    result.clear();
    for (const auto& vertex : vertices) {
        result.emplace_back(matrix * glm::vec4(vertex, 1.f));
    }
}
void apply_matrix(const std::vector<glm::vec2>& vertices, const glm::mat3& matrix, std::vector<glm::vec2>& result) {
    result.resize(vertices.size());
    std::transform(vertices.cbegin(), vertices.cend(), result.begin(), [&matrix](const auto& vertex) {
        return glm::vec2(matrix * glm::vec3(vertex, 1.f));
    });
}


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
    [[nodiscard]] const T& at(const glm::ivec2& position) const {
        assert(position.x >= 0);
        assert(position.x < m_width);
        assert(position.y >= 0);
        assert(position.y < m_height);
        return m_items[m_width*position.y + position.x];
    }

    /**
     * @brief Returns the buffer value at the given xy position.
     *
     * @param position xy coordinates of the value. Should be an integer in [0,width-1]x[0,height-1].
     * @return Value of the buffer at the given position. This can be modified.
     */
    T& at(const glm::ivec2& position) {
        assert(position.x >= 0);
        assert(position.x < m_width);
        assert(position.y >= 0);
        assert(position.y < m_height);
        return m_items[m_width*position.y + position.x];
    }

    /**
     * @brief Clears the buffer back to the default clear value.
     */
    void clear() {
        std::fill(m_items.begin(), m_items.end(), m_clear_value);
    }

private:
    const size_t m_width;
    const size_t m_height;
    const T m_clear_value;
    std::vector<T> m_items;
};


/**
 * A group of three vertices representing the face of a single triangle
 * @tparam T Type of vertex.
 */
template<class T = glm::vec3>
struct Face {
    explicit Face(T p1, T p2, T p3)
      : p1(std::move(p1)), p2(std::move(p2)), p3(std::move(p3)) {}
    T p1, p2, p3;
};


/**
 * @brief Compute barycentric coordinates of a point given a 2d triangle
 *
 * See https://en.wikipedia.org/wiki/Barycentric_coordinate_system
 *
 * @param pos Position of the pixel
 * @param p1  Coordinates of the first vertex
 * @param p2  Coordinates of the second vertex
 * @param p3  Coordinates of the third vertex
 * @return The barycentric coordinates of the point, as (lambda1, lambda2, lambda3)
 */
inline glm::vec3 compute_barycentric(const glm::vec2& pos, const glm::vec2& p1, const glm::vec2& p2, const glm::vec2& p3) {
    const auto denom = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
    glm::vec3 barycentric(0.f);
    barycentric.x = ( (p2.y - p3.y) * (pos.x - p3.x) + (p3.x - p2.x) * (pos.y - p3.y) ) / denom;
    barycentric.y = ( (p3.y - p1.y) * (pos.x - p3.x) + (p1.x - p3.x) * (pos.y - p3.y) ) / denom;
    barycentric.z = 1.f - barycentric.x - barycentric.y;
    return barycentric;
}

/**
 * @brief Calculates perspective-corrected barycentric coordinates.
 *
 * See https://stackoverflow.com/a/24460895
 *
 * @param barycentric Barycentric coordinates in window space
 * @param depth_inverse "depth inverse" of each vertex of the face. This is 1/w, where w is the vertex w coordinate
 *     used in homogonous -> 2D point transform (e.g. w-coordinate in clip space).
 * @return Perspective-corrected barycentric coordinates (lambda1, lambda2, lambda3).
 */
inline glm::vec3 correct_barycentric(const glm::vec3& barycentric, const glm::vec3 depth_inverses) {
    return barycentric * depth_inverses / glm::dot(barycentric, depth_inverses);
}

/**
 * @brief Determines if a point is within a single face (triangle)
 * @param barycentric The barycentric coordinates of the point with respect to the face
 * @return true if and only if the point is within the face
 */
inline bool in_face(const glm::vec3& barycentric) {
    if (barycentric.x < 0 || barycentric.x > 1) return false;
    if (barycentric.y < 0 || barycentric.y > 1) return false;
    if (barycentric.z < 0 || barycentric.z > 1) return false;
    return barycentric.x + barycentric.y + barycentric.z <= 1;
}

/**
 * @brief Executes a function once for every pixel in a face
 * @tparam Functor
 * @param face_window The face's position in window space
 * @param face_clip The face's position in clip space
 * @param functor The function to execute. Should have a signature something like `void(glm::ivec2 window_pos, glm::vec3 barycentric)`,
 *     where `window_pos` is the xy coordinates of the pixel in window space, and `barycentric` is the barycentric
 *     coordinates of the pixel in clip space.
 */
template<typename Functor>
void for_each_pixel(const Face<glm::vec2>& face_window, const Face<glm::vec4>& face_clip, Functor&& functor) {
    const auto min = glm::min(face_window.p1, glm::min(face_window.p2, face_window.p3));
    const auto max = glm::max(face_window.p1, glm::max(face_window.p2, face_window.p3));

    const auto depth_inverses = 1.f / glm::vec3(face_clip.p1.w, face_clip.p2.w, face_clip.p3.w);

    for (int y = min.y; y <= max.y; ++y) {
        for (int x = min.x; x <= max.x; ++x) {
            const auto barycentric_window = compute_barycentric(glm::vec2(x, y),
                                                                glm::vec2(face_window.p1),
                                                                glm::vec2(face_window.p2),
                                                                glm::vec2(face_window.p3));
            if (!in_face(barycentric_window)) continue;
            const auto barycentric_clip = correct_barycentric(barycentric_window, depth_inverses);
            functor(glm::ivec2(x, y), barycentric_clip);
        }
    }
}

/**
 * @brief Performs the perspective divide between clip space and NDC (normalized device coordinates) space.
 *
 * @param positions_clip Homogeneous coordinates in clip space
 * @param positions_ndc Coordinates in NDC space
 */
void perspective_transform(const std::vector<glm::vec4>& positions_clip, std::vector<glm::vec2> positions_ndc) {
    positions_ndc.resize(positions_clip.size());
    std::transform(positions_clip.cbegin(), positions_clip.cend(), positions_ndc.begin(), [](const glm::vec4& position_clip) {
        return glm::vec2(position_clip / position_clip.w);
    });
}

/**
 * @brief Given some geometry, execute some function for each pixel of each face.
 * @tparam Functor
 *     of the pixel, and `barycentric` is the clip-space barycentric coordinates of the pixel.
 * @param indices Indices defining face vertices. Each consecutive three indices should form a face.
 * @param positions_clip Vertex positions in clip space.
 * @param positions_window Vertex positions in window space.
 * @param functor The function to execute. Should have a signiture like `void(glm::ivec3 face_indices, glm::ivec2 position_window, glm::vec3 barycentric)`
 *     where `face_indices` are the indices of each vertex in the face, `position_window` is the window space coordinate
 */
template<typename Functor>
void fragment_shader_pass(const std::vector<size_t>& indices, const std::vector<glm::vec4>& positions_clip, const std::vector<glm::vec2>& positions_window, Functor&& functor) {
    assert(indices.size()%3 == 0);
    assert(positions_window.size() == positions_clip.size());
    for (size_t i=0; i<indices.size(); i += 3) {
        const auto face_window = Face(positions_window[i], positions_window[i + 1], positions_window[i + 2]);
        const auto face_clip = Face(positions_clip[i], positions_clip[i + 1], positions_clip[i + 2]);
        for_each_pixel(face_window, face_clip, [&](const glm::ivec2& position_window, const glm::vec3& barycentric) {
            functor(glm::ivec3(i, i+1, i+2), position_window, barycentric);
        });
    }
}

/**
 * @brief Interpolates vertices of a face to the value of some point on the face.
 *
 * @tparam T
 * @param vertices Vertices to interpolate. Could be positions, colours, normals, etc.
 * @param face_indices Indices defining the face.
 * @param barycentric Barycentric coordinates of the point on the face.
 * @return Interpolated value of the vertices at the point.
 */
template<typename T>
inline T interpolate(const std::vector<T>& vertices, const glm::ivec3& face_indices, const glm::vec3& barycentric) {
    return barycentric.x * vertices[face_indices.x] +
           barycentric.y * vertices[face_indices.y] +
           barycentric.z * vertices[face_indices.z];
}


int main() {
    // Define geometry
    const std::vector<glm::vec4> positions_clip = {
            {1.f, 0.f, 0.f, 1.f},
            {0.f, -1.f, 0.f, 1.f},
            {-1.f, 0.f, 0.f, 1.f}
    };
    const std::vector<glm::vec3> colours = {
            {1.f, 0.f, 0.f},
            {0.f, 1.f, 0.0},
            {0.f, 0.f, 0.f}
    };
    const std::vector<size_t> indices = {0, 1, 2};

    // Define buffers
    const size_t SCREEN_WIDTH = 640;
    const size_t SCREEN_HEIGHT = 480;
    Buffer<float> depth_buffer(SCREEN_WIDTH, SCREEN_HEIGHT, std::numeric_limits<float>::lowest());
    Buffer<glm::vec4> position_clip_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);
    Buffer<glm::vec3> colour_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);

    const auto window_transform = glm::mat3(1.f);

    // Transform vertices to window coordinates
    std::vector<glm::vec2> positions_ndc;
    std::vector<glm::vec2> positions_window;
    perspective_transform(positions_clip, positions_ndc);
    apply_matrix(positions_ndc, window_transform, positions_window);

    // Update position + colour buffers from geometry
    fragment_shader_pass(indices, positions_clip, positions_window, [&](const glm::ivec3& face_indices, const glm::ivec2& position_window, const glm::vec3& barycentric) {
        if (position_window.x < 0 || position_window.x >= SCREEN_WIDTH) return;
        if (position_window.y < 0 || position_window.y >= SCREEN_HEIGHT) return;

        const auto position_clip = interpolate(positions_clip, face_indices, barycentric);
        if (position_clip.z <= depth_buffer.at(position_window)) return;
        depth_buffer.at(position_window) = position_clip.z;

        const auto colour = interpolate(colours, face_indices, barycentric);

        position_clip_buffer.at(position_window) = position_clip;
        colour_buffer.at(position_window) = colour;
    });

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
