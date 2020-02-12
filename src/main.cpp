/**
 * @file Main source file
 */
#include <iostream>
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>


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


inline bool in_face(const glm::vec3& barycentric) {
    if (barycentric.x < 0 || barycentric.x > 1) return false;
    if (barycentric.y < 0 || barycentric.y > 1) return false;
    if (barycentric.z < 0 || barycentric.z > 1) return false;
    return barycentric.x + barycentric.y + barycentric.z <= 1;
}


template<typename Functor>
void for_each_pixel(const Face<glm::vec2>& face_window, const Face<glm::vec4>& face_clip, Functor&& functor) {
    const auto min = glm::min(face_window.p1, glm::min(face_window.p2, face_window.p3));
    const auto max = glm::max(face_window.p1, glm::max(face_window.p2, face_window.p3));

    const auto depth_inverses = 1.f / glm::vec3(face_clip.p1.w, face_clip.p2.w, face_clip.p3.w);

    for (int x = min.x; x <= max.x; ++x) {
        for (int y = min.y; y <= max.y; ++y) {
            const auto barycentric_window = compute_barycentric(glm::vec2(x, y),
                                                                glm::vec2(face_window.p1),
                                                                glm::vec2(face_window.p2),
                                                                glm::vec2(face_window.p3));
            if (!in_face(barycentric_window)) continue;
            const auto barycentric_clip = correct_barycentric(barycentric_window, depth_inverses);
            functor(barycentric_clip);
        }
    }
}

void perspective_transform(const std::vector<glm::vec4>& positions_clip, std::vector<glm::vec2> positions_ndc) {
    positions_ndc.resize(positions_clip.size());
    std::transform(positions_clip.cbegin(), positions_clip.cend(), positions_ndc.begin(), [](const glm::vec4& position_clip) {
        return glm::vec2(position_clip / position_clip.w);
    });
}


void render_faces(const std::vector<size_t>& indices, const std::vector<glm::vec4>& positions_clip, const std::vector<glm::vec2>& positions_window) {
    assert(indices.size()%3 == 0);
    assert(positions_window.size() == positions_clip.size());
    for (size_t i=0; i<indices.size(); i += 3) {
        const auto face_window = Face(positions_window[i], positions_window[i + 1], positions_window[i + 2]);
        const auto face_clip = Face(positions_clip[i], positions_clip[i + 1], positions_clip[i + 2]);
        for_each_pixel(face_window, face_clip, [](const glm::vec3& barycentric) {
            // Do something for each pixel (no depth checking yet)
        });
    }
}


int main() {
    const std::vector<glm::vec4> positions_clip = {
            {1.f, 1.f, 1.f, 1.f}
    };
    const std::vector<size_t> indices = {1};
    const auto window_transform = glm::mat3(1.f);

    std::vector<glm::vec2> positions_ndc;
    std::vector<glm::vec2> positions_window;

    perspective_transform(positions_clip, positions_ndc);
    apply_matrix(positions_ndc, window_transform, positions_window);

    render_faces(indices, positions_clip, positions_window);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
