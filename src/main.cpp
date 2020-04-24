/**
 * @file Main source file
 */
#include <iostream>
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include "Buffer.hpp"
#include "ApplicationWindow.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <chrono>
#include "Model.hpp"


using buffer::Buffer;
using app::ApplicationWindow;
using texture::load_texture;
using model::load_meshes;
using namespace std::filesystem;

void apply_matrix(const std::vector<glm::vec3>& vertices, const glm::mat4& matrix, std::vector<glm::vec4>& result) {
    result.resize(vertices.size());
    std::transform(vertices.cbegin(), vertices.cend(), result.begin(), [&matrix](const auto& vertex) {
        return matrix * glm::vec4(vertex, 1.f);
    });
}
void apply_matrix(const std::vector<glm::vec3>& vertices, const glm::mat4& matrix, std::vector<glm::vec3>& result) {
    result.resize(vertices.size());
    std::transform(vertices.cbegin(), vertices.cend(), result.begin(), [&matrix](const auto& vertex) {
        return glm::vec3(matrix * glm::vec4(vertex, 1.f));
    });
}
void apply_matrix_linear(const std::vector<glm::vec3>& vertices, const glm::mat4& matrix, std::vector<glm::vec3>& result) {
    result.resize(vertices.size());
    std::transform(vertices.cbegin(), vertices.cend(), result.begin(), [&matrix](const auto& vertex) {
        return glm::vec3(matrix * glm::vec4(vertex, 0.f));
    });
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
 * @brief Computes barycentric coordinates of a point given a 2d triangle
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
void perspective_divide(const std::vector<glm::vec4>& positions_clip, std::vector<glm::vec2>& positions_ndc) {
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
 * @param functor The function to execute. Should have a signature like `void(glm::ivec3 face_indices, glm::ivec2 position_window, glm::vec3 barycentric)`
 *     where `face_indices` are the indices of each vertex in the face, `position_window` is the window space coordinates
 *     of the pixel, and `barycentric` is the clip-space barycentric coordinates of the pixel.
 */
template<typename Functor>
void fragment_shader_pass(const std::vector<size_t>& indices, const std::vector<glm::vec4>& positions_clip, const std::vector<glm::vec2>& positions_window, Functor&& functor) {
    assert(indices.size()%3 == 0);
    assert(positions_window.size() == positions_clip.size());

    for (size_t i=0; i<indices.size(); i += 3) {
        const auto i1 = indices[i];
        const auto i2 = indices[i+1];
        const auto i3 = indices[i+2];
        const auto face_window = Face(positions_window[i1], positions_window[i2], positions_window[i3]);
        const auto face_clip = Face(positions_clip[i1], positions_clip[i2], positions_clip[i3]);
        for_each_pixel(face_window, face_clip, [&](const glm::ivec2& position_window, const glm::vec3& barycentric) {
            functor(glm::ivec3(i1, i2, i3), position_window, barycentric);
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


/**
 * @brief Creates a square matrix from a list of rows.
 *
 * To see why the input is a list of rows, consider this example:
 * <pre>@code
 *     make_matrix<2>({
 *       {1, 2},
 *       {3, 4}
 *     });
 * @endcode</pre>
 * This will make the matrix with columns (1,3) and (2,4), just as it appears visually.
 *
 * @tparam Size Size of the matrix to square matrix to create. If the size of the initializer lists does not match,
 *     an assertion is thrown.
 * @param args An initializer_list of rows, each row an initializer_list of elements of the matrix.
 * @return A matrix with the specified elements.
 */
template<size_t Size>
glm::mat<Size, Size, float> make_matrix(std::initializer_list<std::initializer_list<float>> args) {
    float data[Size*Size];
    auto index = 0;
    assert(args.size() == Size);
    for (auto row : args) {
        assert(row.size() == Size);
        for (auto element : row) {
            data[index++] = element;
        }
    }
    glm::mat<Size, Size, float> matrix;
    std::memcpy(glm::value_ptr(matrix), data, sizeof(data));
    return glm::transpose(matrix);
}

/**
 * Makes the perspective transform which maps view space to clip space.
 *
 * @param fovy Vertical FOV
 * @param aspect_ratio Ratio of screen width to height
 * @param near_depth Distance to near plane (positive)
 * @param far_depth Distance to far plane (positive)
 * @return Perspective transform from view space to clip space
 */
glm::mat4 make_perspective_transform(float fovy, float aspect_ratio, float near_depth, float far_depth) {
    auto perspective = glm::perspective(fovy, aspect_ratio, near_depth, far_depth);
    return make_matrix<4>({
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, -1, 0},
        {0, 0, 0, 1}
    }) * perspective;
}

/**
 * Sample a texture at specific texture coordinate
 * @tparam T Type of pixel in the texture (e.g. glm::vec3)
 * @param buffer The texture to sample
 * @param texcoord Texture coordinate to sample. (0,0) samples buffer.at({0,0}); (1,1)
 *     samples buffer.at({width-1, height-1}). Values outside this range are wrapped to repeat the texture.
 * @return Pixel value of the texture at the coordinate
 */
template<typename T>
inline T sample_texture(const Buffer<T>& buffer, const glm::vec2& texcoord) {
    const auto x = (static_cast<int>(texcoord.x * (buffer.width()-1)) + buffer.width()) % buffer.width();
    const auto y = (static_cast<int>(texcoord.y * (buffer.height()-1)) + buffer.height()) % buffer.height();
    return buffer.at({x, y});
}

/**
 * @brief Calculates Phong reflection model for a point and a single light.
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model
 *
 * While the parameter descriptions refer to world frame, the coordinate frame of the parameters doesn't matter so long
 * as it is consistent and there is only rotation/translation between the world frame and the frame used.
 *
 * @param camera_position Position of the camera in world frame
 * @param point_position Position of the point in world frame
 * @param point_normal Surface normal of the point in world frame
 * @param point_ambient Ambient colour of the point
 * @param point_diffuse Diffuse colour of the point
 * @param point_specular Specular colour of the point
 * @param point_shininess Shininess of the point
 * @param light_position Position of the light
 * @param light_ambient Ambient colour of the light (if using multiple lights, only a single light should have an
 *     ambient component)
 * @param light_diffuse Diffuse colour of the light
 * @param light_specular Specular colour of the light
 * @return The colour of the point as lit by the light. The effects of multiple lights are linear.
 */
glm::vec3 phong_reflection(const glm::vec3& camera_position,
                           const glm::vec3& point_position,
                           const glm::vec3& point_normal,
                           const glm::vec3& point_ambient,
                           const glm::vec3& point_diffuse,
                           const glm::vec3& point_specular,
                           const float point_shininess,
                           const glm::vec3& light_position,
                           const glm::vec3& light_ambient,
                           const glm::vec3& light_diffuse,
                           const glm::vec3& light_specular) {
    const auto L = glm::normalize(light_position - camera_position);
    const auto R = glm::normalize(2*glm::dot(L, point_normal) * point_normal - L);
    const auto V = glm::normalize(camera_position - point_position);

    const auto LN = glm::dot(L, point_normal);
    const auto RV = glm::dot(R, V);

    auto result = light_ambient * point_ambient;
    if (LN > 0) {
        result += light_diffuse * point_diffuse * LN;
        if (RV > 0) {
            result += light_specular * point_specular * glm::pow(RV, point_shininess);
        }
    }
    return glm::min(result, glm::vec3(1.f));
}

int main() {
    // Configuration
    const int SCREEN_WIDTH = 640;
    const int SCREEN_HEIGHT = 480;
    const float ROTATION_PERIOD = 5.f;
    const float TAU = 6.283185307179586476925286766559005768394338798750211641949f;
    const float FOV = TAU/4.f;
    const float NEAR_PLANE = 0.1f;
    const float FAR_PLANE = 10.f;

    // Define geometry
    const std::string model_name = "DamagedHelmet";
    const auto model_path = path("../glTF-Sample-Models/2.0") / model_name / path("glTF") / (model_name + ".gltf");
    const auto meshes = load_meshes(model_path);

    // Define buffers
    Buffer<glm::vec3> screen_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);
    Buffer<float> depth_buffer(SCREEN_WIDTH, SCREEN_HEIGHT, std::numeric_limits<float>::lowest());
    Buffer<glm::vec3> positions_world_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);
    Buffer<glm::vec3> diffuse_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);
    Buffer<glm::vec3> normals_world_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);

    // Define transforms
    const auto window_transform = make_matrix<3>({
        {SCREEN_WIDTH/2.f, 0.f, SCREEN_WIDTH/2.f},
        {0.f, SCREEN_HEIGHT/2.f, SCREEN_HEIGHT/2.f},
        {0.f, 0.f, 1.f}
    });
    const auto camera_pose_transform = make_matrix<4>({
        {1.f,  0.f, 0.f, 0.f},
        {0.f,  1.f, 0.f, 0.f},
        {0.f,  0.f, 1.f, 0.f},
        {0.f,  0.f, 0.f, 1.f}
    });
    const auto perspective_transform = make_perspective_transform(FOV, (float)SCREEN_WIDTH/SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE);
    auto model_transform = glm::mat4(1.f);

    // Results of transforms
    std::vector<glm::vec3> positions_world;
    std::vector<glm::vec3> normals_world;
    std::vector<glm::vec3> positions_view;
    std::vector<glm::vec4> positions_clip;
    std::vector<glm::vec2> positions_ndc;
    std::vector<glm::vec2> positions_window;

    // Create a window and render things
    ApplicationWindow window(SCREEN_WIDTH, SCREEN_HEIGHT);

    const auto start_time = std::chrono::system_clock::now();
    while (window.poll_events()) {
        // Update model transform
        const auto elapsed_seconds = std::chrono::duration<float>(std::chrono::system_clock::now() - start_time).count();
        const auto theta = elapsed_seconds / ROTATION_PERIOD * TAU;
        model_transform = make_matrix<4>({
            {std::cos(theta), 0.f, -std::sin(theta),  0.f},
            {0.f,             1.f, 0.f,              0.f},
            {std::sin(theta), 0.f, std::cos(theta), -2.f},
            {0.f,             0.f, 0.f,              1.f}
        });

        depth_buffer.clear();
        positions_world_buffer.clear();
        diffuse_buffer.clear();
        for (const auto& mesh : meshes) {
            // Transform vertices to window coordinates
            apply_matrix(mesh.positions, model_transform, positions_world);
            apply_matrix_linear(mesh.normals, model_transform, normals_world);
            apply_matrix(positions_world, camera_pose_transform, positions_view);
            apply_matrix(positions_view, perspective_transform, positions_clip);
            perspective_divide(positions_clip, positions_ndc);
            apply_matrix(positions_ndc, window_transform, positions_window);

            // Update position + colour buffers from geometry
            fragment_shader_pass(mesh.indices, positions_clip, positions_window,
                                 [&](const glm::ivec3 &face_indices, const glm::ivec2 &position_window,
                                     const glm::vec3 &barycentric) {

                                     if (position_window.x < 0 || position_window.x >= SCREEN_WIDTH) return;
                                     if (position_window.y < 0 || position_window.y >= SCREEN_HEIGHT) return;

                                     const auto position_clip = interpolate(positions_clip, face_indices, barycentric);
                                     if (position_clip.z >= 0) return;
                                     if (position_clip.z <= depth_buffer.at(position_window)) return;
                                     depth_buffer.at(position_window) = position_clip.z;

                                     const auto position_world = interpolate(positions_world, face_indices, barycentric);
                                     const auto normal_world = glm::normalize(interpolate(normals_world, face_indices, barycentric));
                                     const auto texcoord = interpolate(mesh.texcoords, face_indices, barycentric);

                                     glm::vec3 diffuse;
                                     if (mesh.has_tex_coords && mesh.has_diffuse_texture) {
                                         diffuse = sample_texture(*mesh.diffuse_texture, texcoord);
                                     } else {
                                         diffuse = interpolate(mesh.diffuse_colour, face_indices, barycentric);
                                     }

                                     positions_world_buffer.at(position_window) = position_world;
                                     normals_world_buffer.at(position_window) = normal_world;
                                     diffuse_buffer.at(position_window) = diffuse;
                                 });

        }
        // Render geometry from buffers
        screen_buffer.for_each_pixel([&](const glm::ivec2 &position) {
            screen_buffer.at(position) = phong_reflection({0.f, 0.f, 0.f},
                                                          positions_world_buffer.at(position),
                                                          normals_world_buffer.at(position),
                                                          diffuse_buffer.at(position),
                                                          diffuse_buffer.at(position),
                                                          diffuse_buffer.at(position),
                                                          1.f,
                                                          {-1.f, 0.f, 0.f},
                                                          {0.5f, 0.5f, 0.5f},
                                                          {0.5f, 0.5f, 0.5f},
                                                          {1.f, 1.f, 1.f});
        });
        window.draw(screen_buffer);
    }

    return 0;
}
