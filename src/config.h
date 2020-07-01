#ifndef RENDERER_PROTOTYPE_CONFIG_H
#define RENDERER_PROTOTYPE_CONFIG_H

#include <execution>

namespace config {

#ifdef NO_PARALLEL
    const auto par = std::execution::seq;
    const auto par_unseq = std::execution::unseq;
#else
    const auto par = std::execution::par;
    const auto par_unseq = std::execution::par_unseq;
#endif
    const auto seq = std::execution::seq;
    const auto unseq = std::execution::unseq;
}

#endif //RENDERER_PROTOTYPE_CONFIG_H
