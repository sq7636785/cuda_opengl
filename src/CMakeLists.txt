set(SOURCE_FILES
    "image.cpp"
    "image.h"
    "interactions.h"
    "intersections.h"
    "glslUtility.h"
    "glslUtility.cpp"
    "pathtrace_my.cu"
    "path_tracer.h"
    "scene.cpp"
    "scene.h"
    "data_structure"
    "preview.h"
    "preview.cpp"
    "utilities.cpp"
    "utilities.h"
    "tiny_obj_loader.h"
    "tiny_obj_loader.cc"
    "bounds.h"
    "bounds.cpp"
    "bvh.h"
    "bvh.cpp"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
