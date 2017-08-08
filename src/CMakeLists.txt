
add_executable(gpu_image_filter gpu_image_filter.cpp)

target_link_libraries(gpu_image_filter ${OpenCV_LIBS} ${Boost_LIBS} ${CLEW_LIBS})

add_dependencies(gpu_image_filter libclew)
