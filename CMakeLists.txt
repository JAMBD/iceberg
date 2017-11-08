cmake_minimum_required(VERSION 2.8)
project( iceberg )
find_package( OpenCV REQUIRED )
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -O2 -g -ftest-coverage -fprofile-arcs")

add_executable( icebergTrainer icebergTrainer.cpp )
target_link_libraries( icebergTrainer ${JSONCPP_LIBRARIES})
target_link_libraries( icebergTrainer ${OpenCV_LIBS} )
