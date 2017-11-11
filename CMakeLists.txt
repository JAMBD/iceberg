cmake_minimum_required(VERSION 2.8)
project( iceberg )
find_package( OpenCV REQUIRED )

add_subdirectory( jsoncpp )
set(JSONCPP_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp/src/lib_json/libjsoncpp.a)
set(JSONCPP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp/include)

include_directories( JSONCPP_INCLUDE )

list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -O2 -g -ftest-coverage -fprofile-arcs")

add_executable( icebergTrainer icebergTrainer.cpp )

target_link_libraries( icebergTrainer ${JSONCPP_LIBRARY} )
target_link_libraries( icebergTrainer ${OpenCV_LIBS} )
