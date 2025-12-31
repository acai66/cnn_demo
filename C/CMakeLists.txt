cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(cnn_demo VERSION 1.0.0)

include_directories(
	include
)

add_executable(demo "cnn_demo.c")

if(UNIX)
	target_link_libraries(demo dl)
endif()

install(TARGETS demo 
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
