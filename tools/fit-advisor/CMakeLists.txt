# llama-fit-advisor-impl: fit-advisor logic, reusable by tests and other tools

set(TARGET llama-fit-advisor-impl)

add_library(${TARGET} inventory.cpp probe.cpp fit-advisor.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-fit-advisor executable

set(TARGET llama-fit-advisor)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-fit-advisor-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
