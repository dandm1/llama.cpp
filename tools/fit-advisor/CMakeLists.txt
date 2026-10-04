# llama-fit-advisor-impl: fit-advisor logic, reusable by tests and other tools

set(TARGET llama-fit-advisor-impl)

add_library(${TARGET} inventory.cpp probe.cpp allocation.cpp measure.cpp cost.cpp search.cpp emit.cpp validate.cpp fit-advisor.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

# the placement is solved exactly as a mixed-integer program with HiGHS (MIT); without it the tool falls back to
# the simulated-annealing search
option(FIT_ADVISOR_HIGHS "llama-fit-advisor: exact placement with the HiGHS MIP solver (fetched at configure time)" ON)
if (FIT_ADVISOR_HIGHS)
    include(FetchContent)
    set(FAST_BUILD     ON  CACHE BOOL "" FORCE)
    set(BUILD_TESTING  OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ZLIB           OFF CACHE BOOL "" FORCE)
    set(HIGHS_NO_DEFAULT_THREADS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(highs
        GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
        GIT_TAG        v1.15.1
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(highs)
    target_link_libraries(${TARGET} PRIVATE highs)
    target_compile_definitions(${TARGET} PRIVATE FIT_ADVISOR_HIGHS)
    target_sources(${TARGET} PRIVATE solve.cpp)
endif()

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

