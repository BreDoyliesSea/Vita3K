add_library(
	savestate
	STATIC
	include/savestate/savestate.h
	src/savestate.cpp
)

target_include_directories(savestate PUBLIC include)
target_link_libraries(savestate PUBLIC emuenv mem)
target_link_libraries(savestate PRIVATE config cpu io kernel miniz nids util)
