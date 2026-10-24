add_library(
	savestate
	STATIC
	include/savestate/savestate.h
	src/kernel_snapshot.cpp
	src/kernel_snapshot.h
	src/quick.cpp
	src/savestate.cpp
)

target_include_directories(savestate PUBLIC include)
target_link_libraries(savestate PUBLIC emuenv mem)
target_link_libraries(savestate PRIVATE app audio codec config cpu gxm io kernel miniz ngs nids renderer rtc util)
