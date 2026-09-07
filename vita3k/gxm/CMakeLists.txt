add_library(
	gxm
	STATIC
	include/gxm/functions.h
	include/gxm/state.h
	include/gxm/types.h
	src/attributes.cpp
	src/color.cpp
	src/gxp.cpp
	src/stream.cpp
	src/textures.cpp
	src/transfer.cpp
)

target_include_directories(gxm PUBLIC include)
# threads is PUBLIC: gxm/state.h is a public header and includes threads/queue.h, so every
# consumer of it needs that include path. It happened to work only transitively before.
target_link_libraries(gxm PUBLIC util threads)
target_link_libraries(gxm PRIVATE)
