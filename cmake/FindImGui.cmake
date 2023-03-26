#
# Find ImGui
#
# Try to find ImGui : OpenGL Mathematics.
# This module defines
# - IMGUI_INCLUDE_DIRS
# - IMGUI_FOUND
#
# The following variables can be set as arguments for the module.
# - IMGUI_ROOT_DIR : Root library directory of ImGui

# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
  # Find include files
  find_path(
    IMGUI_INCLUDE_DIR
    NAMES imgui/imgui.h
    PATHS
    $ENV{PROGRAMFILES}
    ${IMGUI_ROOT_DIR}
    DOC "The directory where imgui/imgui.h resides")
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(IMGUI DEFAULT_MSG IMGUI_INCLUDE_DIR)

# Define IMGUI_INCLUDE_DIRS
if (IMGUI_FOUND)
  set(IMGUI_INCLUDE_DIRS ${IMGUI_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(IMGUI_INCLUDE_DIR)
