# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/dange/Documents/cse168/torrey-cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/dange/Documents/cse168/torrey-cuda/cmake-build-default-wsl

# Include any dependencies generated for this target.
include CMakeFiles/torrey.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/torrey.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/torrey.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torrey.dir/flags.make

CMakeFiles/torrey.dir/src/main.cu.o: CMakeFiles/torrey.dir/flags.make
CMakeFiles/torrey.dir/src/main.cu.o: ../src/main.cu
CMakeFiles/torrey.dir/src/main.cu.o: CMakeFiles/torrey.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/dange/Documents/cse168/torrey-cuda/cmake-build-default-wsl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/torrey.dir/src/main.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/torrey.dir/src/main.cu.o -MF CMakeFiles/torrey.dir/src/main.cu.o.d -x cu -c /mnt/c/Users/dange/Documents/cse168/torrey-cuda/src/main.cu -o CMakeFiles/torrey.dir/src/main.cu.o

CMakeFiles/torrey.dir/src/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/torrey.dir/src/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/torrey.dir/src/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/torrey.dir/src/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target torrey
torrey_OBJECTS = \
"CMakeFiles/torrey.dir/src/main.cu.o"

# External object files for target torrey
torrey_EXTERNAL_OBJECTS =

torrey: CMakeFiles/torrey.dir/src/main.cu.o
torrey: CMakeFiles/torrey.dir/build.make
torrey: libtorrey_lib.a
torrey: CMakeFiles/torrey.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/dange/Documents/cse168/torrey-cuda/cmake-build-default-wsl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable torrey"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torrey.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torrey.dir/build: torrey
.PHONY : CMakeFiles/torrey.dir/build

CMakeFiles/torrey.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torrey.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torrey.dir/clean

CMakeFiles/torrey.dir/depend:
	cd /mnt/c/Users/dange/Documents/cse168/torrey-cuda/cmake-build-default-wsl && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/dange/Documents/cse168/torrey-cuda /mnt/c/Users/dange/Documents/cse168/torrey-cuda /mnt/c/Users/dange/Documents/cse168/torrey-cuda/cmake-build-default-wsl /mnt/c/Users/dange/Documents/cse168/torrey-cuda/cmake-build-default-wsl /mnt/c/Users/dange/Documents/cse168/torrey-cuda/cmake-build-default-wsl/CMakeFiles/torrey.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torrey.dir/depend

