cmake_minimum_required(VERSION 3.10)
project(CellInspection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Boost (required for cubic spline interpolation)
find_package(Boost REQUIRED COMPONENTS math)

# Add executable
add_executable(CellInspection
    Cell_inspection.cpp
    GradientCalculator.cpp
    StructureTensorAnalysis.cpp
)

# Link libraries
target_link_libraries(CellInspection
    ${OpenCV_LIBS}
    Boost::math
)

# Install target (optional)
install(TARGETS CellInspection DESTINATION bin)