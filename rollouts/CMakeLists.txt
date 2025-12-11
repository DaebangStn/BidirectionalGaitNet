# Rollout modules
#
# Structure:
#   common/  - Shared components (RolloutRecord, RecordConfig)
#   ray/     - Ray-based distributed rollout (pyrollout.so)
#   sample/  - Single-core sequential rollout (pysamplerollout.so)

# Add sim directory to include path (for all subdirectories)
include_directories(${CMAKE_SOURCE_DIR}/sim)
link_directories(${CMAKE_SOURCE_DIR}/sim)

# Build order matters: common first, then ray/sample
add_subdirectory(common)
add_subdirectory(ray)
add_subdirectory(sample)
