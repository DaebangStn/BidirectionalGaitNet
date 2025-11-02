#!/bin/bash
# Refactor GLFWApp to use single Motion* instead of vector

FILE="viewer/GLFWApp.cpp"

# Backup original
cp "$FILE" "$FILE.backup"

# Replace mMotions[mMotionIdx] with mMotion
sed -i 's/mMotions\[mMotionIdx\]/mMotion/g' "$FILE"

# Replace mMotionStates[mMotionIdx] with mMotionState
sed -i 's/mMotionStates\[mMotionIdx\]/mMotionState/g' "$FILE"

# Replace mMotionStates[i] with mMotionState (for loop contexts)
sed -i 's/mMotionStates\[i\]/mMotionState/g' "$FILE"

# Replace !mMotions.empty() checks with mMotion != nullptr
sed -i 's/!mMotions\.empty()/mMotion != nullptr/g' "$FILE"

# Replace mMotions.empty() checks with mMotion == nullptr
sed -i 's/mMotions\.empty()/mMotion == nullptr/g' "$FILE"

# Replace mMotionStates.empty() with mMotion == nullptr
sed -i 's/!mMotionStates\.empty()/mMotion != nullptr/g' "$FILE"
sed -i 's/mMotionStates\.empty()/mMotion == nullptr/g' "$FILE"

# Replace mMotions.size() > 0 with mMotion != nullptr
sed -i 's/mMotions\.size() > 0/mMotion != nullptr/g' "$FILE"

echo "Refactoring complete. Original backed up to $FILE.backup"
echo "Manual review required for:"
echo "  - Motion selection UI removal"
echo "  - mMotionIdx boundary checks"
echo "  - Loop iterations over mMotions/mMotionStates"
