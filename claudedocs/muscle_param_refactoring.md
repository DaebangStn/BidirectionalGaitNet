# Muscle Parameter Refactoring

## Overview
Refactored muscle parameter modification code to improve encapsulation and follow the Law of Demeter. Environment class no longer directly accesses and modifies Muscle objects, instead using a dedicated Character method.

## Changes Made

### Before (Violation of Encapsulation)
```cpp
// Environment.cpp - Direct access to Muscle internals
for (auto m : mCharacters[0]->getMuscles())
    if (name.substr(14) == m->GetName())
    {
        m->change_l(_param_state[idx]);
        break;
    }
```

### After (Proper Encapsulation)
```cpp
// Environment.cpp - Uses Character's public interface with loop-based calls
idx = 0;
for (auto name : mParamName)
{
    if (name.find("muscle_length") != std::string::npos)
    {
        mCharacters[0]->setMuscleParam(name.substr(14), "length", _param_state[idx]);
    }
    else if (name.find("muscle_force") != std::string::npos)
    {
        mCharacters[0]->setMuscleParam(name.substr(13), "force", _param_state[idx]);
    }
    idx++;
}
```

## Implementation Details

### New Method in Character Class

**Header (Character.h)**:
```cpp
void setMuscleParam(const std::string& muscleName, const std::string& paramType, double value);
```

**Implementation (Character.cpp)**:
```cpp
void Character::setMuscleParam(const std::string& muscleName, const std::string& paramType, double value)
{
    for (auto m : mMuscles)
    {
        if (m->GetName() == muscleName)
        {
            if (paramType == "length")
                m->change_l(value);
            else if (paramType == "force")
                m->change_f(value);
            return;
        }
    }
}
```

### Modified Methods in Environment

**setParamState()** (lines 1448-1460):
- Loops through parameter names
- Calls `setMuscleParam()` individually for each muscle parameter

**setNormalizedParamState()** (lines 1491-1505):
- Same loop-based pattern with normalized parameter calculation
- Maintains consistency with setParamState()

## Benefits

### 1. **Improved Encapsulation**
- Environment no longer accesses Muscle objects directly
- Character manages its own internal muscle collection
- Follows Law of Demeter (don't talk to strangers)

### 2. **Better Maintainability**
- Muscle modification logic centralized in Character
- Easier to add validation, logging, or side effects
- Clear separation of concerns

### 3. **Reduced Coupling**
- Environment depends only on Character's public interface
- Changes to Muscle internal structure won't affect Environment
- More flexible for future refactoring

### 4. **Single Responsibility**
- Character is responsible for muscle management
- Environment focuses on simulation parameter orchestration
- Each class has a clear, focused purpose

### 5. **Simple and Direct**
- Clear one-parameter-per-call pattern
- Easy to understand and maintain
- Flexible for individual parameter updates
- No collection overhead

## Performance Considerations

**Time Complexity**:
- Before: O(n × m) where n = number of params, m = number of muscles
- After: O(n × m) - same complexity but with better encapsulation
- Each setMuscleParam() call is O(m) for muscle lookup

**Design Trade-off**:
- Prioritizes encapsulation and simplicity over micro-optimization
- Performance impact negligible for typical muscle counts (< 100)
- Clear, maintainable code is more valuable than small performance gains

## Usage Example

```cpp
// Setting individual muscle parameters
character->setMuscleParam("MuscleA", "length", 1.05);
character->setMuscleParam("MuscleA", "force", 1.10);
character->setMuscleParam("MuscleB", "length", 0.95);

// Environment.cpp pattern - loop-based calls
for (auto name : mParamName)
{
    if (name.find("muscle_length") != std::string::npos)
        mCharacters[0]->setMuscleParam(name.substr(14), "length", param_value);
    else if (name.find("muscle_force") != std::string::npos)
        mCharacters[0]->setMuscleParam(name.substr(13), "force", param_value);
}
```

## Build Verification
✅ Build successful - all 27 targets compiled without errors
✅ No API changes to external interfaces
✅ Backward compatible refactoring

## Files Modified
- `sim/Character.h` - Added setMuscleParam() declaration (line 115)
- `sim/Character.cpp` - Implemented setMuscleParam() method (lines 605-618)
- `sim/Environment.cpp` - Refactored setParamState() (lines 1448-1460) and setNormalizedParamState() (lines 1491-1505)

## Design Notes

The final implementation uses a **loop-based individual call pattern** rather than bulk operations. This design choice prioritizes:

1. **Simplicity**: Each parameter modification is a single, clear method call
2. **Flexibility**: Easy to add logging, validation, or side effects per parameter
3. **Maintainability**: Straightforward to understand and modify
4. **Encapsulation**: Environment doesn't need to know about Muscle internals

The method signature `setMuscleParam(muscleName, paramType, value)` uses string-based `paramType` ("length" or "force") to keep the interface simple and extensible for future parameter types.
