#ifndef RENDER_ENVIRONMENT_H
#define RENDER_ENVIRONMENT_H

#include "Environment.h"
#include "CBufferData.h"
#include <map>
#include <string>

class RenderEnvironment {
public:
    RenderEnvironment(std::string metadata, CBufferData<double>* graph_data);
    ~RenderEnvironment();

    // Step with automatic graph data recording
    void step();

    // Direct access to underlying environment (for other GLFWApp operations)
    Environment* GetEnvironment() { return mEnv; }

    // Reward map for visualization (delegates to Environment)
    const std::map<std::string, double>& GetRewardMap() { return mEnv->getRewardMap(); }

private:
    void RecordGraphData();
    void RecordRewardData();

    Environment* mEnv;
    CBufferData<double>* mGraphData;
};

#endif // RENDER_ENVIRONMENT_H

