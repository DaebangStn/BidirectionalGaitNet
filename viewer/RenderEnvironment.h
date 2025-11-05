#ifndef RENDER_ENVIRONMENT_H
#define RENDER_ENVIRONMENT_H

#include "Environment.h"
#include "CBufferData.h"
#include <map>
#include <string>

class RenderEnvironment : public Environment {
public:
    RenderEnvironment(std::string metadata, CBufferData<double>* graph_data);
    ~RenderEnvironment();

    // Override step to add automatic graph data recording
    void step() override;

    // Direct access to underlying environment (for legacy/compatibility)
    Environment* GetEnvironment() { return this; }

private:
    void RecordGraphData();
    void RecordInfoData();

    CBufferData<double>* mGraphData;
};

#endif // RENDER_ENVIRONMENT_H
