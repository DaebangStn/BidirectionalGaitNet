#ifndef RENDER_ENVIRONMENT_H
#define RENDER_ENVIRONMENT_H

#include "Environment.h"
#include "CBufferData.h"
#include "IncrementalLeastSquares.h"
#include <map>
#include <string>

class RenderEnvironment : public Environment {
public:
    RenderEnvironment(const std::string& filepath, CBufferData<double>* graph_data);
    ~RenderEnvironment();

    // Override step to add automatic graph data recording
    void step() override;

    // Direct access to underlying environment (for legacy/compatibility)
    Environment* GetEnvironment() { return this; }

    // Velocity method selection
    void setVelocityMethod(int method) { mVelocityMethod = method; }
    int getVelocityMethod() const { return mVelocityMethod; }

private:
    void RecordGraphData();
    void RecordInfoData();

    CBufferData<double>* mGraphData;

    // Incremental least squares estimators for COM velocity
    IncrementalLeastSquares mVelocityX_Estimator;
    IncrementalLeastSquares mVelocityZ_Estimator;

    // Incremental least squares estimator for X-Z regression (lateral deviation)
    IncrementalLeastSquares mXZ_Regression;

    // Velocity calculation method: 0 = Least Squares, 1 = Avg Horizon
    int mVelocityMethod;
};

#endif // RENDER_ENVIRONMENT_H
