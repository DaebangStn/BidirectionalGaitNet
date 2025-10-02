#include "PhysicalExam.h"
#include <iostream>

int main() {
    // Example: paste your stdout output here
    std::string pastedData = R"(
Pelvis               | X: 0.0000   Y: 1.0809   Z: -0.0116
FemurR               | X: -0.0959  Y: 1.0698   Z: 0.2452
TibiaR               | X: -0.0928  Y: 1.0584   Z: 0.6675
TalusR               | X: -0.0926  Y: 1.0683   Z: 0.9290
FootPinkyR           | X: -0.1244  Y: 1.1735   Z: 0.9384
FootThumbR           | X: -0.0765  Y: 1.1863   Z: 0.9385
FemurL               | X: 0.0959   Y: 1.0698   Z: 0.2452
TibiaL               | X: 0.0928   Y: 1.0584   Z: 0.6675
TalusL               | X: 0.0926   Y: 1.0683   Z: 0.9290
FootPinkyL           | X: 0.1244   Y: 1.1735   Z: 0.9384
FootThumbL           | X: 0.0765   Y: 1.1863   Z: 0.9385
Spine                | X: 0.0000   Y: 1.0809   Z: -0.1511
Torso                | X: 0.0000   Y: 1.0809   Z: -0.3539
Neck                 | X: 0.0000   Y: 1.0809   Z: -0.5604
Head                 | X: 0.0000   Y: 1.0839   Z: -0.6834
ShoulderR            | X: -0.0981  Y: 1.0572   Z: -0.4951
ArmR                 | X: -0.3578  Y: 1.0790   Z: -0.4829
ForeArmR             | X: -0.6674  Y: 1.0866   Z: -0.5006
HandR                | X: -0.8813  Y: 1.1240   Z: -0.4947
ShoulderL            | X: 0.0981   Y: 1.0572   Z: -0.4951
ArmL                 | X: 0.3578   Y: 1.0790   Z: -0.4829
ForeArmL             | X: 0.6674   Y: 1.0866   Z: -0.5006
HandL                | X: 0.8813   Y: 1.1240   Z: -0.4947
)";

    PMuscle::PhysicalExam exam;
    exam.parseAndPrintPostureConfig(pastedData);

    return 0;
}
