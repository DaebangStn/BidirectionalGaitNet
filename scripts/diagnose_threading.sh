#!/bin/bash
# Threading Diagnostic Script for BatchRolloutEnv Performance Issues

echo "================================================================================"
echo "THREADING DIAGNOSTIC SCRIPT"
echo "================================================================================"
echo ""

# 1. Check environment variables at runtime
echo "=== Step 1: Environment Variables Check ==="
echo "Current shell environment:"
echo "  OMP_NUM_THREADS = ${OMP_NUM_THREADS:-<not set>}"
echo "  MKL_NUM_THREADS = ${MKL_NUM_THREADS:-<not set>}"
echo "  OPENBLAS_NUM_THREADS = ${OPENBLAS_NUM_THREADS:-<not set>}"
echo "  VECLIB_MAXIMUM_THREADS = ${VECLIB_MAXIMUM_THREADS:-<not set>}"
echo ""

# 2. Check CPU information
echo "=== Step 2: CPU Information ==="
echo "Physical cores: $(lscpu | grep "^Core(s) per socket" | awk '{print $4}')"
echo "Threads per core: $(lscpu | grep "^Thread(s) per core" | awk '{print $4}')"
echo "Total logical cores: $(nproc)"
echo ""

# 3. Create test script to verify threading at runtime
cat > /tmp/test_threading.py << 'EOF'
import os
import torch
import sys

print("=== Step 3: Python Runtime Threading Check ===")
print(f"OMP_NUM_THREADS (from os.environ): {os.environ.get('OMP_NUM_THREADS', '<not set>')}")
print(f"MKL_NUM_THREADS (from os.environ): {os.environ.get('MKL_NUM_THREADS', '<not set>')}")

# Check PyTorch threading
print(f"\nPyTorch threading:")
print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")

# Check if libtorch sees the settings
try:
    import subprocess
    result = subprocess.run(['ldd', sys.executable], capture_output=True, text=True)
    if 'libgomp' in result.stdout or 'libomp' in result.stdout:
        print(f"\nOpenMP library detected in Python executable")
except:
    pass

# Test actual thread count during computation
print(f"\n=== Step 4: Active Thread Count During Computation ===")
print("Starting tensor operations to trigger threading...")

import threading
import time

# Monitor thread count during computation
def count_threads():
    import subprocess
    pid = os.getpid()
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-L', '-o', 'nlwp'],
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            return int(lines[1].strip())
    except:
        pass
    return -1

print(f"Threads before computation: {count_threads()}")

# Trigger computation
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)
for i in range(10):
    z = torch.mm(x, y)
    if i == 0:
        print(f"Threads during computation: {count_threads()}")

print(f"Threads after computation: {count_threads()}")
EOF

python3 /tmp/test_threading.py

# 4. Check actual thread usage during benchmark
echo ""
echo "=== Step 5: Real Benchmark Thread Monitoring ==="
echo "We'll run a short benchmark while monitoring thread count..."
echo ""

# Create monitoring script
cat > /tmp/monitor_threads.sh << 'EOF'
#!/bin/bash
# Monitor thread count of a process
PID=$1
INTERVAL=${2:-0.5}

while kill -0 $PID 2>/dev/null; do
    THREAD_COUNT=$(ps -p $PID -L -o nlwp | tail -1 | tr -d ' ')
    CPU_USAGE=$(ps -p $PID -o %cpu | tail -1 | tr -d ' ')
    echo "$(date +%H:%M:%S) | Threads: $THREAD_COUNT | CPU: $CPU_USAGE%"
    sleep $INTERVAL
done
EOF
chmod +x /tmp/monitor_threads.sh

echo "To monitor during your actual benchmark run:"
echo "  1. Start your benchmark in one terminal"
echo "  2. Get its PID: pgrep -f 'ppo_rollout_learner'"
echo "  3. Run: /tmp/monitor_threads.sh <PID>"
echo ""

# 5. Check libtorch threading configuration
echo "=== Step 6: LibTorch Threading Check ==="
cat > /tmp/check_libtorch_threads.cpp << 'EOF'
#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch threading configuration:" << std::endl;
    std::cout << "  at::get_num_threads(): " << at::get_num_threads() << std::endl;
    std::cout << "  at::get_num_interop_threads(): " << at::get_num_interop_threads() << std::endl;

    // Test actual threading
    std::cout << "\nPerforming computation to check actual threading..." << std::endl;
    auto x = torch::randn({1000, 1000});
    auto y = torch::randn({1000, 1000});
    auto z = torch::mm(x, y);

    return 0;
}
EOF

echo "Created C++ threading test: /tmp/check_libtorch_threads.cpp"
echo "Compile with: g++ -o /tmp/test_threads /tmp/check_libtorch_threads.cpp \\"
echo "              -I/opt/miniconda3/envs/bidir/include \\"
echo "              -L/opt/miniconda3/envs/bidir/lib -ltorch -lc10"
echo ""

# 6. Suggested diagnostic commands
echo "=== Step 7: Manual Diagnostic Commands ==="
echo ""
echo "Run these commands to diagnose threading issues:"
echo ""
echo "1. Check environment variables during Python execution:"
echo "   python3 -c \"import os; print('OMP:', os.environ.get('OMP_NUM_THREADS')); print('MKL:', os.environ.get('MKL_NUM_THREADS'))\""
echo ""
echo "2. Monitor thread count during benchmark:"
echo "   python3 ppo/benchmark_num_envs.py --num-envs-list 2 --total-timesteps 128 &"
echo "   PID=\$!; while kill -0 \$PID 2>/dev/null; do ps -p \$PID -L -o nlwp | tail -1; sleep 0.5; done"
echo ""
echo "3. Check for thread contention with perf:"
echo "   perf stat -e context-switches,cpu-migrations python3 ppo/benchmark_num_envs.py --num-envs-list 2 4 8 --total-timesteps 256"
echo ""
echo "4. Profile with htop (install if needed):"
echo "   htop -p \$(pgrep -f ppo_rollout_learner)"
echo ""
echo "5. Check if MKL is actually being used:"
echo "   python3 -c \"import torch; print(torch.__config__.show())\""
echo ""

echo "================================================================================"
echo "DIAGNOSTIC SCRIPT COMPLETE"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Run the commands above to verify threading configuration"
echo "2. Monitor actual thread count during benchmark execution"
echo "3. Compare thread counts between num_envs=2 and num_envs=16"
echo "4. Check for context switching overhead with 'perf stat'"
echo ""
