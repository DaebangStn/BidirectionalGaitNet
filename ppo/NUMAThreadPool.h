#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#include <pthread.h>
#endif

/**
 * NUMA-aware thread pool with CPU affinity
 *
 * Distributes threads across NUMA nodes and pins each thread to CPUs on its assigned node.
 * This eliminates cross-NUMA memory access overhead during parallel environment stepping.
 *
 * Thread assignment:
 * - Thread i → NUMA node: i / threads_per_node
 * - Each thread pinned to CPUs on its NUMA node via pthread_setaffinity_np()
 *
 * Fallback: If NUMA not available or disabled, behaves like standard ThreadPool without affinity.
 */
class NUMAThreadPool {
public:
    /**
     * Create NUMA-aware thread pool
     *
     * @param num_threads Total number of worker threads
     * @param enable_numa Enable NUMA awareness (thread affinity + topology detection)
     */
    NUMAThreadPool(int num_threads, bool enable_numa = false);

    /**
     * Destructor - stops all threads
     */
    ~NUMAThreadPool();

    /**
     * Enqueue a task for execution
     *
     * @param task Function to execute in worker thread
     */
    void enqueue(std::function<void()> task);

    /**
     * Wait for all enqueued tasks to complete
     */
    void wait();

    /**
     * Get NUMA node for given thread ID
     *
     * @param thread_id Thread index (0 to num_threads-1)
     * @return NUMA node ID, or 0 if NUMA disabled
     */
    int get_numa_node(int thread_id) const;

    /**
     * Check if NUMA is enabled
     */
    bool numa_enabled() const { return numa_enabled_; }

    /**
     * Get number of NUMA nodes
     */
    int num_numa_nodes() const { return num_nodes_; }

private:
    void worker_thread(int thread_id);
    void set_thread_affinity(int thread_id);

    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable wait_condition_;

    std::atomic<int> active_tasks_{0};
    std::atomic<int> pending_tasks_{0};
    bool stop_{false};

    // NUMA configuration
    bool numa_enabled_{false};
    int num_nodes_{1};
    int threads_per_node_{0};
    int num_threads_{0};
};
