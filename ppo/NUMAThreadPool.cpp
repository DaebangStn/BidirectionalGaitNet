#include "NUMAThreadPool.h"
#include <stdexcept>
#include <iostream>

NUMAThreadPool::NUMAThreadPool(int num_threads, bool enable_numa)
    : num_threads_(num_threads), numa_enabled_(enable_numa && num_threads > 1)
{
    if (num_threads <= 0) {
        throw std::invalid_argument("Number of threads must be positive");
    }

#ifdef __linux__
    // Detect NUMA topology if enabled
    if (numa_enabled_ && numa_available() != -1) {
        num_nodes_ = numa_max_node() + 1;
        threads_per_node_ = num_threads / num_nodes_;

        if (threads_per_node_ == 0) {
            // Too few threads for multi-node distribution
            numa_enabled_ = false;
            num_nodes_ = 1;
            threads_per_node_ = num_threads;
            std::cerr << "Warning: Too few threads for NUMA distribution. Disabling NUMA." << std::endl;
        } else {
            std::cout << "NUMA enabled: " << num_nodes_ << " nodes, "
                      << threads_per_node_ << " threads per node" << std::endl;
        }
    } else {
        numa_enabled_ = false;
        num_nodes_ = 1;
        threads_per_node_ = num_threads;
    }
#else
    // Non-Linux: NUMA not supported
    numa_enabled_ = false;
    num_nodes_ = 1;
    threads_per_node_ = num_threads;
#endif

    // Create worker threads
    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&NUMAThreadPool::worker_thread, this, i);
    }
}

NUMAThreadPool::~NUMAThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();

    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void NUMAThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(std::move(task));
        pending_tasks_++;
    }
    condition_.notify_one();
}

void NUMAThreadPool::wait() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    wait_condition_.wait(lock, [this] {
        return pending_tasks_ == 0 && active_tasks_ == 0;
    });
}

int NUMAThreadPool::get_numa_node(int thread_id) const {
    if (!numa_enabled_ || thread_id < 0 || thread_id >= num_threads_) {
        return 0;
    }
    return thread_id / threads_per_node_;
}

void NUMAThreadPool::set_thread_affinity(int thread_id) {
#ifdef __linux__
    if (!numa_enabled_) {
        return;
    }

    int node_id = get_numa_node(thread_id);

    // Set CPU affinity to this NUMA node
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    struct bitmask *cpus = numa_allocate_cpumask();
    if (numa_node_to_cpus(node_id, cpus) < 0) {
        std::cerr << "Warning: Failed to get CPUs for NUMA node " << node_id << std::endl;
        numa_free_cpumask(cpus);
        return;
    }

    // Add all CPUs from this NUMA node to the affinity mask
    int num_cpus = numa_num_configured_cpus();
    for (int cpu = 0; cpu < num_cpus; ++cpu) {
        if (numa_bitmask_isbitset(cpus, cpu)) {
            CPU_SET(cpu, &cpuset);
        }
    }
    numa_free_cpumask(cpus);

    // Set thread affinity
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
        std::cerr << "Warning: Failed to set CPU affinity for thread " << thread_id
                  << " to NUMA node " << node_id << std::endl;
    }
#endif
}

void NUMAThreadPool::worker_thread(int thread_id) {
    // Set CPU affinity for this thread
    set_thread_affinity(thread_id);

    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });

            if (stop_ && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            pending_tasks_--;
            active_tasks_++;
        }

        // Execute task (without holding lock)
        task();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            active_tasks_--;
            if (pending_tasks_ == 0 && active_tasks_ == 0) {
                wait_condition_.notify_all();
            }
        }
    }
}
