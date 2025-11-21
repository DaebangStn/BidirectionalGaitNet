#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <stdexcept>

/**
 * Simple FIFO ThreadPool for parallel task execution.
 *
 * Usage:
 *   ThreadPool pool(8);  // 8 worker threads
 *   pool.enqueue([]{  task_1(); });
 *   pool.enqueue([]{  task_2(); });
 *   pool.wait();  // Wait for all tasks to complete
 */
class ThreadPool {
public:
    explicit ThreadPool(int num_threads = std::thread::hardware_concurrency())
        : stop_(false), active_tasks_(0) {

        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                        ++active_tasks_;
                    }

                    task();

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        --active_tasks_;
                        wait_condition_.notify_all();
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.emplace(std::forward<F>(f));
        }
        condition_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        wait_condition_.wait(lock, [this] {
            return tasks_.empty() && active_tasks_ == 0;
        });
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable wait_condition_;
    int active_tasks_;
    bool stop_;
};
