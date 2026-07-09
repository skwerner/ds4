// Compare malloc_host vs malloc_device with model-range-copy pattern.
// Compile: icpx -O3 -fsycl -o submit_overhead submit_overhead.cpp
// Run:     ./submit_overhead [n_iter=50]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double now_ms() {
    return std::chrono::steady_clock::now().time_since_epoch().count() / 1000000.0;
}

int main(int argc, char **argv) {
    int n_iter = argc > 1 ? atoi(argv[1]) : 50;
    uint64_t sz = 4 * 1024 * 1024ull; // 4 MiB

    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{
                      sycl::property::queue::in_order{},
                      sycl::ext::intel::property::queue::immediate_command_list{}});

    fprintf(stderr, "Device: %s\n\n",
            q.get_device().get_info<sycl::info::device::name>().c_str());

    // Source: regular malloc'd buffer (simulates mmap'd model file)
    char *src = (char *)malloc(sz);
    memset(src, 1, sz);

    // Allocate host and device USM buffers
    char *host_buf = (char *)sycl::malloc_host(sz, q);
    char *dev_buf  = (char *)sycl::malloc_device(sz, q);
    memset(host_buf, 0, sz);
    q.memset(dev_buf, 0, sz).wait();

    // A: plain→malloc_host memcpy latency
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++)
            memcpy(host_buf, src, sz);
        double ms = now_ms() - t0;
        fprintf(stderr, "A — plain→malloc_host 4 MiB:  %.3f ms  = %.1f GB/s\n",
                ms / n_iter, (double)sz * n_iter / (ms * 1e6));
    }

    // B: q.memcpy host→device latency
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++)
            q.memcpy(dev_buf, host_buf, sz);
        q.wait();
        double ms = now_ms() - t0;
        fprintf(stderr, "B — q.memcpy host→device:     %.3f ms  = %.1f GB/s\n",
                ms / n_iter, (double)sz * n_iter / (ms * 1e6));
    }

    // C: plain→device with q.memcpy + kernel submit
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            q.memcpy(dev_buf, src, sz);
            q.parallel_for(7168, [=](auto i) {
                ((float *)dev_buf)[i % 7168] += 1.0f;
            });
        }
        q.wait();
        double ms = now_ms() - t0;
        fprintf(stderr, "C — q.memcpy + kernel (dev):   %.3f ms/iter\n", ms / n_iter);
    }

    // D: plain→host memcpy + kernel submit (current real-app pattern)
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            memcpy(host_buf, src, sz);
            q.parallel_for(7168, [=](auto i) {
                ((float *)host_buf)[i % 7168] += 1.0f;
            });
        }
        q.wait();
        double ms = now_ms() - t0;
        fprintf(stderr, "D — memcpy + kernel (host):    %.3f ms/iter\n", ms / n_iter);
    }

    // E: kernel only on device buf (pure submit overhead)
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++)
            q.parallel_for(7168, [=](auto i) {
                ((float *)dev_buf)[i % 7168] += 1.0f;
            });
        q.wait();
        double ms = now_ms() - t0;
        fprintf(stderr, "E — kernel only (dev):         %.3f ms/iter\n", ms / n_iter);
    }

    // F: kernel only on host buf (pure submit + potential migration overhead)
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++)
            q.parallel_for(7168, [=](auto i) {
                ((float *)host_buf)[i % 7168] += 1.0f;
            });
        q.wait();
        double ms = now_ms() - t0;
        fprintf(stderr, "F — kernel only (host):        %.3f ms/iter\n", ms / n_iter);
    }

    // G: memcpy host→device THEN kernel on device (explicit 2-step)
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            memcpy(host_buf, src, sz);
            q.memcpy(dev_buf, host_buf, sz);
            q.parallel_for(7168, [=](auto i) {
                ((float *)dev_buf)[i % 7168] += 1.0f;
            });
        }
        q.wait();
        double ms = now_ms() - t0;
        fprintf(stderr, "G — memcpy→q.memcpy→kernel:    %.3f ms/iter\n", ms / n_iter);
    }

    // H: memcpy host→device (async via q.memcpy), skip kernel
    {
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++)
            q.memcpy(dev_buf, host_buf, sz);
        q.wait();
        double ms = now_ms() - t0;
        fprintf(stderr, "H — q.memcpy host→device only: %.3f ms/iter\n", ms / n_iter);
    }

    sycl::free(host_buf, q);
    sycl::free(dev_buf, q);
    free(src);
    return 0;
}
