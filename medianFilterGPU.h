#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "utils.h"

class MedianFilterGPU {
private:
    static uint8_t median_9(uint8_t window[9]);
public:
    static void median_filter_3x3_v1(const uint8_t* input, uint8_t* output,
                                      size_t width, size_t height, size_t stride,
                                      sycl::queue& q);
    static void median_filter_3x3_v2(const uint8_t* input, uint8_t* output,
                                      size_t width, size_t height, size_t stride,
                                      sycl::queue& q);

    // RGB — наивная версия (только глобальная память)
    static void median_filter_3x3_rgb_v1(
        const uint8_t* inputR,  const uint8_t* inputG,  const uint8_t* inputB,
              uint8_t* outputR,       uint8_t* outputG,       uint8_t* outputB,
        size_t width, size_t height, size_t stride, sycl::queue& q);

    // RGB — версия с shared memory и рабочими группами
    static void median_filter_3x3_rgb_v2(
        const uint8_t* inputR,  const uint8_t* inputG,  const uint8_t* inputB,
              uint8_t* outputR,       uint8_t* outputG,       uint8_t* outputB,
        size_t width, size_t height, size_t stride, sycl::queue& q);
};


uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);
    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);
    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);
    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);
    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);
    window[4] = get_max(window[3], window[4]);
    return window[4];
}


void MedianFilterGPU::median_filter_3x3_v1(const uint8_t* input, uint8_t* output,
                                             size_t width, size_t height, size_t stride,
                                             sycl::queue& q) {
    uint8_t* d_input  = sycl::malloc_shared<uint8_t>(height * stride, q);
    uint8_t* d_output = sycl::malloc_shared<uint8_t>(height * stride, q);
    q.memcpy(d_input, input, height * stride).wait();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
            size_t y = idx[0], x = idx[1];
            size_t y0 = (y > 0)          ? y - 1 : 0;
            size_t y2 = (y < height - 1) ? y + 1 : y;
            size_t x0 = (x > 0)          ? x - 1 : 0;
            size_t x2 = (x < width  - 1) ? x + 1 : x;
            uint8_t window[9] = {
                d_input[y0*stride+x0], d_input[y0*stride+x], d_input[y0*stride+x2],
                d_input[y *stride+x0], d_input[y *stride+x], d_input[y *stride+x2],
                d_input[y2*stride+x0], d_input[y2*stride+x], d_input[y2*stride+x2]
            };
            d_output[y * stride + x] = median_9(window);
        });
    });
    q.wait();

    q.memcpy(output, d_output, height * stride).wait();
    sycl::free(d_input,  q);
    sycl::free(d_output, q);
}


void MedianFilterGPU::median_filter_3x3_v2(const uint8_t* input, uint8_t* output,
                                             size_t width, size_t height, size_t stride,
                                             sycl::queue& q) {
    uint8_t* d_input  = sycl::malloc_shared<uint8_t>(height * stride, q);
    uint8_t* d_output = sycl::malloc_shared<uint8_t>(height * stride, q);
    q.memcpy(d_input, input, height * stride).wait();

    constexpr size_t BLOCK_SIZE  = 16;
    constexpr size_t SHARED_SIZE = BLOCK_SIZE + 2;
    size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t blocks_x = (width  + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> shared(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(blocks_y * BLOCK_SIZE, blocks_x * BLOCK_SIZE),
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
            ),
            [=](sycl::nd_item<2> item) {
                size_t global_y  = item.get_global_id(0);
                size_t global_x  = item.get_global_id(1);
                size_t local_y   = item.get_local_id(0);
                size_t local_x   = item.get_local_id(1);
                int block_start_y = item.get_group(0) * BLOCK_SIZE;
                int block_start_x = item.get_group(1) * BLOCK_SIZE;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int src_y = std::clamp(block_start_y + (int)local_y + dy, 0, (int)height - 1);
                        int src_x = std::clamp(block_start_x + (int)local_x + dx, 0, (int)width  - 1);
                        shared[local_y + 1 + dy][local_x + 1 + dx] = d_input[src_y * stride + src_x];
                    }
                }
                item.barrier();

                if (global_y < height && global_x < width) {
                    uint8_t window[9] = {
                        shared[local_y][local_x],     shared[local_y][local_x+1],     shared[local_y][local_x+2],
                        shared[local_y+1][local_x],   shared[local_y+1][local_x+1],   shared[local_y+1][local_x+2],
                        shared[local_y+2][local_x],   shared[local_y+2][local_x+1],   shared[local_y+2][local_x+2]
                    };
                    d_output[global_y * stride + global_x] = median_9(window);
                }
            }
        );
    });
    q.wait();

    q.memcpy(output, d_output, height * stride).wait();
    sycl::free(d_input,  q);
    sycl::free(d_output, q);
}

// RGB V1: НАИВНАЯ (глобальная память) 
// запускаем один kernel на все 3 канала сразу — каждый work-item обрабатывает
// один пиксель (y, x) и читает окно 3x3 для R, G, B из глобальной памяти.

void MedianFilterGPU::median_filter_3x3_rgb_v1(
    const uint8_t* inputR,  const uint8_t* inputG,  const uint8_t* inputB,
          uint8_t* outputR,       uint8_t* outputG,       uint8_t* outputB,
    size_t width, size_t height, size_t stride, sycl::queue& q)
{
    size_t sz = height * stride;

    uint8_t* dR_in  = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dG_in  = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dB_in  = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dR_out = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dG_out = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dB_out = sycl::malloc_shared<uint8_t>(sz, q);

    q.memcpy(dR_in, inputR, sz);
    q.memcpy(dG_in, inputG, sz);
    q.memcpy(dB_in, inputB, sz);
    q.wait();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
            size_t y = idx[0], x = idx[1];
            size_t y0 = (y > 0)          ? y - 1 : 0;
            size_t y2 = (y < height - 1) ? y + 1 : y;
            size_t x0 = (x > 0)          ? x - 1 : 0;
            size_t x2 = (x < width  - 1) ? x + 1 : x;

            // R
            uint8_t wr[9] = {
                dR_in[y0*stride+x0], dR_in[y0*stride+x], dR_in[y0*stride+x2],
                dR_in[y *stride+x0], dR_in[y *stride+x], dR_in[y *stride+x2],
                dR_in[y2*stride+x0], dR_in[y2*stride+x], dR_in[y2*stride+x2]
            };
            // G
            uint8_t wg[9] = {
                dG_in[y0*stride+x0], dG_in[y0*stride+x], dG_in[y0*stride+x2],
                dG_in[y *stride+x0], dG_in[y *stride+x], dG_in[y *stride+x2],
                dG_in[y2*stride+x0], dG_in[y2*stride+x], dG_in[y2*stride+x2]
            };
            // B
            uint8_t wb[9] = {
                dB_in[y0*stride+x0], dB_in[y0*stride+x], dB_in[y0*stride+x2],
                dB_in[y *stride+x0], dB_in[y *stride+x], dB_in[y *stride+x2],
                dB_in[y2*stride+x0], dB_in[y2*stride+x], dB_in[y2*stride+x2]
            };

            auto med9 = [](uint8_t w[9]) -> uint8_t {
                auto cs = [](uint8_t& a, uint8_t& b) { if (a > b) { uint8_t t = a; a = b; b = t; } };
                auto mn = [](uint8_t a, uint8_t b) -> uint8_t { return a < b ? a : b; };
                auto mx = [](uint8_t a, uint8_t b) -> uint8_t { return a > b ? a : b; };
                cs(w[0],w[3]); cs(w[1],w[7]); cs(w[2],w[5]); cs(w[4],w[8]);
                cs(w[0],w[7]); cs(w[2],w[4]); cs(w[3],w[8]); cs(w[5],w[6]);
                w[2]=mx(w[0],w[2]); cs(w[1],w[3]); cs(w[4],w[5]);
                w[7]=mn(w[7],w[8]); w[4]=mx(w[1],w[4]); w[3]=mn(w[3],w[6]);
                w[5]=mn(w[5],w[7]); cs(w[2],w[4]); cs(w[3],w[5]);
                w[3]=mx(w[2],w[3]); w[4]=mn(w[4],w[5]); w[4]=mx(w[3],w[4]);
                return w[4];
            };

            dR_out[y*stride+x] = med9(wr);
            dG_out[y*stride+x] = med9(wg);
            dB_out[y*stride+x] = med9(wb);
        });
    });
    q.wait();

    q.memcpy(outputR, dR_out, sz);
    q.memcpy(outputG, dG_out, sz);
    q.memcpy(outputB, dB_out, sz);
    q.wait();

    sycl::free(dR_in,  q); sycl::free(dG_in,  q); sycl::free(dB_in,  q);
    sycl::free(dR_out, q); sycl::free(dG_out, q); sycl::free(dB_out, q);
}

// RGB V2: SHARED MEMORY 
// каждая рабочая группа BLOCK_SIZE x BLOCK_SIZE загружает тайл (BLOCK_SIZE+2)^2
// пикселей в shared memory — отдельно для каждого канала (R, G, B).
// затем каждый work-item читает окно 3x3 из shared memory и вычисляет медиану.

void MedianFilterGPU::median_filter_3x3_rgb_v2(
    const uint8_t* inputR,  const uint8_t* inputG,  const uint8_t* inputB,
          uint8_t* outputR,       uint8_t* outputG,       uint8_t* outputB,
    size_t width, size_t height, size_t stride, sycl::queue& q)
{
    size_t sz = height * stride;

    uint8_t* dR_in  = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dG_in  = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dB_in  = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dR_out = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dG_out = sycl::malloc_shared<uint8_t>(sz, q);
    uint8_t* dB_out = sycl::malloc_shared<uint8_t>(sz, q);

    q.memcpy(dR_in, inputR, sz);
    q.memcpy(dG_in, inputG, sz);
    q.memcpy(dB_in, inputB, sz);
    q.wait();

    constexpr size_t BLOCK_SIZE  = 16;
    constexpr size_t SHARED_SIZE = BLOCK_SIZE + 2;  // +1 с каждой стороны
    size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t blocks_x = (width  + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> sharedR(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);
        sycl::local_accessor<uint8_t, 2> sharedG(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);
        sycl::local_accessor<uint8_t, 2> sharedB(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(blocks_y * BLOCK_SIZE, blocks_x * BLOCK_SIZE),
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
            ),
            [=](sycl::nd_item<2> item) {
                size_t global_y   = item.get_global_id(0);
                size_t global_x   = item.get_global_id(1);
                size_t local_y    = item.get_local_id(0);
                size_t local_x    = item.get_local_id(1);
                int block_start_y = item.get_group(0) * BLOCK_SIZE;
                int block_start_x = item.get_group(1) * BLOCK_SIZE;

                // загружаем окрестность 3x3 для каждого work-item в shared memory.
                // каждый work-item загружает 9 позиций (с перекрытием — допустимо,
                // т.к. все потоки читают из глобальной памяти одинаковые значения).
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int src_y = std::clamp(block_start_y + (int)local_y + dy, 0, (int)height - 1);
                        int src_x = std::clamp(block_start_x + (int)local_x + dx, 0, (int)width  - 1);
                        int dst_y = (int)local_y + 1 + dy;
                        int dst_x = (int)local_x + 1 + dx;
                        sharedR[dst_y][dst_x] = dR_in[src_y * stride + src_x];
                        sharedG[dst_y][dst_x] = dG_in[src_y * stride + src_x];
                        sharedB[dst_y][dst_x] = dB_in[src_y * stride + src_x];
                    }
                }

                //  все потоки группы завершили загрузку в shared memory
                item.barrier();

                if (global_y < height && global_x < width) {
                    size_t ly = local_y, lx = local_x;

                    uint8_t wr[9] = {
                        sharedR[ly][lx],   sharedR[ly][lx+1],   sharedR[ly][lx+2],
                        sharedR[ly+1][lx], sharedR[ly+1][lx+1], sharedR[ly+1][lx+2],
                        sharedR[ly+2][lx], sharedR[ly+2][lx+1], sharedR[ly+2][lx+2]
                    };
                    uint8_t wg[9] = {
                        sharedG[ly][lx],   sharedG[ly][lx+1],   sharedG[ly][lx+2],
                        sharedG[ly+1][lx], sharedG[ly+1][lx+1], sharedG[ly+1][lx+2],
                        sharedG[ly+2][lx], sharedG[ly+2][lx+1], sharedG[ly+2][lx+2]
                    };
                    uint8_t wb[9] = {
                        sharedB[ly][lx],   sharedB[ly][lx+1],   sharedB[ly][lx+2],
                        sharedB[ly+1][lx], sharedB[ly+1][lx+1], sharedB[ly+1][lx+2],
                        sharedB[ly+2][lx], sharedB[ly+2][lx+1], sharedB[ly+2][lx+2]
                    };

                    auto med9 = [](uint8_t w[9]) -> uint8_t {
                        auto cs = [](uint8_t& a, uint8_t& b) { if (a > b) { uint8_t t = a; a = b; b = t; } };
                        auto mn = [](uint8_t a, uint8_t b) -> uint8_t { return a < b ? a : b; };
                        auto mx = [](uint8_t a, uint8_t b) -> uint8_t { return a > b ? a : b; };
                        cs(w[0],w[3]); cs(w[1],w[7]); cs(w[2],w[5]); cs(w[4],w[8]);
                        cs(w[0],w[7]); cs(w[2],w[4]); cs(w[3],w[8]); cs(w[5],w[6]);
                        w[2]=mx(w[0],w[2]); cs(w[1],w[3]); cs(w[4],w[5]);
                        w[7]=mn(w[7],w[8]); w[4]=mx(w[1],w[4]); w[3]=mn(w[3],w[6]);
                        w[5]=mn(w[5],w[7]); cs(w[2],w[4]); cs(w[3],w[5]);
                        w[3]=mx(w[2],w[3]); w[4]=mn(w[4],w[5]); w[4]=mx(w[3],w[4]);
                        return w[4];
                    };

                    dR_out[global_y*stride+global_x] = med9(wr);
                    dG_out[global_y*stride+global_x] = med9(wg);
                    dB_out[global_y*stride+global_x] = med9(wb);
                }
            }
        );
    });
    q.wait();

    q.memcpy(outputR, dR_out, sz);
    q.memcpy(outputG, dG_out, sz);
    q.memcpy(outputB, dB_out, sz);
    q.wait();

    sycl::free(dR_in,  q); sycl::free(dG_in,  q); sycl::free(dB_in,  q);
    sycl::free(dR_out, q); sycl::free(dG_out, q); sycl::free(dB_out, q);
}
