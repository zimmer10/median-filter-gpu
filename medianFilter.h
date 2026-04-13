#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include "utils.h"

class MedianFilter {
private:
    static uint8_t median_9(uint8_t window[9]);
public:
    // Grayscale (1 канал)
    static void median_filter_3x3(const uint8_t* input, uint8_t* output,
                                   size_t width, size_t height, size_t stride);

    // RGB (3 канала) — однопоточная версия
    static void median_filter_3x3_rgb(
        const uint8_t* inputR,  const uint8_t* inputG,  const uint8_t* inputB,
              uint8_t* outputR,       uint8_t* outputG,       uint8_t* outputB,
        size_t width, size_t height, size_t stride);
};


uint8_t MedianFilter::median_9(uint8_t window[9]) {
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

// Grayscale: width — ширина в пикселях, stride — шаг строки 
void MedianFilter::median_filter_3x3(const uint8_t* input, uint8_t* output,
                                      size_t width, size_t height, size_t stride) {
    for (size_t y = 0; y < height; ++y) {
        const uint8_t* y0 = input + (y > 0          ? y - 1 : 0) * stride;
        const uint8_t* y1 = input + y * stride;
        const uint8_t* y2 = input + (y < height - 1 ? y + 1 : y) * stride;

        for (size_t x = 0; x < width; ++x) {
            size_t x0 = (x > 0         ? x - 1 : 0);
            size_t x1 = x;
            size_t x2 = (x < width - 1 ? x + 1 : x);

            uint8_t window[9] = {
                y0[x0], y0[x1], y0[x2],
                y1[x0], y1[x1], y1[x2],
                y2[x0], y2[x1], y2[x2]
            };
            output[y * stride + x] = median_9(window);
        }
    }
}

// RGB: применяем тот же фильтр к каждому каналу независимо
void MedianFilter::median_filter_3x3_rgb(
    const uint8_t* inputR,  const uint8_t* inputG,  const uint8_t* inputB,
          uint8_t* outputR,       uint8_t* outputG,       uint8_t* outputB,
    size_t width, size_t height, size_t stride)
{
    median_filter_3x3(inputR, outputR, width, height, stride);
    median_filter_3x3(inputG, outputG, width, height, stride);
    median_filter_3x3(inputB, outputB, width, height, stride);
}
