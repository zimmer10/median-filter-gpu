#pragma once
#include "EasyBMP/EasyBMP.h"
#include <cstdint>

// GRAYSCALE

void create_BMP_grayscale(BMP& inputBMP, BMP& outputBMP, uint8_t* outputPixels) {
    const int w = inputBMP.TellWidth();
    const int h = inputBMP.TellHeight();
    outputBMP.SetSize(w, h);
    outputBMP.SetBitDepth(inputBMP.TellBitDepth());

    if (inputBMP.TellBitDepth() <= 8) {
        int numColors = inputBMP.TellNumberOfColors();
        for (int i = 0; i < numColors; i++) {
            RGBApixel color = inputBMP.GetColor(i);
            outputBMP.SetColor(i, color);
        }
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint8_t color = outputPixels[y * w + x];
                RGBApixel pixel = outputBMP.GetColor(color);
                outputBMP.SetPixel(x, y, pixel);
            }
        }
    }
}

// RGB

// загружает R, G, B каналы из BMP в три отдельных планарных буфера (каждый w*h байт).
void load_rgb_from_bmp(BMP& bmp, uint8_t* r, uint8_t* g, uint8_t* b) {
    const int w = bmp.TellWidth();
    const int h = bmp.TellHeight();
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            RGBApixel pixel = bmp.GetPixel(x, y);
            r[y * w + x] = static_cast<uint8_t>(pixel.Red);
            g[y * w + x] = static_cast<uint8_t>(pixel.Green);
            b[y * w + x] = static_cast<uint8_t>(pixel.Blue);
        }
    }
}

// создаёт 24-bit RGB BMP из трёх планарных буферов R, G, B (каждый w*h байт).
void create_BMP_rgb(BMP& outputBMP, int w, int h,
                    const uint8_t* r, const uint8_t* g, const uint8_t* b) {
    outputBMP.SetSize(w, h);
    outputBMP.SetBitDepth(24);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            RGBApixel pixel;
            pixel.Red   = r[y * w + x];
            pixel.Green = g[y * w + x];
            pixel.Blue  = b[y * w + x];
            pixel.Alpha = 0;
            outputBMP.SetPixel(x, y, pixel);
        }
    }
}
