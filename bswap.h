#ifndef BSWAP_H
#define BSWAP_H

#include <stdint.h>

/**
 * @brief Convert a 32-bit unsigned integer from big-endian to little-endian or
 * vice versa.
 *
 * @param x The 32-bit unsigned integer to byte-swap.
 * @return The byte-swapped 32-bit unsigned integer.
 */
uint32_t bswap32(uint32_t x);

#endif
