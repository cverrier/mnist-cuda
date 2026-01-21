#include <assert.h>
#include <stdio.h>

#include "bswap.h"

void test_bswap32() {
  assert(bswap32(0x12345678) == 0x78563412);

  uint32_t original = 0xDEADBEEF;
  assert(bswap32(bswap32(original)) == original);

  assert(bswap32(0x00000000) == 0x00000000);

  assert(bswap32(0xFFFFFFFF) == 0xFFFFFFFF);

  assert(bswap32(0x000000FF) == 0xFF000000);
  assert(bswap32(0x0000FF00) == 0x00FF0000);
  assert(bswap32(0x00FF0000) == 0x0000FF00);
  assert(bswap32(0xFF000000) == 0x000000FF);

  printf("bswap32 tests passed\n");
}

int main() {
  test_bswap32();
  return 0;
}
