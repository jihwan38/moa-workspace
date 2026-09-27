#include <stddef.h>
#include <stdint.h>
uint64_t fnv64(const unsigned char *data, size_t size) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t i=0; i<size; ++i) hash=(hash ^ data[i])*UINT64_C(0x100000001b3);
    return hash;
}
