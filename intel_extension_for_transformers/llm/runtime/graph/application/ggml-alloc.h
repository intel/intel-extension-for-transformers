#pragma once

#include "core/ne.h"
#include "core/ne_layers.h"

#ifdef  __cplusplus
extern "C" {
#endif
#define NE_GRAPH_HASHTABLE_SIZE 8273
#define NE_MAX_NODES         4096

struct ne_allocr * ne_allocr_new(void * data, size_t size, size_t alignment);
struct ne_allocr * ne_allocr_new_measure(size_t alignment);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
void   ne_allocr_set_parse_seq(struct ne_allocr * alloc, const int * list, int n);

void   ne_allocr_free(struct ne_allocr * alloc);
bool   ne_allocr_is_measure(struct ne_allocr * alloc);
void   ne_allocr_reset(struct ne_allocr * alloc);
void   ne_allocr_alloc(struct ne_allocr * alloc, struct ne_tensor * tensor);
size_t ne_allocr_alloc_graph(struct ne_allocr * alloc, struct ne_cgraph * graph);


#ifdef  __cplusplus
}
#endif
