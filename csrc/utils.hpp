#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
#endif  /* __cplusplus */
  void onCallback(int);
  void startProfiling(char*);
  double* endProfiling(uint64_t*);
#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif /* __UTILS_HPP__ */

