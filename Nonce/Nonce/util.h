
#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

struct timeb start;

void description(char* name, char* desc);

void progress(char* name, int r, double time);

void report(char* name, double* times);

#ifdef __cplusplus
}
#endif

#endif
