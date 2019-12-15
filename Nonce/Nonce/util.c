#include "util.h"


void description(char* name, char* desc) {
    sprintf(desc, "input=%s pattern=%s iterations=%d threads_per_block=%d", INPUT_STRING, PATTERN, MAX_ITERATION, THREADS_PER_BLOCK);
}

void progress(char* name, int r, double time) {
    char desc[100];
    description(name, desc);
    fprintf(stderr, "name=%s %s repeat=%d/%d duration=%.2lf\n", name, desc, r+1, REPEAT, time);
}

void report(char* name, double* times) {
    int r;
    double avg, stdev, min, max, mean;
    char desc[100], rep[255];
    FILE* f;

    description(name, desc);

    avg = 0;
    min = times[0];
    max = times[0];
    for (r = 0; r < REPEAT; r++) {
        avg += times[r];
        if (min > times[r]) {
            min = times[r];
        }
        if (max < times[r]) {
            max = times[r];
        }
    }
    avg /= REPEAT;

    stdev = 0;
    for (r = 0; r < REPEAT; r++) {
        stdev += (times[r] - avg)*(times[r] - avg);
    }
    stdev = sqrt(stdev / REPEAT);

    mean = times[REPEAT / 2];

    sprintf(rep, "name=%s %s min=%.2lf max=%.2lf mean=%.2lf avg=%.2lf stdev=%.2lf\n", name, desc, min, max, mean, avg, stdev);

    f = fopen(REPORT, "a");
    fprintf(f, rep);
    fclose(f);
    fprintf(stderr, rep);
}
