#define ArrayCount(x) (sizeof((x))/sizeof((x)[0]))

#include <stdio.h>
#include <stdint.h>

#if RIOT
# include "ztimer.h"
# include "benchmark.h"
#else
# define BENCHMARK_FUNC(name, runs, func) func
#endif

static char *labels[] =
{
#include "labels.h"
};

#include "model.h"

#ifdef EMLEARN
# define LIBNAME "EMLEARN"
# define Predict(f) model_predict((float *)f, 0)

typedef uint32_t ml_uint;

#elif defined MICROMLGEN
# define LIBNAME "MICROMLGEN"
# define Predict(f) predict((float *)f)

typedef uint32_t ml_uint;

#elif defined M2CGEN
# define LIBNAME "M2CGEN"

#include <float.h>

typedef uint64_t ml_uint;

static unsigned Predict(ml_uint *features)
{
    double output[ArrayCount(labels)];
    score((double *)features, output);

    double max = -DBL_MIN;
    unsigned index = 0;
    for (int i = 0; i < ArrayCount(output); ++i)
    {
	double val = output[i];
	if (val > max)
	{
	    max = val;
	    index = i;
	}
    }

    return index;
}

#else
# error "ML library not defined"
#endif

static ml_uint features[] =
{
#include "features.h"
};

int main(void)
{
    unsigned index;

    printf("[" LIBNAME "] ");

    BENCHMARK_FUNC("Prediction", 5, index = Predict(features));

    printf("Predicted label is %s\n", labels[index]);
}
