#define ArrayCount(x) (sizeof((x))/sizeof((x)[0]))

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef uint32_t u32;
typedef uint64_t u64;

typedef struct entire_file
{
    u32 size;
    char *buf;
} entire_file;

union bits32 { u32 i; float f; };
union bits64 { u64 i; double f; };

static entire_file ReadEntireFile(char *path)
{
    entire_file result = {0};

    FILE *file = fopen(path, "rb");
    if (file)
    {
	fseek(file, 0, SEEK_END);
	result.size = ftell(file);
	fseek(file, 0, SEEK_SET);

	if (result.size)
	{
	    result.buf = malloc(result.size);
	    if (result.buf)
	    {
		size_t rb = fread(result.buf, 1, result.size, file);
		if (rb != result.size)
		{
		    fprintf(stderr, "Error: short read %lu/%u\n", rb, result.size);
		    free(result.buf);
		    result.buf = 0;
		}
	    }
	    else
	    {
		fprintf(stderr, "Error: unable to allocate %u bytes\n", result.size);
		result.size = 0;
	    }
	}
    }
    else
    {
	fprintf(stderr, "Error: unable to open '%s'\n", path);
    }

    return result;
}

#if 0
static void FreeEntireFile(entire_file *file)
{
    if (file->buf)
    {
	free(file->buf);
	file->buf = 0;
    }
    file->size = 0;
}
#endif

int main(int argc, char **argv)
{
    if (argc != 4)
    {
	fprintf(stderr, "Usage: %s <input> <float|double> <output>\n", argv[0]);
	return 1;
    }

    char *path = argv[1];
    char *floatType = argv[2];
    char *outputPath = argv[3];

    bool outputAsF32 = true;
    if (strcmp(floatType, "double") == 0)
    {
	outputAsF32 = false;
    }
    else if (strcmp(floatType, "float") != 0)
    {
	fprintf(stderr, "Error: invalid float type '%s'\n", floatType);
	return 1;
    }

    entire_file file = ReadEntireFile(path);
    if (!file.size)
	return 1;

    FILE *out = fopen(outputPath, "w+");
    if (!out)
    {
	fprintf(stderr, "Error: unable to open '%s' for writing\n", outputPath);
	return 1;
    }

    float *digits = (float *)file.buf;
    u32 digitCount = file.size / sizeof(u32);
    if (outputAsF32)
    {
	for (u32 i = 0; i < digitCount; ++i)
	{
	    union bits32 cvt = {.f = digits[i]};
	    fprintf(out, "%u", cvt.i);
	    if (i + 1 < digitCount)
		fprintf(out, ", ");
	}
    }
    else
    {
	for (u32 i = 0; i < digitCount; ++i)
	{
	    union bits64 cvt = {.f = (double)digits[i]};
	    fprintf(out, "%lu", cvt.i);
	    if (i + 1 < digitCount)
		fprintf(out, ", ");
	}
    }

    fclose(out);
    //FreeEntireFile(&file);

    return 0;
}
