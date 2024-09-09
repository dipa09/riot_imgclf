//#include <stdio.h>

typedef __UINT8_TYPE__  u8;
typedef __UINT32_TYPE__ u32;

static inline u8 EvalPixel(u8 *image, u32 width, u32 height, u8 center, u32 x, u32 y)
{
    u8 pixel = 0;
    if (x < width && y < height)
    {
	pixel = image[y*width + x] >= center;
    }

    return pixel;
}

static inline u8 LBP(u8 *image, u32 width, u32 height, u32 x, u32 y)
{
    u8 center = image[y*width + x];
    u32 result = (1   * EvalPixel(image, width, height, center, x-1, y-1) + // top left
		  2   * EvalPixel(image, width, height, center, x,   y-1) + // top
		  4   * EvalPixel(image, width, height, center, x+1, y-1) + // top right
		  8   * EvalPixel(image, width, height, center, x+1, y)   + // right
		  16  * EvalPixel(image, width, height, center, x+1, y+1) + // bottom right
		  32  * EvalPixel(image, width, height, center, x,   y+1) + // bottom
		  64  * EvalPixel(image, width, height, center, x-1, y+1) + // bottom left
		  128 * EvalPixel(image, width, height, center, x-1, y));   // left

    return result;
}

void ExtractLBP(u8 *grayImage, u32 width, u32 height, u8 *out)
{
    for (u32 y = 0; y < height; y++)
    {
	for (u32 x = 0; x < width; ++x)
	{
	    u32 index = y*width + x;
	    out[index] = LBP(grayImage, width, height, x, y);
	}
    }
}
