#include "recolor.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Image loading
Image *loadImage(const char *filename)
{
    Image *img = (Image *)malloc(sizeof(Image));
    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 3);

    if (!img->data)
    {
        fprintf(stderr, "Failed to load image: %s\n", filename);
        free(img);
        return NULL;
    }

    img->channels = 3; // Force RGB
    return img;
}

// Image saving
void saveImage(const char *filename, Image *img)
{
    stbi_write_jpg(filename, img->width, img->height, img->channels, img->data, 95);
}

// Free image memory
void freeImage(Image *img)
{
    if (img)
    {
        if (img->data)
            stbi_image_free(img->data);
        free(img);
    }
}

// Print image info
void printImageInfo(Image *img)
{
    printf("Image: %dx%d, %d channels\n", img->width, img->height, img->channels);
}