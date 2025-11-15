This is a CUDA-based project for recoloring an image.

You can compile with `make clean && make` and then to run, execute `./build/recolor <input-image> <output-image> <pixel-radius>` where `<pixel-radius>` is the radius around the source pixel to randomly select from for its value. It ends up with a "blur" type effect, though there is not gradient, so it's not a _good_ blur effect, but was an interesting exploration of randomness and pixel manipulation in CUDA.
