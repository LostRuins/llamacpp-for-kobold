#include "ggml.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"

#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>


int main(int argc, char ** argv) {
    ggml_time_init();

     if (argc != 3) {
        fprintf(stderr, "usage: %s mmproj-f16.gguf output-mmproj-quantized.gguf\n", argv[0]);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    printf("quantizing mmproj clip model to q4_1... ");
    clip_model_quantize(fname_inp.c_str(), fname_out.c_str(), GGML_TYPE_Q4_1);
    printf("done\n");

    return 0;
}
