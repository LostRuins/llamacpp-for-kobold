#pragma once

#include "ggml_v3.h"

#include <fstream>
#include <vector>
#include <string>

enum ggml_v3_ftype ggml_v3_parse_ftype(const char * str);

void ggml_v3_print_ftypes(FILE * fp = stderr);

bool ggml_v3_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_v3_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);