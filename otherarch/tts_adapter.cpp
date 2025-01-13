#include "model_adapter.h"
#include "otherarch/utils.h"

#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "corecrt_math_defines.h"

#include "src/llama-context.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

static std::string save_wav16_base64(const std::vector<float> &data, int sample_rate) {
    std::ostringstream oss;
    wav_header header;

    // Fill header fields
    header.sample_rate = sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = data.size() * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_size;

    // Write header
    oss.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write samples
    for (const auto &sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
        oss.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    // Get binary WAV data
    std::string wav_data = oss.str();
    return kcpp_base64_encode(wav_data); //return as base64 string
}

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

// very poor-man fft
static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}


static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

// TODO: not optimized at all
static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }

    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

// Convert a number less than 1000 to words
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

static std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        // Handle decimal part
        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        // Skip if fails
        return " ";
    }
}

static std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

static std::string process_text(const std::string & text) {

    std::string processed_text = replace_numbers_with_words(text);

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");
    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");
    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");
    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), "<|text_sep|>");

    return processed_text;
}


static void prompt_add(llama_tokens & prompt, const llama_tokens & tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}
static void prompt_add(llama_tokens & prompt, const llama_model * model, const std::string & txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(model, txt, add_special, parse_special);
    prompt_add(prompt, tmp);
}
static void prompt_init(llama_tokens & prompt, const llama_model * model) {
    prompt.clear();
    prompt_add(prompt, model, "<|im_start|>\n", true, true);
}

static std::vector<llama_token> prepare_guide_tokens(const llama_model * model, const std::string& str)
{
    const std::string& delimiter = "<|text_sep|>";

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        std::string current_word = str.substr(start, end - start);
        auto tmp = common_tokenize(model, current_word, false, true);
        result.push_back(tmp[0]);
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    // Add the last part
    std::string current_word = str.substr(start);
    if(current_word!="")
    {
        auto tmp = common_tokenize(model, current_word, false, true);
        if(tmp.size()>0){
            result.push_back(tmp[0]);
        }
    }
    return result;
}

std::string trim_words(const std::string& input, const std::string& separator, size_t maxWords) {
    // Split the input string by the separator
    std::vector<std::string> words;
    size_t start = 0, end;
    while ((end = input.find(separator, start)) != std::string::npos) {
        std::string last = input.substr(start, end - start);
        if (last != "") {
            words.push_back(last);
        }
        start = end + separator.length();
    }
    std::string last = input.substr(start);
    if(last!="")
    {
        words.push_back(last); // Add the last word
    }

    // Ensure no more than maxWords are kept
    if (words.size() > maxWords) {
        words.resize(maxWords);
    }

    // Reconstruct the string with the separator
    std::ostringstream result;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) result << separator;
        result << words[i];
    }

    return result.str();
}

static llama_context * ttc_ctx = nullptr; //text to codes ctx
static llama_context * cts_ctx = nullptr; //codes to speech

static int ttsdebugmode = 0;
static std::string ttsplatformenv, ttsdeviceenv, ttsvulkandeviceenv;
static std::string last_generated_audio = "";
static std::vector<llama_token> last_speaker_codes; //will store cached speaker
static int last_speaker_seed = -999;

bool ttstype_load_model(const tts_load_model_inputs inputs)
{
    //duplicated from expose.cpp
    int cl_parseinfo = inputs.clblast_info; //first digit is whether configured, second is platform, third is devices
    std::string usingclblast = "GGML_OPENCL_CONFIGURED="+std::to_string(cl_parseinfo>0?1:0);
    putenv((char*)usingclblast.c_str());
    cl_parseinfo = cl_parseinfo%100; //keep last 2 digits
    int platform = cl_parseinfo/10;
    int devices = cl_parseinfo%10;
    ttsplatformenv = "GGML_OPENCL_PLATFORM="+std::to_string(platform);
    ttsdeviceenv = "GGML_OPENCL_DEVICE="+std::to_string(devices);
    putenv((char*)ttsplatformenv.c_str());
    putenv((char*)ttsdeviceenv.c_str());
    std::string vulkan_info_raw = inputs.vulkan_info;
    std::string vulkan_info_str = "";
    for (size_t i = 0; i < vulkan_info_raw.length(); ++i) {
        vulkan_info_str += vulkan_info_raw[i];
        if (i < vulkan_info_raw.length() - 1) {
            vulkan_info_str += ",";
        }
    }
    if(vulkan_info_str!="")
    {
        ttsvulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
        putenv((char*)ttsvulkandeviceenv.c_str());
    }

    llama_backend_init();

    std::string modelfile_ttc = inputs.ttc_model_filename;
    std::string modelfile_cts = inputs.cts_model_filename;
    printf("\nLoading TTS Model, OuteTTS: %s \nWavTokenizer: %s \n",modelfile_ttc.c_str(),modelfile_cts.c_str());

    ttsdebugmode = inputs.debugmode;

    // tts init
    llama_model_params tts_model_params = llama_model_default_params();
    llama_context_params tts_ctx_params = llama_context_default_params();

    const int nthreads = 4;

    tts_model_params.use_mmap = false;
    tts_model_params.use_mlock = false;
    tts_model_params.n_gpu_layers = inputs.gpulayers; //offload if possible
    tts_model_params.split_mode = llama_split_mode::LLAMA_SPLIT_MODE_LAYER;
    tts_ctx_params.n_ctx = 8192;
    tts_ctx_params.logits_all = false;
    tts_ctx_params.offload_kqv = true;
    tts_ctx_params.n_batch = 8192;
    tts_ctx_params.n_ubatch = 512;
    tts_ctx_params.n_threads = nthreads;
    tts_ctx_params.n_threads_batch = nthreads;
    tts_ctx_params.flash_attn = false;

    llama_model * ttcmodel = llama_model_load_from_file(modelfile_ttc.c_str(), tts_model_params);
    ttc_ctx = llama_new_context_with_model(ttcmodel, tts_ctx_params);

    if (ttc_ctx == nullptr) {
        printf("\nTTS Load Error: Failed to initialize ttc context!\n");
        return false;
    }

    llama_model * ctsmodel = llama_model_load_from_file(modelfile_cts.c_str(), tts_model_params);

    tts_ctx_params.embeddings = true; //this requires embeddings instead
    cts_ctx = llama_new_context_with_model(ctsmodel, tts_ctx_params);

    if (cts_ctx == nullptr) {
        printf("\nTTS Load Error: Failed to initialize cts context!\n");
        return false;
    }

    std::vector<int> tmp = {1, 2, 3, 4};
    llama_kv_cache_clear(ttc_ctx);
    auto er = llama_decode(ttc_ctx, llama_batch_get_one(tmp.data(), tmp.size()));
    if(er!=0)
    {
        printf("\nTTS Eval returned nonzero: %d\n",er);
        return false;
    }

    printf("\nTTS Load Complete.\n");
    return true;
}

tts_generation_outputs ttstype_generate(const tts_generation_inputs inputs)
{
    tts_generation_outputs output;

    if(ttc_ctx==nullptr || cts_ctx==nullptr)
    {
        printf("\nWarning: KCPP TTS not initialized!\n");
        output.data = "";
        output.status = 0;
        return output;
    }

    std::vector<llama_token> codes;
    std::vector<llama_token> guide_tokens;
    const llama_model * model_ttc = &(ttc_ctx->model);
    const llama_model * model_cts = &(cts_ctx->model);
    const int ttc_n_vocab = llama_n_vocab(model_ttc);
    std::string prompt = inputs.prompt;
    const std::string sampletext = "but<|text_sep|>that<|text_sep|>is<|text_sep|>what<|text_sep|>it<|text_sep|>is";

    // process prompt and generate voice codes
    llama_kv_cache_clear(ttc_ctx);
    llama_kv_cache_clear(cts_ctx);
    std::vector<llama_token> prompt_inp;
    prompt_init(prompt_inp, model_ttc);
    prompt_add(prompt_inp, model_ttc, "<|text_start|>", false, true);

    int speaker_seed = inputs.speaker_seed;
    int audio_seed = inputs.audio_seed;
    if (speaker_seed <= 0 || speaker_seed==0xFFFFFFFF)
    {
        speaker_seed = (((uint32_t)time(NULL)) % 1000000u);
    }
    if (audio_seed <= 0 || audio_seed==0xFFFFFFFF)
    {
        audio_seed = (((uint32_t)time(NULL)) % 1000000u);
    }
    if(ttsdebugmode==1)
    {
        printf("\nUsing Speaker Seed: %d", speaker_seed);
        printf("\nUsing Audio Seed: %d", audio_seed);
    }

    std::mt19937 tts_rng(audio_seed);
    std::mt19937 speaker_rng(speaker_seed);

    int n_decode = 0;
    int n_predict = 2048; //will be updated later
    bool next_token_uses_guide_token = true;

    // convert the input text into the necessary format expected by OuteTTS
    std::string prompt_clean = process_text(prompt);

    //further clean it by keeping only the last 300 words
    prompt_clean = trim_words(prompt_clean,"<|text_sep|>",300);

    if(prompt_clean.size()==0)
    {
        //no input
         if(!inputs.quiet)
        {
            printf("\nTTS sent empty input.\n");
            last_generated_audio = "";
            output.data = last_generated_audio.c_str();
            output.status = 1;
            return output;
        }
    }

    if(!inputs.quiet && ttsdebugmode==1)
    {
        printf("\nInput: %s\n", prompt_clean.c_str());
    }

    //2 passes. first pass, we generate the speaker voice if required, then cache it for reuse
    //second pass, we use the speaker snipper to align output voice to match the desired speaker
    if(speaker_seed>0) //first pass
    {
        //if we have a cached speaker, reuse it
        if(last_speaker_seed==speaker_seed && !last_speaker_codes.empty())
        {
            //able to proceed, do nothing
            if(!inputs.quiet && ttsdebugmode==1)
            {
                printf("\nReuse speaker ID=%d (%d tokens)...", last_speaker_seed, last_speaker_codes.size());
            }
        } else {
            //generate the voice texture of our new speaker
            last_speaker_codes.clear();
            guide_tokens = prepare_guide_tokens(model_ttc,sampletext);
            prompt_add(prompt_inp, model_ttc, sampletext, false, true);
            prompt_add(prompt_inp, model_ttc, "<|text_end|>\n<|audio_start|>\n", false, true);
            if(!inputs.quiet && ttsdebugmode==1)
            {
                printf("\nPrepare new speaker (%d input tokens)...", prompt_inp.size());
            }
            kcpp_embd_batch tts_batch = kcpp_embd_batch(prompt_inp, 0, false, true);
            auto evalok = (llama_decode(ttc_ctx, tts_batch.batch)==0);
            if (!evalok) {
                printf("\nError: TTS prompt batch processing failed\n");
                output.data = "";
                output.status = 0;
                return output;
            }

            while (n_decode <= n_predict)
            {
                float * logits = llama_get_logits(ttc_ctx);

                //use creative settings to generate speakers
                const int topk = 20;
                const float temp = 1.2f;
                llama_token new_token_id = kcpp_quick_sample(logits,ttc_n_vocab,topk,temp,speaker_rng);

                //guide tokens help prevent hallucinations by forcing the TTS to use the correct word
                if(next_token_uses_guide_token && !llama_token_is_control(model_ttc, new_token_id) && !llama_token_is_eog(model_ttc, new_token_id))
                {
                    if(!guide_tokens.empty())
                    {
                        llama_token guide_token = guide_tokens[0];
                        guide_tokens.erase(guide_tokens.begin());
                        new_token_id = guide_token; //ensure correct word fragment is used
                    } else {
                        n_decode = n_predict; //stop generation
                    }
                }

                //this is the token id that always precedes a new word
                next_token_uses_guide_token = (new_token_id == 198);
                last_speaker_codes.push_back(new_token_id);

                // is it an end of generation? -> mark the stream as finished
                if (llama_token_is_eog(model_ttc, new_token_id) || n_decode >= n_predict) {
                    break;
                }

                n_decode += 1;
                std::vector<llama_token> next = {new_token_id};
                llama_batch batch = llama_batch_get_one(next.data(), next.size());

                // evaluate the current batch with the transformer model
                if (llama_decode(ttc_ctx, batch)) {
                    printf("\nError: TTS code generation failed!\n");
                    output.data = "";
                    output.status = 0;
                    return output;
                }
            }

            //trim everything after final <|code_end|>
            auto it = std::find(last_speaker_codes.rbegin(), last_speaker_codes.rend(), 151670);
            if (it != last_speaker_codes.rend()) {
                // Erase elements after the found 999 (inclusive)
                last_speaker_codes.erase(it.base(), last_speaker_codes.end());
            }
            last_speaker_seed = speaker_seed;
            if(!inputs.quiet && ttsdebugmode==1)
            {
                printf("\nNew speaker ID=%d created (%d tokens)...", last_speaker_seed, last_speaker_codes.size());
                const std::string inp_txt = common_detokenize(ttc_ctx, last_speaker_codes, true);
                printf("\n%s\n", inp_txt.c_str());
            }
        }
        guide_tokens.clear();
        llama_kv_cache_clear(ttc_ctx);
        prompt_init(prompt_inp, model_ttc);
        prompt_add(prompt_inp, model_ttc, "<|text_start|>", false, true);
        next_token_uses_guide_token = true;
    }

    //second pass: add the speaker before the actual prompt
    guide_tokens = prepare_guide_tokens(model_ttc,prompt_clean);
    if(speaker_seed > 0)
    {
        prompt_clean = sampletext + "<|text_sep|>" + prompt_clean;
    }
    prompt_add(prompt_inp, model_ttc, prompt_clean, false, true);

    if(!inputs.quiet)
    {
        printf("\nTTS Generating (%d input tokens)...", prompt_inp.size());
    }

    prompt_add(prompt_inp, model_ttc, "<|text_end|>\n<|audio_start|>\n", false, true);

    if(!last_speaker_codes.empty() && speaker_seed > 0) //apply speaker voice output
    {
       prompt_add(prompt_inp, last_speaker_codes);
    }

    if(!inputs.quiet && ttsdebugmode==1)
    {
        printf("\nDUMP TTS PROMPT (%d tokens):\n", prompt_inp.size());
        const std::string inp_txt = common_detokenize(ttc_ctx, prompt_inp, true);
        printf("\n%s\n", inp_txt.c_str());
    }

    //create batch with tokens for decoding prompt processing
    kcpp_embd_batch tts_batch = kcpp_embd_batch(prompt_inp, 0, false, true);

    auto evalok = (llama_decode(ttc_ctx, tts_batch.batch)==0);
    if (!evalok) {
        printf("\nError: TTS prompt batch processing failed\n");
        output.data = "";
        output.status = 0;
        return output;
    }

    // main loop
    n_decode = 0;
    n_predict = 4096; //max 4096 tokens

    while (n_decode <= n_predict)
    {
        float * logits = llama_get_logits(ttc_ctx);

        //use predictable settings to generate voice
        const int topk = 4;
        const float temp = 0.75f;
        llama_token new_token_id = kcpp_quick_sample(logits,ttc_n_vocab,topk,temp,tts_rng);

        //guide tokens help prevent hallucinations by forcing the TTS to use the correct word
        if(next_token_uses_guide_token && !llama_token_is_control(model_ttc, new_token_id) && !llama_token_is_eog(model_ttc, new_token_id))
        {
            if(!guide_tokens.empty())
            {
                llama_token guide_token = guide_tokens[0];
                guide_tokens.erase(guide_tokens.begin());
                new_token_id = guide_token; //ensure correct word fragment is used
            } else {
                n_decode = n_predict; //end generation
            }
        }

        //this is the token id that always precedes a new word
        next_token_uses_guide_token = (new_token_id == 198);
        codes.push_back(new_token_id);

        // is it an end of generation? -> mark the stream as finished
        if (llama_token_is_eog(model_ttc, new_token_id) || n_decode >= n_predict) {
            break;
        }

        n_decode += 1;
        std::vector<llama_token> next = {new_token_id};
        llama_batch batch = llama_batch_get_one(next.data(), next.size());

        // evaluate the current batch with the transformer model
        if (llama_decode(ttc_ctx, batch)) {
            printf("\nError: TTS code generation failed!\n");
            output.data = "";
            output.status = 0;
            return output;
        }
    }

    if(!inputs.quiet && ttsdebugmode==1)
    {
        const std::string inp_txt = common_detokenize(ttc_ctx, codes, true);
        printf("\nGenerated %d Codes: '%s'\n",codes.size(), inp_txt.c_str());
    }

    // remove all non-audio tokens (i.e. < 151672 || > 155772)
    codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());

    for (auto & token : codes) {
        token -= 151672;
    }

    const int n_codes = codes.size();
    if(n_codes<=1)
    {
        printf("\nWarning: TTS vocoder generated nothing!\n");
        last_generated_audio = "";
        output.data = last_generated_audio.c_str();
        output.status = 1;
        return output;
    }
    kcpp_embd_batch codebatch = kcpp_embd_batch(codes,0,false,true);

    if (llama_decode(cts_ctx, codebatch.batch) != 0) {
        printf("\nError: TTS vocoder generation failed!\n");
        output.data = "";
        output.status = 0;
        return output;
    }
    else
    {
        // spectral operations
        const int n_embd = llama_n_embd(model_cts);
        const float * embd = llama_get_embeddings(cts_ctx);
        std::vector<float> audio = embd_to_audio(embd, n_codes, n_embd, 4);

        const int n_sr = 24000; // sampling rate

        // zero out first 0.25 seconds or 0.05 depending on whether its seeded
        const int cutout = (speaker_seed>0?(24000/4):(24000/20));
        for (int i = 0; i < cutout; ++i) {
            audio[i] = 0.0f;
        }
        //add some silence at the end
        for (int i = 0; i < 24000/20; ++i) {
            audio.push_back(0.0f);
        }

        last_generated_audio = save_wav16_base64(audio, n_sr);

        if(!inputs.quiet)
        {
            printf("\nTTS Generated %d audio tokens.\n",(int) codes.size());
        }

        output.data = last_generated_audio.c_str();
        output.status = 1;
        return output;
    }
}
