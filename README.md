# koboldcpp

KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models, inspired by the original **KoboldAI**. It's a single self-contained distributable from Concedo, that builds off llama.cpp, and adds a versatile **KoboldAI API endpoint**, additional format support, Stable Diffusion image generation, speech-to-text, backward compatibility, as well as a fancy UI with persistent stories, editing tools, save formats, memory, world info, author's note, characters, scenarios and everything KoboldAI and KoboldAI Lite have to offer.

![Preview](media/preview.png)
![Preview](media/preview2.png)
![Preview](media/preview3.png)
![Preview](media/preview4.png)

## Windows Usage (Precompiled Binary, Recommended)
- Windows binaries are provided in the form of **koboldcpp.exe**, which is a pyinstaller wrapper containing all necessary files. **[Download the latest koboldcpp.exe release here](https://github.com/LostRuins/koboldcpp/releases/latest)**
- To run, simply execute **koboldcpp.exe**.
- Launching with no command line arguments displays a GUI containing a subset of configurable settings. Generally you dont have to change much besides the `Presets` and `GPU Layers`. Read the `--help` for more info about each settings.
- By default, you can connect to http://localhost:5001
- You can also run it using the command line. For info, please check `koboldcpp.exe --help`

## Linux Usage (Precompiled Binary, Recommended)
On modern Linux systems, you should download the `koboldcpp-linux-x64-cuda1150` prebuilt PyInstaller binary on the **[releases page](https://github.com/LostRuins/koboldcpp/releases/latest)**. Simply download and run the binary.

Alternatively, you can also install koboldcpp to the current directory by running the following terminal command:
```
curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64-cuda1150 && chmod +x koboldcpp
```
After running this command you can launch Koboldcpp from the current directory using `./koboldcpp` in the terminal (for CLI usage, run with `--help`).

## Run on Colab
- KoboldCpp now has an **official Colab GPU Notebook**! This is an easy way to get started without installing anything in a minute or two. [Try it here!](https://colab.research.google.com/github/LostRuins/koboldcpp/blob/concedo/colab.ipynb).
- Note that KoboldCpp is not responsible for your usage of this Colab Notebook, you should ensure that your own usage complies with Google Colab's terms of use.

## Run on RunPod
- KoboldCpp can now be used on RunPod cloud GPUs! This is an easy way to get started without installing anything in a minute or two, and is very scalable, capable of running 70B+ models at afforable cost. [Try our RunPod image here!](https://koboldai.org/runpodcpp).

## Docker
- The official docker can be found at https://hub.docker.com/r/koboldai/koboldcpp
- If you're building your own docker, remember to set CUDA_DOCKER_ARCH or enable LLAMA_PORTABLE

## MacOS
- You will need to clone the repo and compile from source code, see Compiling for MacOS below.

## Obtaining a GGUF model
- KoboldCpp uses GGUF models. They are not included here, but you can download GGUF files from other places such as [TheBloke's Huggingface](https://huggingface.co/TheBloke). Search for "GGUF" on huggingface.co for plenty of compatible models in the `.gguf` format.
- For beginners, we recommend the models [Airoboros Mistral](https://huggingface.co/TheBloke/airoboros-mistral2.2-7B-GGUF/resolve/main/airoboros-mistral2.2-7b.Q4_K_S.gguf) or [Tiefighter 13B](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter-GGUF/resolve/main/LLaMA2-13B-Tiefighter.Q4_K_S.gguf) (larger model).
- [Alternatively, you can download the tools to convert models to the GGUF format yourself here](https://github.com/LostRuins/koboldcpp/releases/download/v1.69.1/koboldcpp_tools_6jul.zip). Run `convert-hf-to-gguf.py` to convert them, then `quantize_gguf.exe` to quantize the result.

## Improving Performance
- **GPU Acceleration**: If you're on Windows with an Nvidia GPU you can get CUDA support out of the box using the `--usecublas`  flag (Nvidia Only), or `--usevulkan` (Any GPU), make sure you select the correct .exe with CUDA support.
- **GPU Layer Offloading**: Add `--gpulayers` to offload model layers to the GPU. The more layers you offload to VRAM, the faster generation speed will become. Experiment to determine number of layers to offload, and reduce by a few if you run out of memory.
- **Increasing Context Size**: Use `--contextsize (number)` to increase context size, allowing the model to read more text. Note that you may also need to increase the max context in the KoboldAI Lite UI as well (click and edit the number text field).
- **Old CPU Compatibility**: If you are having crashes or issues, you can try turning off BLAS with the `--noblas` flag. You can also try running in a non-avx2 compatibility mode with `--noavx2`. Lastly, you can try turning off mmap with `--nommap`.

For more information, be sure to run the program with the `--help` flag, or **[check the wiki](https://github.com/LostRuins/koboldcpp/wiki).**

## Compiling KoboldCpp From Source Code

### Compiling on Linux (Using koboldcpp.sh automated compiler script)
when you can't use the precompiled binary directly, we provide an automated build script which uses conda to obtain all dependencies, and generates (from source) a ready-to-use a pyinstaller binary for linux users.
- Clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- Simply execute the build script with `./koboldcpp.sh dist` and run the generated binary. (Not recomended for systems that already have an existing installation of conda. Dependencies: curl, bzip2)
```
./koboldcpp.sh # This launches the GUI for easy configuration and launching (X11 required).
./koboldcpp.sh --help # List all available terminal commands for using Koboldcpp, you can use koboldcpp.sh the same way as our python script and binaries.
./koboldcpp.sh rebuild # Automatically generates a new conda runtime and compiles a fresh copy of the libraries. Do this after updating Koboldcpp to keep everything functional.
./koboldcpp.sh dist # Generate your own precompiled binary (Due to the nature of Linux compiling these will only work on distributions equal or newer than your own.)
```

### Compiling on Linux (Manual Method)
- To compile your binaries from source, clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- A makefile is provided, simply run `make`.
- Optional OpenBLAS: Link your own install of OpenBLAS manually with `make LLAMA_OPENBLAS=1`
- Optional CLBlast: Link your own install of CLBlast manually with `make LLAMA_CLBLAST=1`
- Note: for these you will need to obtain and link OpenCL and CLBlast libraries.
  - For Arch Linux: Install `cblas` `openblas` and `clblast`.
  - For Debian: Install `libclblast-dev` and `libopenblas-dev`.
- You can attempt a CuBLAS build with `LLAMA_CUBLAS=1`. You will need CUDA Toolkit installed. Some have also reported success with the CMake file, though that is more for windows.
- For a full featured build (all backends), do `make LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 LLAMA_CUBLAS=1 LLAMA_VULKAN=1`. (Note that `LLAMA_CUBLAS=1` will not work on windows, you need visual studio)
- After all binaries are built, you can run the python script with the command `koboldcpp.py [ggml_model.gguf] [port]`

### Compiling on Windows
- You're encouraged to use the .exe released, but if you want to compile your binaries from source at Windows, the easiest way is:
  - Get the latest release of w64devkit (https://github.com/skeeto/w64devkit). Be sure to use the "vanilla one", not i686 or other different stuff. If you try they will conflit with the precompiled libs!
  - Clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
  - Make sure you are using the w64devkit integrated terminal, then run `make` at the KoboldCpp source folder. This will create the .dll files.
  - If you want to generate the .exe file, make sure you have the python module PyInstaller installed with pip (`pip install PyInstaller`). Then run the script `make_pyinstaller.bat`
  - The koboldcpp.exe file will be at your dist folder.
- **Building with CUDA**: Visual Studio, CMake and CUDA Toolkit is required. Clone the repo, then open the CMake file and compile it in Visual Studio. Copy the `koboldcpp_cublas.dll` generated into the same directory as the `koboldcpp.py` file. If you are bundling executables, you may need to include CUDA dynamic libraries (such as `cublasLt64_11.dll` and `cublas64_11.dll`) in order for the executable to work correctly on a different PC.
- **Replacing Libraries (Not Recommended)**: If you wish to use your own version of the additional Windows libraries (OpenCL, CLBlast and OpenBLAS), you can do it with:
  - OpenCL - tested with https://github.com/KhronosGroup/OpenCL-SDK . If you wish to compile it, follow the repository instructions. You will need vcpkg.
  - CLBlast - tested with https://github.com/CNugteren/CLBlast . If you wish to compile it you will need to reference the OpenCL files. It will only generate the ".lib" file if you compile using MSVC.
  - OpenBLAS - tested with https://github.com/xianyi/OpenBLAS .
  - Move the respectives .lib files to the /lib folder of your project, overwriting the older files.
  - Also, replace the existing versions of the corresponding .dll files located in the project directory root (e.g. libopenblas.dll).
  - Make the KoboldCpp project using the instructions above.

### Compiling on MacOS
- You can compile your binaries from source. You can clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- A makefile is provided, simply run `make`.
- If you want Metal GPU support, instead run `make LLAMA_METAL=1`, note that MacOS metal libraries need to be installed.
- After all binaries are built, you can run the python script with the command `koboldcpp.py --model [ggml_model.gguf]` (and add `--gpulayers (number of layer)` if you wish to offload layers to GPU).

### Compiling on Android (Termux Installation)
- [Install and run Termux from F-Droid](https://f-droid.org/en/packages/com.termux/)
- Enter the command `termux-change-repo` and choose `Mirror by BFSU`
- Install dependencies with `pkg install wget git python` (plus any other missing packages)
- Install dependencies `apt install openssl` (if needed)
- Clone the repo `git clone https://github.com/LostRuins/koboldcpp.git`
- Navigate to the koboldcpp folder `cd koboldcpp`
- Build the project `make`
- Grab a small GGUF model, such as `wget https://huggingface.co/concedo/KobbleTinyV2-1.1B-GGUF/resolve/main/KobbleTiny-Q4_K.gguf`
- Start the python server `python koboldcpp.py --model KobbleTiny-Q4_K.gguf`
- Connect to `http://localhost:5001` on your mobile browser
- If you encounter any errors, make sure your packages are up-to-date with `pkg up`
- GPU acceleration for Termux may be possible but I have not explored it. If you find a good cross-device solution, do share or PR it.

## AMD Users
- Please check out https://github.com/YellowRoseCx/koboldcpp-rocm

## Third Party Resources
- These unofficial resources have been contributed by the community, and may be outdated or unmaintained. No official support will be provided for them!
  - Arch Linux Packages: [CUBLAS](https://aur.archlinux.org/packages/koboldcpp-cuda), and [HIPBLAS](https://aur.archlinux.org/packages/koboldcpp-hipblas).
  - Unofficial Dockers: [korewaChino](https://github.com/korewaChino/koboldCppDocker) and [noneabove1182](https://github.com/noneabove1182/koboldcpp-docker)
  - Nix & NixOS: KoboldCpp is available on Nixpkgs and can be installed by adding just `koboldcpp` to your `environment.systemPackages`.
    - Make sure to have `nixpkgs.config.allowUnfree`, `hardware.opengl.enable` *(`hardware.graphics.enable` if you're using unstable channel)* and `nixpkgs.config.cudaSupport` set to `true` to enable CUDA.
    - Metal is enabled by default on macOS, Vulkan support is enabled by default on both Linux and macOS, ROCm support isn't available yet.
    - You can also use `nix3-run` to use KoboldCpp: `nix run --expr ``with import <nixpkgs> { config = { allowUnfree = true; cudaSupport = true; }; }; koboldcpp`` --impure`
    - Or use `nix-shell`: `nix-shell --expr 'with import <nixpkgs> { config = { allowUnfree = true; cudaSupport = true; }; }; koboldcpp' --run "koboldcpp" --impure`
    - Packages (like OpenBlast, CLBLast, Vulkan, etc.) can be overridden, please refer to the [17th Nix Pill - Nixpkgs Overriding Packages](https://nixos.org/guides/nix-pills/17-nixpkgs-overriding-packages)

## Questions and Help Wiki
- **First, please check out [The KoboldCpp FAQ and Knowledgebase](https://github.com/LostRuins/koboldcpp/wiki) which may already have answers to your questions! Also please search through past issues and discussions.**
- If you cannot find an answer, open an issue on this github, or find us on the [KoboldAI Discord](https://koboldai.org/discord).

## KoboldCpp and KoboldAI API Documentation
- [Documentation for KoboldAI and KoboldCpp endpoints can be found here](https://lite.koboldai.net/koboldcpp_api)

## KoboldCpp Public Demo
- [A public KoboldCpp demo can be found at our Huggingface Space. Please do not abuse it.](https://koboldai-koboldcpp-tiefighter.hf.space/)

## Considerations
- For Windows: No installation, single file executable, (It Just Works)
- Since v1.0.6, requires libopenblas, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without BLAS.
- Since v1.15, requires CLBlast if enabled, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without CLBlast.
- Since v1.33, you can set the context size to be above what the model supports officially. It does increases perplexity but should still work well below 4096 even on untuned models. (For GPT-NeoX, GPT-J, and Llama models) Customize this with `--ropeconfig`.
- Since v1.42, supports GGUF models for LLAMA and Falcon
- Since v1.55, lcuda paths on Linux are hardcoded and may require manual changes to the makefile if you do not use koboldcpp.sh for the compilation.
- Since v1.60, provides native image generation with StableDiffusion.cpp, you can load any SD1.5 or SDXL .safetensors model and it will provide an A1111 compatible API to use.
- **I try to keep backwards compatibility with ALL past llama.cpp models**. But you are also encouraged to reconvert/update your models if possible for best results.

## License
- The original GGML library and llama.cpp by ggerganov are licensed under the MIT License
- However, KoboldAI Lite is licensed under the AGPL v3.0 License
- KoboldCpp code and other files are also under the AGPL v3.0 License unless otherwise stated

## Notes
- If you wish, after building the koboldcpp libraries with `make`, you can rebuild the exe yourself with pyinstaller by using `make_pyinstaller.bat`
- API documentation available at `/api` (e.g. `http://localhost:5001/api`) and https://lite.koboldai.net/koboldcpp_api. An OpenAI compatible API is also provided at `/v1` route (e.g. `http://localhost:5001/v1`).
- **All up-to-date GGUF models are supported**, and KoboldCpp also includes backward compatibility for older versions/legacy GGML `.bin` models, though some newer features might be unavailable.
- An incomplete list of models and architectures is listed, but there are *many hundreds of other GGUF models*. In general, if it's GGUF, it should work.
  - Llama / Llama2 / Llama3 / Alpaca / GPT4All / Vicuna / Koala / Pygmalion / Metharme / WizardLM
  - Mistral / Mixtral / Miqu
  - Qwen / Qwen2 / Yi
  - Gemma / Gemma2
  - GPT-2 / Cerebras
  - Phi-2 / Phi-3
  - GPT-NeoX / Pythia / StableLM / Dolly / RedPajama
  - GPT-J / RWKV4 / MPT / Falcon / Starcoder / Deepseek and many more
  - [Stable Diffusion 1.5 and SDXL safetensor models](https://github.com/LostRuins/koboldcpp/wiki#can-i-generate-images-with-koboldcpp)
  - [LLaVA based Vision models and multimodal projectors (mmproj)](https://github.com/LostRuins/koboldcpp/wiki#what-is-llava-and-mmproj)
  - [Whisper models for Speech-To-Text](https://huggingface.co/koboldcpp/whisper/tree/main)
