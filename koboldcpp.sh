#!/bin/bash

if [ ! -f "bin/micromamba" ]; then
	curl -Ls https://anaconda.org/conda-forge/micromamba/1.5.3/download/linux-64/micromamba-1.5.3-0.tar.bz2 | tar -xvj bin/micromamba
fi

if [[ ! -f "conda/envs/linux/bin/python" || $1 == "rebuild" ]]; then
	cp environment.yaml environment.tmp.yaml
	if [ -n "$KCPP_CUDA" ]; then
		sed -i -e "s/nvidia\/label\/cuda-11.5.0/nvidia\/label\/cuda-$KCPP_CUDA/g" environment.tmp.yaml
	else
		KCPP_CUDA=11.5.0
	fi
	bin/micromamba create --no-shortcuts -r conda -n linux -f environment.tmp.yaml -y
	bin/micromamba create --no-shortcuts -r conda -n linux -f environment.tmp.yaml -y
	bin/micromamba run -r conda -n linux make clean
	echo $KCPP_CUDA > conda/envs/linux/cudaver
	echo rm environment.tmp.yaml
fi
KCPP_CUDA=$(<conda/envs/linux/cudaver)
KCPP_CUDAAPPEND=-cuda${KCPP_CUDA//.}$KCPP_APPEND

bin/micromamba run -r conda -n linux make LLAMA_VULKAN=1 LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 LLAMA_CUBLAS=1 LLAMA_PORTABLE=1 LLAMA_ADD_CONDA_PATHS=1

if [[ $1 == "rebuild" ]]; then
	echo Rebuild complete, you can now try to launch Koboldcpp.
elif [[ $1 == "dist" ]]; then
	bin/micromamba remove -r conda -n linux --force ocl-icd -y
	bin/micromamba run -r conda -n linux pyinstaller --noconfirm --onefile --collect-all customtkinter --add-data='./koboldcpp_default.so:.' --add-data='./koboldcpp_cublas.so:.' --add-data='./koboldcpp_vulkan.so:.' --add-data='./koboldcpp_openblas.so:.' --add-data='./koboldcpp_clblast.so:.' --add-data "./koboldcpp_failsafe.so:." --add-data "./koboldcpp_noavx2.so:." --add-data "./koboldcpp_clblast_noavx2.so:." --add-data "./koboldcpp_vulkan_noavx2.so:." --add-data "./koboldcpp_vulkan.so:." --add-data='./klite.embd:.' --add-data='./kcpp_docs.embd:.' --add-data='./rwkv_vocab.embd:.' --add-data='./rwkv_world_vocab.embd:.' --clean --console koboldcpp.py -n "koboldcpp-linux-x64$KCPP_CUDAAPPEND"
	bin/micromamba run -r conda -n linux pyinstaller --noconfirm --onefile --collect-all customtkinter --add-data='./koboldcpp_default.so:.' --add-data='./koboldcpp_openblas.so:.' --add-data='./koboldcpp_vulkan.so:.' --add-data='./koboldcpp_clblast.so:.' --add-data "./koboldcpp_failsafe.so:." --add-data "./koboldcpp_noavx2.so:." --add-data "./koboldcpp_clblast_noavx2.so:." --add-data "./koboldcpp_vulkan_noavx2.so:." --add-data "./koboldcpp_vulkan.so:." --add-data='./klite.embd:.' --add-data='./kcpp_docs.embd:.' --add-data='./rwkv_vocab.embd:.' --add-data='./rwkv_world_vocab.embd:.' --clean --console koboldcpp.py -n "koboldcpp-linux-x64-nocuda$KCPP_APPEND"
	bin/micromamba install -r conda -n linux ocl-icd -c conda-forge -y
else
	bin/micromamba run -r conda -n linux python koboldcpp.py $*
fi
