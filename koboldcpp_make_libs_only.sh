#!/bin/bash
if [ ! -f "bin/micromamba" ]; then
	curl -Ls https://anaconda.org/conda-forge/micromamba/1.5.3/download/linux-64/micromamba-1.5.3-0.tar.bz2 | tar -xvj bin/micromamba
fi
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
bin/micromamba run -r conda -n linux make clean
bin/micromamba run -r conda -n linux make -j$((`nproc`+1)) LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 LLAMA_PORTABLE=1
