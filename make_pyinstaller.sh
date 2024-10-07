#!/bin/bash

pyinstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all psutil --icon "./niko.ico" \
--add-data "./kcpp_adapters:./kcpp_adapters" \
--add-data "./koboldcpp.py:." \
--add-data "./klite.embd:." \
--add-data "./kcpp_docs.embd:." \
--add-data "./kcpp_sdui.embd:." \
--add-data "./taesd.embd:." \
--add-data "./taesd_xl.embd:." \
--add-data "./koboldcpp_default.so:." \
--add-data "./koboldcpp_openblas.so:." \
--add-data "./koboldcpp_failsafe.so:." \
--add-data "./koboldcpp_noavx2.so:." \
--add-data "./koboldcpp_clblast.so:." \
--add-data "./koboldcpp_clblast_noavx2.so:." \
--add-data "./koboldcpp_vulkan_noavx2.so:." \
--add-data "./koboldcpp_vulkan.so:." \
--add-data "./rwkv_vocab.embd:." \
--add-data "./rwkv_world_vocab.embd:." \
"./koboldcpp.py" -n "koboldcpp"
