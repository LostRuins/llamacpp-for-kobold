# Guide to Nix for KoboldCpp

- KoboldCpp is available on Nixpkgs and can be installed by adding just `koboldcpp` to your `environment.systemPackages` *(or it can also be placed in `home.packages`)*.

## KoboldCpp Nix - CUDA Support

- In order to enable NVIDIA CUDA support, you will need to set `nixpkgs.config.allowUnfree`, `hardware.opengl.enable` (*`hardware.graphics.enable` if you're using 24.11 or the unstable channel*) and `nixpkgs.config.cudaSupport` to `true`, and set `nixpkgs.config.cudaArches` *(e.g., if you have an RTX 2080, you need to set `cudaArches` to `[ "sm_75" ]`)* to your GPU architecture. Find your architecture here: [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

## Hardware Support

- ✅ Vulkan: Enabled by default on Linux and macOS
- ✅ Metal: Enabled by default on macOS
- ❌ ROCm: Not currently available

## Example KoboldCpp Nix Configuration

```nix
nixpkgs.config = {
  allowUnfree = true;
  cudaSupport = true;
  cudaArches = [ "sm_75" ];
};
# NixOS 24.05
hardware.opengl.enable = true;
# NixOS 24.11 or unstable
# hardware.graphics.enable = true;
environment.systemPackages = [ pkgs.koboldcpp ];
# If you're using home-manager to install KoboldCpp
# home.packages = [ pkgs.koboldcpp ];
```

## KoboldCpp - Home Manager

The setup for Home Manager is the same as regular Nix, with one exception regarding Home Manager's instance of nixpkgs. By default, Home Manager manages its own isolated instance of nixpkgs, which has two implications:

1. You can keep your private Home Manager nixpkgs instance and simply repeat your `nixpkgs.config` in home manager.
2. You can set `home-manager.useGlobalPkgs = true;` to copy your module system's nixpkgs instance. This way, you only need to define it in your `configuration.nix`, and Home Manager will "inherit" this configuration.

## Getting Help for KoboldCpp Nix

- If you face any issues with running KoboldCpp on Nix, please open an issue [here](https://github.com/NixOS/nixpkgs/issues/new?assignees=&labels=0.kind%3A+bug&projects=&template=bug_report.md&title=)
