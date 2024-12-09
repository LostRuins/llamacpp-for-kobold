# Guide to Nix for KoboldCpp

- KoboldCpp is available on Nixpkgs and can be installed by adding just `koboldcpp` to your `environment.systemPackages` *(or it can also be placed in `home.packages`)*.

## KoboldCpp Nix - CUDA Support

- In order to enable NVIDIA CUDA support, you will need to set `nixpkgs.config.allowUnfree`, `hardware.opengl.enable` (*`hardware.graphics.enable` if you're using 24.11 or the unstable channel*) and `nixpkgs.config.cudaSupport` to `true`, and set `nixpkgs.config.cudaArches` *(e.g., if you have an RTX 2080, you need to set `cudaArches` to `[ "sm_75" ]`)* to your GPU architecture. Find your architecture here: [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- Make sure to configure `nixpkgs` settings again/separately for home-manager if `home-manager.useGlobalPkgs` is set to `false`.
- Metal is enabled by default on macOS, Vulkan support is enabled by default on both Linux and macOS, ROCm support isn't available yet.

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

## Getting Help for KoboldCpp Nix

- If you face any issues with running KoboldCpp on Nix, please open an issue [here](https://github.com/NixOS/nixpkgs/issues/new?assignees=&labels=0.kind%3A+bug&projects=&template=bug_report.md&title=)
