# DLSS Super Resolution and DLSS Frame Generation via Streamline

This sample demonstrates integration of [Streamline](https://developer.nvidia.com/rtx/streamline) into a Vulkan-based application and using it to add [NVIDIA Reflex](https://developer.nvidia.com/performance-rendering-tools/reflex), [DLSS Super Resolution, and DLSS Frame Generation](https://developer.nvidia.com/rtx/dlss/get-started).

Streamline generally supports two methods of integrating it, either automatically by linking against the `sl.interposer.lib` library instead of `vulkan-1.lib` (assuming the application was previously statically linking Vulkan already), or manually by only retrieving the relevant Vulkan entry points from Streamline and having the rest go directly through Vulkan as usual (an application can use the `vkGetInstanceProcAddr`/`vkGetDeviceProcAddr` provided by Streamline to fill its dispatch tables). This sample implements both methods, which can be toggled between via a CMake option called `STREAMLINE_MANUAL_HOOKING`.\
When enabled, the sample will link against Vulkan normally, dynamically load Streamline at runtime and only get the required Vulkan functions from it to call. When disabled, the sample will link against `sl.interposer.lib` instead of `vulkan-1.lib`.\
The manual hooking method can offer better performance because of less overhead (avoids having to redirect all Vulkan calls through Streamline) and is not too difficult to implement when an application already loads all the Vulkan entry points dynamically. It does however also require querying and adding all the necessary Vulkan extensions and features Streamline wants, while the automatic method will add those during device creation behind the scenes without changes to the application.

To debug DLSS, you can replace the DLSS DLLs installed by CMake with their [development variants](https://github.com/NVIDIAGameWorks/Streamline/tree/main/bin/x64/development) and enable their overlay with a [special registry key](https://github.com/NVIDIAGameWorks/Streamline/blob/main/scripts/ngx_driver_onscreenindicator.reg).\
In addition, check out the Streamline ImGui plugin [documentation](https://github.com/NVIDIAGameWorks/Streamline/tree/main/docs). It can be enabled by adding `sl::kFeatureImGUI` to the `SL_FEATURES` array at the top of main.cpp.

## Build and Run

Clone https://github.com/nvpro-samples/nvpro_core.git next to this repository (or pull latest `master` if you already have it).

`mkdir build && cd build && cmake .. # Or use CMake GUI`

If there are missing dependencies (e.g. glfw), run `git submodule update --init --recursive --checkout --force` in the `nvpro_core` repository (and also this one).

Then start the generated `.sln` in VS or run `make -j`.

Run `vk_streamline` or `../../bin_x64/Release/vk_streamline.exe`.

## License

Released under the Apache License, Version 2.0. Please see the copyright notice in the [LICENSE](LICENSE) file.

This project uses the NVIDIA nvpro-samples framework and NVIDIA Streamline SDK. Please see the license for nvpro_core [here](https://github.com/nvpro-samples/nvpro_core/blob/master/LICENSE), and the third-party packages it uses [here](https://github.com/nvpro-samples/nvpro_core/tree/master/PACKAGE-LICENSES). Please see the license for Streamline [here](https://github.com/NVIDIAGameWorks/Streamline/blob/main/license.txt), and the third-party packages it uses [here](https://github.com/NVIDIAGameWorks/Streamline/blob/main/3rd-party-licenses.md).
