# `Base: Vector Add` Sample

The `Base: Vector Add` is the equivalent of a **Hello, World!** sample for data parallel programs, so the sample code demonstrates some core features of SYCL*. Additionally, building and running this sample verifies that your development environment is configured correctly for [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html).

| Area                     | Description
|:---                      |:---
| What you will learn      | How to begin using SYCL* to offload computations to CPUs and accelerators
| Time to complete         | ~15 minutes
| Category                 | Getting Started

## Purpose

The `Base: Vector Add` is a relatively simple program that adds two large vectors of integers and verifies the results. This program uses C++ and SYCL* for Intel® CPU and accelerators. By reviewing the code in this sample, you can learn how to use C++ code to offload computations to a GPU. This includes using Unified Shared Memory (USM) and buffers.

- USM requires an explicit wait for the asynchronous computation on the kernel to complete.

- When they go out of scope, buffers synchronize main memory with device memory implicitly; an explicit wait on the event is not required.

This sample provides example implementations of both Unified Shared Memory (USM) and buffers for side-by-side comparison.

>**Note**: See the `Simple Add` sample to examine another getting started sample you can use to learn more about using the Intel® oneAPI Toolkits to develop SYCL-compliant applications for CPU and GPU devices.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                          | GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

The basic SYCL implementation explained in the code includes device selector, USM, buffer, accessor, kernel, and command groups.

The code attempts to execute on an available GPU and fallback to the system CPU if a compatible GPU is not detected. If successful, the name of the offload device and a success message is displayed, which indicates your development environment is set up correctly.

> **Note**: For comprehensive information about oneAPI programming, see the [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Base: Vector Add` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Using Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
3. Open a terminal in VS Code (**Terminal > New Terminal**).
4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

#### Configure the build system

1. Change to the sample directory.
2.
   Configure the project to use the buffer-based implementation.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   or

   Configure the project to use the Unified Shared Memory (USM) based implementation.
   ```
   mkdir build
   cd build
   cmake .. -DUSM=1
   ```

#### Build

1. Build the program.
   ```
   make cpu-gpu
   ```
2. Clean the program. (Optional)
   ```
   make clean
   ```

### On Windows*

#### Configure the build system

1. Change to the sample directory.
2.
   Configure the project to use the buffer-based implementation.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   or

   Configure the project to use the Unified Shared Memory (USM) based implementation.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" .. -DUSM=1
   ```

#### Build

1. Build the program.
   ```
   nmake cpu-gpu
   ```
2. Clean the program. (Optional)
   ```
   nmake clean
   ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `Base: Vector Add` Program

### Configurable Parameters

The source files (`vector-add-buffers.cpp` and `vector-add-usm.cpp`) specify the default vector size of **10000**. You can change the vector size in one or both files if necessary.

### On Linux

1. Run the program for Unified Shared Memory (USM) and buffers.
    ```
    ./vector-add-buffers
    ./vector-add-usm
    ```

### On Windows

1. Run the program for Unified Shared Memory (USM) and buffers.
    ```
    vector-add-usm.exe
    vector-add-buffers.exe
    ```

## Example Output
```
Running on device:        Intel(R) Gen(R) HD Graphics NEO
Vector size: 10000
[0]: 0 + 0 = 0
[1]: 1 + 1 = 2
[2]: 2 + 2 = 4
...
[9999]: 9999 + 9999 = 19998
Vector add successfully completed on device.
```

## License

Code samples are licensed under the MIT license. See [License.txt](License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
