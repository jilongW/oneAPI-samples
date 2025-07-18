# `Simple Add` Sample

The `Simple Add` sample demonstrates the simplest programming methods for using SYCL*-compliant buffers and Unified Shared Memory (USM). Additionally, building and running this sample verifies that your development environment is configured correctly for [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html).

| Property            | Description
|:---                 |:---
| What you will learn | How to use SYCL*-compliant extensions to offload computations using both buffers and USM.
| Time to complete    | 15 minutes
| Category            | Getting Started

## Purpose

The `Simple Add` sample is a simple program that adds two large vectors of integers and verifies the results. In this sample, you will see how to use the most basic code in C++ language that offloads computations to a GPU, which includes using USM and buffers.

The basic SYCL implementations explained in the sample includes device selector,
USM, buffer, accessor, kernel, and command groups.

>**Note**: See the `Base: Vector Add` sample to examine another getting started sample you can use to learn more about using the Intel® oneAPI Toolkits to develop SYCL-compliant applications for CPU and GPU devices.

## Prerequisites

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware           | GEN9 or newer
| Software           | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

This sample provides examples of both buffers and USM implementations for simple side-by-side comparison.

- USM requires an explicit wait for the asynchronous kernel's
computation to complete.
- Buffers, at the time they go out of scope, bring main
memory in sync with device memory implicitly. The explicit wait on the event is
not required as a result.

The program attempts first to run on an available GPU, and it will fall back to the system CPU if it does not detect a compatible GPU. If the program runs successfully, the name of the offload device and a success message is displayed.

> **Note**: For comprehensive information about oneAPI programming, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Simple Add` Program

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
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

### Using Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
3. Open a terminal in VS Code (**Terminal > New Terminal**).
4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
*[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

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
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Run the `Simple Add` Program

### On Linux

1. Change to the output directory.

2. Run the program for Unified Shared Memory (USM) and buffers.
   ```
   ./simple-add-buffers
   ./simple-add-usm
   ```

### On Windows

1. Change to the output directory.

2. Run the program for Unified Shared Memory (USM) and buffers.
   ```
   simple-add-usm.exe
   simple-add-buffers.exe
   ```

## Example Output
```
simple-add output snippet changed to:
Running on device:        Intel(R) Gen9 HD Graphics NEO
Array size: 10000
[0]: 0 + 100000 = 100000
[1]: 1 + 100000 = 100001
[2]: 2 + 100000 = 100002
...
[9999]: 9999 + 100000 = 109999
Successfully completed on device.
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
