# `ScalarProd` Sample

The `ScalarProd` sample calculates the scalar products of a given set of input vector pairs. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to begin migrating CUDA to SYCL
| Time to complete       | 15 minutes
| Category               | Concepts and Functionality

>**Note**: This sample is migrated from the NVIDIA CUDA sample. See the [ScalarProd](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/scalarProd) sample in the NVIDIA/cuda-samples GitHub.

## Purpose

The `ScalarProd` sample shows the scalar product computation on both GPU and CPU devices. The parallel implementation demonstrates the use of concepts, such as

- Cooperative groups
- Memory coalescing
- Reduction

> **Note**: The sample used the open-source SYCLomatic tool that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the Intel® DPC++ Compatibility Tool available to augment the Base Toolkit.

This sample contains two versions of the code in the following folders:

| Folder Name          | Description
|:---                  |:---
|`01_dpct_output`      | Contains the output of the SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. 
|`02_sycl_migrated`    | Contains migrated and cleaned-up SYCL code from CUDA code.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br>Intel® Gen11 <br>Intel® Xeon CPU <br>Intel® Data Center GPU Max <br> Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software              | SYCLomatic (Tag - 20241104) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2025.0 <br> oneAPI for NVIDIA GPU plugin(version 2024.1) from Codeplay (to run SYCL™ applications on NVIDIA® GPUs)

For more information on how to install Syclomatic Tool & DPC++ CUDA® plugin, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) .<br>
How to run SYCL™ applications on NVIDIA® GPUs, refer to 
[oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from Codeplay.

## Key Implementation Details

The kernel implementation calculates the scalar products of `VectorN` vectors of `ElementN` elements on GPU with the following parameter restrictions:
1) ElementN is strongly preferred to be a multiple of warp size to meet alignment constraints of memory coalescing.
2) The total number of accumulators must be a power of two to meet memory coalescing alignment constraints.

>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.

## CUDA source code evaluation
The `ScalarProd` CUDA sample implementation consists of calculating the scalar product of a given set of input vector pairs on both GPU and CPU devices.
- `scalarProdCPU` function calculates the scalar product on the CPU.
- `scalarProdGPU` kernel calculates the scalar product on the GPU
After the computation of scalar products, the max absolute difference and the L1 distance i.e., L1 Error between CPU and GPU results are calculated.

> **Note**: For more information on how to use the Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the Code Using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the ScalarProd sample directory.
   ```
   cd cuda-samples/Samples/2_Concepts_and_Techniques/scalarProd/
   ```
3. Generate a compilation database with intercept-build.
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The `--in-root` specifies the path to the root of the source tree to be migrated. The `--gen-helper-function`  option will make a copy of the dpct header files/functions used in migrated code into the dpct_output folder as included folder. 
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function 
   ```

## Build and Run the `ScalarProd` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script at the root of your oneAPI installation.
>
> Linux*:
> - For system-wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```
   > **Note**:
   > - By default, no flags are enabled during the build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, and Xeon CPU.
   > - Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
   > - Enable **NVIDIA_GPU** flag during the build which supports NVIDIA GPUs([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from 
   Codeplay is required to build for NVIDIA GPUs).
   >   

   By default, this command sequence will build the  `01_dpct_output` and `02_sycl_migrated` versions of the program.

4. Run the program.

   Run `01_dpct_output` on GPU.
   ```
   $ make run
   ```   
   Run `01_dpct_output` for CPU.
   ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run
    $ unset ONEAPI_DEVICE_SELECTOR
   ```
   
   Run `02_sycl_migrated` on GPU.
   ```
   $ make run_sm
   ```   
   Run `02_sycl_migrated` for CPU.
   ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run_sm
    $ unset ONEAPI_DEVICE_SELECTOR
   ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2025-0/overview.html) for more information on using the utility.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

