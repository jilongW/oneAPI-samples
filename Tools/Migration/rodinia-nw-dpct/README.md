# Intel® DPC++ Compatibility Tool: `Needleman-Wunsch` Sample

This project demonstrates how to migrate a Make/CMake project from CUDA to
SYCL* using the Intel® DPC++ Compatibility Tool.

| Optimized for         | Description
|:---                   |:---
| OS                    | Linux* Ubuntu* 18.04 <br> Windows* 10, 11
| Software              | Intel® DPC++ Compatibility Tool
| What you will learn   | Simple invocation of dpct to migrate CUDA code
| Time to complete      | 10 minutes

The program, `needleman-Wunsch`, naively implements the [Needleman-Wunsch
algorithm](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm), which is used in bioinformatics to align protein and
nucleotide sequences. The code is based on [Rodinia](http://lava.cs.virginia.edu/Rodinia/download_links.htm), a set of
benchmarks for heterogeneous computing. As compared to the `Intel® DPC++
Compatibility Tool: Vector Add` sample, this sample represents a more typical
example of migrating a working project.

If your project uses Make or CMake, you can use compilation database support to
provide compilation options, settings, macro definitions, and include paths to
the Intel® DPC++ Compatibility Tool. The compilations database is a JSON* file
containing the commands required to build a particular project. You can generate
a compilation database by running the intercept-build script described below.


## Key Implementation Details

Use the `intercept-build` tool to automatically generate a compilation
database in a JSON file that contains the build commands for the Intel® DPC++
Compatibility Tool to use. Migrate the project and prepare the project to
build and run using the Intel&reg; oneAPI DPC++ Compiler


## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## Migrating the CUDA Sample to Data Parallel C++ with the Intel® DPC++ Compatibility Tool

Building and running the CUDA sample is not required to migrate this project
to a SYCL-compliant project.

> **Note**: Certain CUDA header files, referenced by the CUDA application
> source files to be migrated, need to be accessible for the migration step.
> See *Before you Begin* in [Get Started with the Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-dpcpp-compatibility-tool/top.html#top_BEFORE_YOU_BEGIN).

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### On a Linux* System

1. Generate a [compilation database](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
   for the project using `intercept-build`. The compilation database is a
   listing of all compilation commands invoked during the build process. The
   `dpct` utility uses this information to determine the files to process.

```sh
$ make clean
$ intercept-build make
```
   This creates the file `compile_commands.json` in the working directory.

2. Use the Intel® DPC++ Compatibility Tool and compilation database to migrate
   the CUDA code. The new project will be created in the `migration` directory.
   The dpct `--in-root` option is used to set the root location of the program
   sources that are to be migrated. Only files and folders located within the
   --in-root directory will be considered for migration by the tool. Files located
   outside the`--in-root` directory are considered system files and will not be
   migrated, even if they are included by a source file located within the
   `--in-root`directory.

```sh
$ dpct -p compile_commands.json --in-root=. --out-root=migration
```

3. Inspect the migrated source code, address any `DPCT` warnings generated
   by the Intel® DPC++ Compatibility Tool, and verify the new program correctness.

Warnings are printed to the console and added as comments in the migrated
source. See *Diagnostic Reference* in the [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) for more information on what each warning means.

This sample should generate the following warning messages:
```
warning: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
warning: DPCT1043:1: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
warning: DPCT1009:2: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
warning: DPCT1009:3: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
warning: DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
warning: DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
```

See the section titled **Addressing Warnings in Migrated Code** below to
understand how to resolve the warnings.

4. Copy the original `Makefile` into the `migration` folder and update the
   copy to build the migrated project using SYCL. Replace the CUDA
   configurations in that new `Makefile` with the following for SYCL:

```make
CXX = dpcpp
TARGET = needleman_wunsch_dpcpp
SRCS = src/needle.dp.cpp
DEPS = src/needle_kernel.dp.cpp src/needle.h

# The remainder of the makefile should work without changes.
```

5. Switch to the migration directory: `cd migration`.

6. Build the migrated sample: `make`.

If you have not addressed all of the warnings, the compilation step will fail
for this sample due to some code that could not be migrated by the `dpct`
utility. You should see a compilation error similar to the following:
```
error: assigning to 'int' from incompatible type 'typename info::param_traits<info::device, (device)4143U>::return_type' (aka 'basic_string<char>')

```

7. After you have fixed the migrated source files, build and run the migrated
   sample using: `make run`. You should see logs indicating that the matrix is
   being processed.

8. Clean up the build: `make clean`.


## Windows

1. Open the migration wizard at `Extensions` > `Intel` > `Migrate Project to DPC++`
   and choose the `needleman-wunsch.vcxproj` project.

2. Configure and run the migration. Use the default settings to create a new
   project, which will be added to the open solution.

3. Inspect the generated source code and address any `DPCT` warnings generated
   by the Intel® DPC++ Compatibility Tool. Warnings appear in a tool window and
   are written to a `migration.log` file in the project directory.

This sample should generate the following warning messages:
```
warning: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
warning: DPCT1043:1: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
warning: DPCT1009:2: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
warning: DPCT1009:3: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
warning: DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
warning: DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
```

See the section titled **Addressing Warnings in Migrated Code** below to
understand how to resolve the warnings.

4. Address any compilation issues and then build and run the migrated project
   by right-clicking the project in the solution explorer, selecting it as the
   startup project, and running it with the green play button in the top bar.


# Addressing Warnings in Migrated Code

Migration generated warnings based on code that DPCT could not migrate:
```
warning: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
warning: DPCT1043:1: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
warning: DPCT1009:2: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
warning: DPCT1009:3: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
warning: DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
warning: DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
```

## Resolve the DPCT1003, DPCT1009, and DPCT1043
First, start with the warnings specific to `needle.dp.cpp` located in the
migrated source directory.

In this case, the original CUDA call `cudaDriverGetVersion` was migrated to an
equivalent SYCL construct. However, because CUDA uses error codes while SYCL
uses exceptions to handle errors, `dpct` added the DPCT1003 message in the
comments to indicate that additional manual edits are likely required. The
same call also resulted in message DPCT1043, which warns that the device
version has a different meaning versus the CUDA driver version. Lastly, the `if`
statement that checks the error code results in DPCT1009.

To manually resolve these issues:
-	since the error codes are not needed, you can remove code related to them.
-	since the SYCL device version api is different, you need to update this code.

Replace the following lines from `needle.dp.cpp`.

```cpp
int version = 0;
int err_code = 999;
/* ...dpct generated comments... */
err_code =
    (version =
        dpct::get_current_device().get_info<sycl::info::device::version>(),
     0);
if (err_code != 0)
/* ...dpct generated comments... */
    printf("Error \\"%s\\" checking driver version: %s.\\n",
    "cudaGetErrorName not supported" /*cudaGetErrorName(err_code)*/,
    "cudaGetErrorString not supported" /*cudaGetErrorString(err_code)*/);
else
    printf("CUDA driver version: %d.%d\n", version/1000, version%1000/10);
```

Replace with:

```cpp
    std::string version = dpct::get_current_device().get_info<sycl::info::device::version>();
    printf("SYCL device version: %s\n", version.c_str());

```

After the code is fixed, save the file in the text editor.


## Resolve DPCT1049
This DPCT message appears twice in `needle.dp.cpp`, and it simply warns that
the maximum workgroup size for your GPU may be different from the CUDA limit.
If you examine the code, we are running workgroup sizes of `BLOCK_SIZE=16`
which is well under the limit of all GPUs. This message can be safely ignored.


## Check Migration of Code using Variables Declared in Preprocessor Directives

In some cases, where variables are declared using preprocessor directives,
`dpct` may replace the code with the value derived from the preprocessor
directive. This occurred in four lines of this migration.

In `needle.dp.cpp`, search for the variables `temp_range_ct1` and
`ref_range_ct1`. Find the lines that include the `/*BLOCK_SIZE*/` comments
inserted by `dpct`, like these:

```cpp
sycl::range<2> temp_range_ct1(17 /*BLOCK_SIZE+1*/, 17 /*BLOCK_SIZE+1*/);
sycl::range<2> ref_range_ct1(16 /*BLOCK_SIZE*/, 16 /*BLOCK_SIZE*/);
```

and change the code to look like the following:

```cpp
sycl::range<2> temp_range_ct1(BLOCK_SIZE+1, BLOCK_SIZE+1);
sycl::range<2> ref_range_ct1(BLOCK_SIZE, BLOCK_SIZE);
```


## Modify the Block Size

Open `needle.h`, also located in the migrated source directory.

The equivalent of a CUDA block is a SYCL workgroup. The maximum size
of a block for a CUDA-enabled device may be different than the maximum size of
a work-group for a SYCL-enabled device. You can use `clinfo` to get information
about the max workgroup size available on your system. Often the block
size needs to be adjusted for functionality and performance.

In this application, you can try changing the BLOCK_SIZE in needle.h.


# Example Output

Once you have correctly completed the migration and provided the new makefile
to the migrated project, you will be able to build and run the
project to get the following output.

```
./needleman_wunsch_dpcpp 4096 16
WG size of kernel = 128
Start Needleman-Wunsch
Processing top-left matrix
Processing bottom-right matrix
```

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).