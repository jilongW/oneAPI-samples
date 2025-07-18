# `Matrix Multiply` Sample
A sample containing multiple implementations of matrix multiplication code
sample and is implemented using SYCL* for CPU and GPU.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br> Intel&reg; Advisor
| What you will learn               | How to profile an application using Intel&reg; Advisor
| Time to complete                  | 15 minutes

## Purpose

The Matrix Multiplication sample performs basic matrix multiplication. Three
versions are provided that use different SYCL features.

## Key Implementation details

The basic SYCL implementation explained in the code includes device selector,
buffer, accessor, kernel, and command groups. 

## Include Files 
The include folder is located at
`%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## How to Build
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

This sample contains three versions of matrix multiplication:

- `multiply1` – basic implementation of matrix multiply using SYCL
- `multiply1_1` – basic implementation that replaces the buffer store with a local accessor “acc” to reduce memory traffic
- `multiply1_2` – the basic implementation, plus adding the local accessor and matrix tiling

Edit the line in `src/multiply.hpp` to select the version of the multiply function:
`#define MULTIPLY multiply1`.


### On Linux*
1. Change to the sample directory.
2. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
3. Run the program:
   ```
   ./matrix_multiply
   ```
4. Clean the program using:
   ```
    make clean
   ```

If an error occurs, you can get more details by running `make` with `VERBOSE=1`:
```
make VERBOSE=1
```
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### On Windows*
**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
2. Right-click on the project in **Solution Explorer** and select **Rebuild**.

**Using MSBuild**

1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.

2. Change to the sample directory.

3. Run the following command:
   ```
   MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Release"
   ```

   or

   ```
   MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Debug"
   ```
4. Navigate to the Release/Debug folder (example: x64/Release)
5. Run the program:
   ```
   matrix_multiply.exe
   ```

### Example of Output
```
Address of buf1 = 0000020CBE24B040
Offset of buf1 = 0000020CBE24B180
Address of buf2 = 0000020CBEA5E040
Offset of buf2 = 0000020CBEA5E1C0
Address of buf3 = 0000020CBF26C040
Offset of buf3 = 0000020CBF26C100
Address of buf4 = 0000020CBFA71040
Offset of buf4 = 0000020CBFA71140
Using multiply kernel: multiply1

Running on Intel(R) Iris(R) Xe Graphics

Elapsed Time: 0.978114s
```

## Running an Intel® Advisor analysis
See the [Intel® Advisor Cookbook](https://software.intel.com/en-us/advisor-cookbook).

### Running the Matrix Multiply Advisor sample in the DevCloud<a name="run-matmul-advisor-on-devcloud"></a>
This sample contains 3 version of matrix multiplication:

- `multiply1` – basic implementation of matrix multiply using SYCL
- `multiply1_1` – basic implementation that replaces the buffer store with a local accessor “acc” to reduce memory traffic
- `multiply1_2` – the basic implementation, plus adding the local accessor and matrix tiling

Edit the line in `src/multiply.hpp` to select the version of the multiply function:
`#define MULTIPLY multiply1`.

1.  Open a terminal on your Linux system.
2.	Log in to DevCloud.
```
ssh devcloud
```
3.	Download the samples.
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

4. Change directories to the Matrix Multiply Advisor sample directory.
```
cd ~/oneAPI-samples/Tools/Advisor/matrix_multiply_advisor
```
#### Build and run the sample in batch mode
The following describes the process of submitting build and run jobs to PBS.
A job is a script that is submitted to PBS through the qsub utility. By default, the qsub utility does not inherit the current environment variables or your current working directory. For this reason, it is necessary to submit jobs as scripts that handle the setup of the environment variables. In order to address the working directory issue, you can either use absolute paths or pass the -d \<dir\> option to qsub to set the working directory.

#### Create the Job Scripts
1.	Create a build.sh script with your preferred text editor:
```
nano build.sh
```
2.	 Add this text into the build.sh file:
```
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
mkdir build
cd build
cmake ..
make
```

3.	Save and close the build.sh file.

4.	Create a run.sh script with your preferred text editor:
```
nano run.sh
```

5.	 Add this text into the run.sh file:
```
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
cd build
make run
```
6.	Save and close the run.sh file.

#### Build and run
Jobs submitted in batch mode are placed in a queue waiting for the necessary resources (compute nodes) to become available. The jobs will be executed on a first come basis on the first available node(s) having the requested property or label.
1.	Build the sample on a gpu node.

```
qsub -l nodes=1:gpu:ppn=2 -d . build.sh
```

Note: -l nodes=1:gpu:ppn=2 (lower case L) is used to assign one full GPU node to the job.
Note: The -d . is used to configure the current folder as the working directory for the task.

2.	In order to inspect the job progress, use the qstat utility.
```
watch -n 1 qstat -n -1
```
Note: The watch -n 1 command is used to run qstat -n -1 and display its results every second. If no results are displayed, the job has completed.

3.	After the build job completes successfully, run the sample on a gpu node:
```
qsub -l nodes=1:gpu:ppn=2 -d . run.sh
```
4.	When a job terminates, a couple of files are written to the disk:

    <script_name>.sh.eXXXX, which is the job stderr

    <script_name>.sh.oXXXX, which is the job stdout

    Here XXXX is the job ID, which gets printed to the screen after each qsub command.

5.	Inspect the output of the sample.
```
cat run.sh.oXXXX
```
You should see output similar to this:

```
Scanning dependencies of target run
Address of buf1 = 0x7f570456f010
Offset of buf1 = 0x7f570456f180
Address of buf2 = 0x7f5703d6e010
Offset of buf2 = 0x7f5703d6e1c0
Address of buf3 = 0x7f570356d010
Offset of buf3 = 0x7f570356d100
Address of buf4 = 0x7f5702d6c010
Offset of buf4 = 0x7f5702d6c140
Using multiply kernel: multiply1
Running on Intel(R) UHD Graphics P630 [0x3e96]
Elapsed Time: 1.79388s
Built target run
```

6.	Remove the stdout and stderr files and clean up the project files.
```
rm build.sh.*; rm run.sh.*; make clean
```
7.	Disconnect from the Intel DevCloud.
```
exit
```
## Running an Intel&reg; Advisor analysis
See the [Intel® Advisor Cookbook](https://www.intel.com/content/www/us/en/docs/advisor/cookbook/).

### Build and run additional samples
Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion to this sample. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.

## License
Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](third-party-programs.txt).
