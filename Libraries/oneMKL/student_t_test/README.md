# `Student's T-test` Sample
The Student's T-test sample shows how to use the Intel® oneAPI Math Kernel Library (oneMKL) Vector Statistics functionality to decide if the null hypothesis should be accepted or rejected.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use oneMKL Vector Statistics
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

Student’s t-test, in statistics, a method of testing hypotheses about the mean of a small sample drawn from a normally distributed population when the population standard deviation is unknown. It is usually first to formulate a null hypothesis, which states that there is no effective difference between the observed sample mean and the hypothesized or stated population mean—i.e., that any measured difference is due only to chance.

This sample uses the oneMKL Vector Statistics functionality to produce the random numbers and compute statistics.

The computations of the Student's t-test sample are performed on the default SYCL* device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

The student's t-test sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and generate the random numbers themselves. After the numbers are produced, basic statistical properties such as mean and standard deviation are computed to be processed inside the Student's T-test algorithm.

## Building the Student's T-test Sample

### Using Visual Studio Code*  (Optional)

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


### On a Linux* System
Run `make` to build and run the sample. Two programs (t_test and t_test_usm) are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. Two programs (t_test.exe and t_test_usm.exe) are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `nmake clean`.

> **Warning**: On Windows, static linking with oneMKL currently takes a very long time, due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Student's T-test Sample
### Example of Output
If everything is working correctly, after running `make` (`nmake`) you will see step-by-step output from each of the two example programs, providing the decision about accepting null hypothesis.
```
./t_test

Student's T-test Simulation
Buffer Api
-------------------------------------
Number of random samples = 1000000 with mean = 0, std_dev = 1
T-test result with expected mean: 1
T-test result with two input arrays: 1

TEST PASSED

./t_test_usm

Student's T-test Simulation
Unified Shared Memory Api
-------------------------------------
Number of random samples = 1000000 with mean = 0, std_dev = 1
T-test result with expected mean: 1
T-test result with two input arrays: 1

TEST PASSED
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
