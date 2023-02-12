# IFRS17CalculationEngine Python translation

This repository contains a translation of
[IFRS 17 Calculation Engine][1]
into Python.
[The original program][1] is written in C# and made open-source under the MIT license
by a software company called Systemorph. 

[1]: https://github.com/Systemorph/IFRS17CalculationEngine

The Python translation in this repository aims to be executable,
so that it helps understand the logic of the IFRS17 engine with the help
of debugging features offered by IDEs, by examining how the data is
transformed and processed.

Since the purpose of the translation is logic examination,
it is not suitable for any production uses. 
In addition, the Python translation is much more primitive than the original 
because of the following reasons.

* All constructs for asynchronous communication in the original code 
  are ignored. All the asynchronous calls are treated as static function calls in the translation.

* The original code depends on proprietary libraries that are not open-sourced.
  Mock classes, such as `IScope` and `IQuerySource` are defined to mimic the behaviour of the interfaces
  provided by the proprietary libraries.

* The Python translation aims to reproduce the figures presented
  in Systemorph's [tutorial videos](https://www.youtube.com/@systemorph/videos), 
  but it may fail to do so as the data used in the videos may not be 
  the same as the data provided in the repository.

* Cotents of the ifrs17/Report directory in the original repository are not translated
  as the IFRS17 calculations are carried out at the import step. 
  Alternatively, `ReportScopes.ipynb` depicts how reporting variables are filtered
  to output each balance sheet item. PnLMapping.xlsx depicts how reporting variables
  are mapped to the PnL variables.

* The Python translation is written to read all input data from Excel files,
  so all the CSV files in the original repository are converted to Excel files.

* The translation is based on version 1.0.0 of the original code,
  but some parts of it reflects some commits after
  v1.0.0 by accident.

