## Results

### Mean Euclid Distance

We evaluate MED error on 3 examples. We use a Linux server with Intel(R) Core(TM) i7-10700, 64-bit CPU $@ 2.90$GHz, and RAM 16GB.  We sample $1000$ preference vectors and evaluate $30$ random vectors follow-up time of 10 executions.

|    Method     |     Example 1       | Example 2 | Example 3	|
|:-------------:|:-----------------------:|:-----------------------------------:|:----------------------------------:|
PHN-LS  | $0.0033 \pm  0.0006$ | $\bf 0.0026\pm  \bf 0.0001$ | $0.0278\pm  0.0019$|
PHN-Cheby | $\underline{0.0032}\pm  \underline{0.0003}$  |  $0.0047\pm  0.0008$ | $0.0309\pm  0.0023$|
PHN-Utility ($ub = 2.01$) | $0.0052\pm  0.0004$  |  $0.0062\pm  0.0002$ |  $\bf 0.0138\pm \bf 0.0009$ |
PHN-KL   | $0.0052\pm  0.0003$  | $0.0124\pm  0.0002$ | $0.0373\pm  0.0039$|
PHN-Cauchy | $0.0037\pm  0.0003$ | $0.0232\pm  0.0007$ |  $0.0642\pm  0.0025$|
PHN-Cosine | $0.0051\pm  0.0007$ | $0.0259\pm  0.0017$ |  $0.0396\pm  0.0028$|
PHN-Log | $0.0321\pm  0.0343$ | $\underline{0.0031}\pm  \underline{0.0002}$ |  $0.0225\pm  0.0035$|
PHN-Prod | $0.0432\pm  0.0412$ | $0.0084\pm  0.0008$ |  $0.0385\pm  0.0042$|
PHN-AC ($\rho = 0.0001$) | $0.0052\pm  0.0005$ | $0.0091\pm  0.0014$ |  $\underline{0.0222}\pm  \underline{0.0039}$|
PHN-MC ($\rho = 0.0001$) | $0.0078\pm  0.0008$ | $0.0136\pm  0.0008$ |  $0.0267\pm  0.0044$|
PHN-HV ($\rho = 100$) | $\bf 0.0017\pm \bf 0.0002$ | $0.0092\pm  0.0011$ |  $0.0303\pm  0.0087$|
EPO | $0.0058\pm  0.0002$ | $0.0091\pm  0.0009$ |  $0.0463\pm  0.0053$|
