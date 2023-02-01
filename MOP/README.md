## Results

### Mean Euclid Distance

We evaluate MED error on 3 examples. We use a Linux server with Intel(R) Core(TM) i7-10700, 64-bit CPU $@ 2.90$GHz, and RAM 16GB.  We sample $1000$ preference vectors and evaluate $50$ random vectors follow-up time of 30 executions.

|    Method     |     Example 1       | Example 2 | Example 3	|
|:-------------:|:-----------------------:|:-----------------------------------:|:----------------------------------:|
PHN-EPO | $0.0088\pm  0.0006$ | $0.0017\pm  0.0002$ |  $0.0808\pm  0.0052$|
PHN-LS  | $0.0042\pm  0.0008$ | $\bf 0.0011\pm  0.0001$ | $0.0494\pm  0.0043$|
PHN-Cheby | $0.0084\pm  0.0005$  |  $0.0019\pm  0.0002$ | $0.0395\pm  0.0036$|
PHN-Utility ($ub = 2.01$) | $0.0025\pm  0.0003$  |  $0.0049\pm  0.0003$ |  $\bf 0.0201\pm  0.0022$ |
PHN-KL   | $0.0052\pm  0.0003$  | $0.0124\pm  0.0002$ | $0.0373\pm  0.0039$|
PHN-Cauchy | $0.0037\pm  0.0003$ | $0.0232\pm  0.0007$ |  $0.0642\pm  0.0025$|
PHN-Cosine | $0.0051\pm  0.0007$ | $0.0259\pm  0.0017$ |  $0.0396\pm  0.0028$|
PHN-Log | $0.0321\pm  0.0343$ | $0.0031\pm  0.0002$ |  $0.0225\pm  0.0035$|
PHN-Prod | $0.0432\pm  0.0412$ | $0.0084\pm  0.0008$ |  $0.0385\pm  0.0042$|
PHN-AC ($\rho = 0.0001$) | $0.0052\pm  0.0005$ | $0.0091\pm  0.0014$ |  $0.0222\pm  0.0039$|
PHN-MC ($\rho = 0.0001$) | $0.0078\pm  0.0008$ | $0.0136\pm  0.0008$ |  $0.0267\pm  0.0044$|
PHN-HV ($\rho = 100$) | $\bf 0.0017\pm \bf 0.0002$ | $0.0092\pm  0.0011$ |  $0.0303\pm  0.0087$|
