The original .pfd files are in the original_data folder. The 18.09.23 version in this folder is the same, but all H2 and BESS loads have been assigned a type (previously some where defined as H2/BESS and some had no type), and H2 and BESS type loads have been defined as constant current.

Dispatches are generated based on the scenarios below using a security-constrained (AC) optimal power flow (SCOPF).

| Scenario | Description 1     | Description 2   | Gross load S (GW) | Gross load E&W (GW) | DER share | Wind |
|----------|-------------------|-----------------|-------------------|---------------------|-----------|------|
| 1        | Summer minimum AM | Leading the way | 2597              | 25584               | 0.4       | 0.8  |
| 2        | Winter peak       | Leading the way | 5574              | 57179               | 0.4       | 0.8  |