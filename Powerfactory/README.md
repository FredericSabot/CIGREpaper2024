The original .pfd files are in the original_data folder. The 18.09.23 version in this folder is the same, but all H2 and BESS loads have been assigned a type (previously some where defined as H2/BESS and some had no type), and H2 and BESS type loads have been defined as constant current. Also, all zone wind generation scaling factors have been set back to one. Wind availability is now handled in the SCOPF directly.

v3 SF: renamed Slack Load(12) to Slack Load TEAL, rename incorrectly named elements WYHI to WIYH, fixed DER LVRT model (compute min voltage instead of max), add curves for all dynamic equivalents, set qdsl of dynamic equivalent's static load to lead to 0 MVar at the low-voltage side of the transformer instead of the high-voltage side (to be consistent with Dynawo models)
v4 SF: increase transformer rating of AGR NGET to 50GW to represent all NGET sync gens
v5 SF: disable FRT behaviour of HVDC SPIT

Dispatches are generated based on the scenarios below using a security-constrained (AC) optimal power flow (SCOPF). Please refer to the paper for the list of studied scenarios.

Data are in Powerfactory 2022 SP4 format