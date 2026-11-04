# Arm A-M inputs: monthly-resolution bag (Arthritis (any subtype))

_The information-equivalence control. Claims are month granular, so the only thing the sequence arms hold that the order-free bag lacks is which of the 12 observation months an event fell in. Arm A-M gives the bag exactly that and nothing else._

- **Columns:** 9,672 (796 code slots + 8 utilisation counts, each x 12 months, plus 24 unbinned).
- **Rows:** 1,251,833. **Non-zeros:** 98,291,407 (**0.81%** dense).
- Dense storage would be 24 GB, so the matrix is stored as CSR.

**Not binned** (so that binning is the only difference from Arm A): `age_at_index`, `is_female`, `is_male`, `reg_mw`, `reg_ne`, `reg_s`, `reg_w`, `cm_htn`, `cm_lipid`, `cm_diabetes`, `cm_obesity`, `cm_depr_anx`, `cm_ckd`, `cm_ihd`, `cm_osteoporosis`, `cm_backpain`, `cm_jointpain`, `cm_soft_msk`, `cm_tobacco`, `t_claims_recent3`, `t_claims_prior9`, `t_months_since_last`, `t_active_recent3`, `t_accel`

Same frozen vocabulary, same treatment exclusions, same cohort and test set as Arm A.