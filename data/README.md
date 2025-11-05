# SECOM Dataset

Semiconductor manufacturing data from UCI/Kaggle. Each row is a wafer with 590 sensor readings and a pass/fail label.

- **Source:** https://www.kaggle.com/datasets/paresh2047/uci-semcom
- **Samples:** 1567
- **Features:** 590 (numbered 0-589, all numeric sensor measurements)
- **Labels:** -1 = pass, 1 = fail (in the `Pass/Fail` column)
- **Class balance:** ~93% pass, ~7% fail (heavily imbalanced)
- **Missing values:** yes, a lot of them

## Files

After downloading you should have:
- `uci-secom.csv` - single CSV with Time column, 590 feature columns, and Pass/Fail label

Run `bash download.sh` or grab it manually from the Kaggle link above.
