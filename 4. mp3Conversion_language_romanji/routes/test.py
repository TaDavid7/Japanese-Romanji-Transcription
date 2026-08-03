from jamdict import Jamdict
jam = Jamdict()
import pandas as pd
print(jam.lookup("行く").entries[0])

df = pd.read_csv("jsut_results.csv")
diff = df[df["kanji"] != df["hybrid"]]
for _, r in diff.head(15).iterrows():
    print(r["clip"])
    print("  ref :", r["reference"])
    print("  knj :", r["kanji"])
    print("  hyb :", r["hybrid"])