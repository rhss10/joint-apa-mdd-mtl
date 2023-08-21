import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_from_disk
from scipy.stats import pearsonr

test_ds = load_from_disk("/PATH_TO_DATA/speechocean_test_ds")
test_model = pd.read_csv("./FINETUNED_MODEL_YOU_WANT_TO_ANALYZE/prediction_final.csv")


# correlation analysis for human annotations of Speechcean test set
acc_dict = []
acc_mis_dict = []
flu_dict = []
flu_mis_dict = []
pros_dict = []
pros_mis_dict = []
tot_dict = []
tot_mis_dict = []
tot_metric = evaluate.load("pearsonr")
pros_metric = evaluate.load("pearsonr")
flu_metric = evaluate.load("pearsonr")
acc_metric = evaluate.load("pearsonr")

for i in range(len(test_ds)):
    acc_metric.add(prediction=test_ds[i]["accuracy"], reference=len(test_ds[i]["mispronunciations"]))
    acc_dict.append(test_ds[i]["accuracy"])
    acc_mis_dict.append(len(test_ds[i]["mispronunciations"]))
    flu_metric.add(prediction=test_ds[i]["fluency"], reference=len(test_ds[i]["mispronunciations"]))
    flu_dict.append(test_ds[i]["accuracy"])
    flu_mis_dict.append(len(test_ds[i]["mispronunciations"]))
    pros_metric.add(prediction=test_ds[i]["prosodic"], reference=len(test_ds[i]["mispronunciations"]))
    pros_dict.append(test_ds[i]["accuracy"])
    pros_mis_dict.append(len(test_ds[i]["mispronunciations"]))
    tot_metric.add(prediction=test_ds[i]["total"], reference=len(test_ds[i]["mispronunciations"]))
    tot_dict.append(test_ds[i]["accuracy"])
    tot_mis_dict.append(len(test_ds[i]["mispronunciations"]))

tot_res = tot_metric.compute()
pros_res = pros_metric.compute()
flu_res = flu_metric.compute()
acc_res = acc_metric.compute()
print(tot_res)
print(pros_res)
print(flu_res)
print(acc_res)

test_ds = test_ds.map(lambda x: {"mis_num": len(x["mispronunciations"])})
test_df = test_ds.to_pandas()


# correlation analysis for model prediction of Speechcean test set
tot_model_metric = evaluate.load("pearsonr")
pros_model_metric = evaluate.load("pearsonr")
flu_model_metric = evaluate.load("pearsonr")
acc_model_metric = evaluate.load("pearsonr")
acc_model_metric.add_batch(predictions=test_model["acc_preds"], references=test_model["mispronunciations"])
flu_model_metric.add_batch(predictions=test_model["flu_preds"], references=test_model["mispronunciations"])
pros_model_metric.add_batch(predictions=test_model["pros_preds"], references=test_model["mispronunciations"])
tot_model_metric.add_batch(predictions=test_model["tot_preds"], references=test_model["mispronunciations"])

tot_model_res = tot_model_metric.compute()
pros_model_res = pros_model_metric.compute()
flu_model_res = flu_model_metric.compute()
acc_model_res = acc_model_metric.compute()
print(tot_model_res)
print(pros_model_res)
print(flu_model_res)
print(acc_model_res)


# print p-value
# pearsonr(test_model["mispronunciations"], test_model["acc_preds"])


# plot correlation model
fig, ax = plt.subplots(figsize=(8, 4), ncols=2)
g = sns.regplot(
    x=test_df["mis_num"],
    y=test_df["accuracy"],
    line_kws={"color": "#d73027", "alpha": 1.0, "lw": 2},
    scatter=False,
    ci=None,
    label="accuracy",
    ax=ax[0],
)
g = sns.regplot(
    x=test_df["mis_num"],
    y=test_df["fluency"],
    line_kws={"color": "#91bfdb", "alpha": 1.0, "lw": 2},
    scatter=False,
    ci=None,
    label="fluency",
    ax=ax[0],
)
g = sns.regplot(
    x=test_df["mis_num"],
    y=test_df["prosodic"],
    line_kws={"color": "#4575b4", "alpha": 1.0, "lw": 2},
    scatter=False,
    ci=None,
    label="prosodic",
    ax=ax[0],
)
g = sns.regplot(
    x=test_df["mis_num"], y=test_df["total"], line_kws={"color": "#fee090", "alpha": 1.0, "lw": 2}, scatter=False, ci=None, label="total", ax=ax[0]
)
g.set(ylim=(0, None))
g.set(xlim=(0, None))
g.set(xticks=list(range(0, 30, 5)))
# g.set_title("Correlation Plot for Human Evaulators", fontsize=12, fontweight='bold')
g.set_xlabel("Num. of Human Annotated \n Mispronunciations", fontsize=15)
g.set_ylabel("Proficiency Scores", fontsize=15)
g.legend(loc="upper right", fontsize=12)

h = sns.regplot(
    x=test_model["mispronunciations"],
    y=test_model["acc_preds"],
    line_kws={"color": "#d73027", "lw": 2},
    scatter=False,
    ci=None,
    label="accuracy",
    ax=ax[1],
)
h = sns.regplot(
    x=test_model["mispronunciations"],
    y=test_model["flu_preds"],
    line_kws={"color": "#91bfdb", "lw": 2},
    scatter=False,
    ci=None,
    label="fluency",
    ax=ax[1],
)
h = sns.regplot(
    x=test_model["mispronunciations"],
    y=test_model["pros_preds"],
    line_kws={"color": "#4575b4", "lw": 2},
    scatter=False,
    ci=None,
    label="prosodic",
    ax=ax[1],
)
h = sns.regplot(
    x=test_model["mispronunciations"],
    y=test_model["tot_preds"],
    line_kws={"color": "#fee090", "lw": 2},
    scatter=False,
    ci=None,
    label="total",
    ax=ax[1],
)
h.set(ylim=(0, None))
h.set(xlim=(0, None))
h.set(xticks=list(range(0, 30, 5)))
# h.set_title("Correlation Plot for Model Predictions", fontsize=12, fontweight='bold')
h.set_xlabel("Num. of Model Predicted \n Mispronunciations", fontsize=15)
h.set_ylabel("", fontsize=10)
h.legend(loc="upper right", fontsize=12)

plt.tight_layout()
plt.savefig("corr.pdf")
