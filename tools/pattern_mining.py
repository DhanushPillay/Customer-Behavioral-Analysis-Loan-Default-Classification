"""Mine pre-loan account behavior with Apriori and FP-Growth."""

from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def build_baskets() -> pd.DataFrame:
    loans = pd.read_csv(DATA_DIR / "loan.csv", sep=";")
    transactions = pd.read_csv(DATA_DIR / "trans.csv", sep=";")
    loans["date"] = pd.to_datetime(loans["date"], format="%y%m%d")
    transactions["date"] = pd.to_datetime(transactions["date"], format="%y%m%d")
    first_loan = loans.groupby("account_id", as_index=False)["date"].min()
    transactions = transactions.merge(first_loan, on="account_id", suffixes=("", "_loan"))
    transactions = transactions[transactions["date"] < transactions["date_loan"]].copy()
    transactions["item"] = "type_" + transactions["type"].astype(str)
    transactions.loc[transactions["amount"] > transactions["amount"].quantile(0.75), "item"] = "high_amount"
    transactions.loc[transactions["balance"] < 0, "item"] = "negative_balance"
    baskets = pd.crosstab(transactions["account_id"], transactions["item"]).astype(bool)
    default_accounts = loans.loc[loans["status"].isin(["B", "D"]), "account_id"]
    baskets["default_before_loan"] = baskets.index.isin(default_accounts)
    return baskets


def mine(method_name, miner, baskets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    itemsets = miner(baskets, min_support=0.05, use_colnames=True, max_len=3)
    itemsets["method"] = method_name
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.4)
    rules = rules[rules["consequents"].map(lambda values: "default_before_loan" in values)].copy()
    rules["method"] = method_name
    return itemsets, rules


def main() -> None:
    baskets = build_baskets()
    apriori_sets, apriori_rules = mine("apriori", apriori, baskets)
    fp_sets, fp_rules = mine("fpgrowth", fpgrowth, baskets)
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.concat([apriori_sets, fp_sets], ignore_index=True).to_csv(
        OUTPUT_DIR / "pattern_itemsets.csv", index=False
    )
    pd.concat([apriori_rules, fp_rules], ignore_index=True).to_csv(
        OUTPUT_DIR / "pattern_rules.csv", index=False
    )
    print(f"Apriori itemsets: {len(apriori_sets)}, FP-Growth itemsets: {len(fp_sets)}")
    print(f"Default-risk rules: {len(apriori_rules)} Apriori, {len(fp_rules)} FP-Growth")


if __name__ == "__main__":
    main()