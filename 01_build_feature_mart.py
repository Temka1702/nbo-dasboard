import numpy as np
import pandas as pd

OUT_DIR = "data"
SEED = 42

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    rng = np.random.default_rng(SEED)

    clients = pd.read_csv(f"{OUT_DIR}/clients.csv")
    products = pd.read_csv(f"{OUT_DIR}/products.csv")
    tx = pd.read_csv(f"{OUT_DIR}/transactions.csv", parse_dates=["dt"])
    events = pd.read_csv(f"{OUT_DIR}/events.csv", parse_dates=["dt"])
    tickets = pd.read_csv(f"{OUT_DIR}/tickets.csv", parse_dates=["dt"])

    # --- TX агрегаты ---
    tx["is_income"] = (tx["op_type"] == "income").astype(int)
    tx["income_amt"] = tx["amount"] * tx["is_income"]
    tx["spend_amt"] = tx["amount"] * (1 - tx["is_income"])

    tx_agg = tx.groupby("client_id", as_index=False).agg(
        income_sum=("income_amt", "sum"),
        spend_sum=("spend_amt", "sum")
    )
    tx_agg["income_avg"] = tx_agg["income_sum"] / 6.0
    tx_agg["spend_avg"] = tx_agg["spend_sum"] / 6.0
    tx_agg["balance_avg"] = (tx_agg["income_avg"] - tx_agg["spend_avg"]).clip(lower=0)

    spend = tx[tx["is_income"] == 0].copy()
    spend_total = spend.groupby("client_id")["amount"].sum().rename("spend_total")
    spend_cat = spend.pivot_table(index="client_id", columns="category", values="amount", aggfunc="sum").fillna(0)
    spend_cat = spend_cat.join(spend_total, how="left").fillna(0)

    tx_agg = tx_agg.set_index("client_id")
    tx_agg["cash_share"] = (spend_cat.get("cash", 0) / spend_cat["spend_total"]).replace([np.inf, -np.inf], 0).fillna(0)
    tx_agg["travel_share"] = (spend_cat.get("travel", 0) / spend_cat["spend_total"]).replace([np.inf, -np.inf], 0).fillna(0)
    tx_agg = tx_agg.reset_index()

    # --- APP агрегаты ---
    ev_agg = events.groupby("client_id", as_index=False).agg(
        sessions_week=("event_type", lambda s: (s == "app_open").sum() / 12.0),
        invest_views_3m=("screen", lambda s: (s == "investments").sum()),
        banner_clicks=("event_type", lambda s: (s == "banner_click").sum()),
        views=("event_type", lambda s: (s == "view").sum())
    )
    ev_agg["banner_ctr"] = (ev_agg["banner_clicks"] / ev_agg["views"]).replace([np.inf, -np.inf], 0).fillna(0)

    # --- Tickets агрегаты ---
    tk_agg = tickets.groupby("client_id", as_index=False).agg(
        tickets_6m=("dt", "count"),
        complaints_6m=("topic", lambda s: (s == "complaint").sum()),
        last_ticket_dt=("dt", "max")
    )
    max_dt = tickets["dt"].max() if len(tickets) else pd.Timestamp("2025-06-30")
    tk_agg["days_since_last_ticket"] = (pd.to_datetime(max_dt) - pd.to_datetime(tk_agg["last_ticket_dt"])).dt.days
    tk_agg = tk_agg.drop(columns=["last_ticket_dt"])

    # --- Merge ---
    df = (clients
          .merge(products, on="client_id", how="left")
          .merge(tx_agg[["client_id","income_avg","spend_avg","balance_avg","cash_share","travel_share"]],
                 on="client_id", how="left")
          .merge(ev_agg[["client_id","sessions_week","invest_views_3m","banner_ctr"]],
                 on="client_id", how="left")
          .merge(tk_agg[["client_id","tickets_6m","complaints_6m","days_since_last_ticket"]],
                 on="client_id", how="left"))

    for c in df.columns:
        if df[c].dtype.kind in "fi":
            df[c] = df[c].fillna(0)

    # --- target (отклик на вклад) ---
    z = (
        0.00002 * df["balance_avg"] +
        0.06 * df["invest_views_3m"] +
        0.05 * df["sessions_week"] -
        0.7 * df["complaints_6m"] -
        2.0 * df["has_deposit"] -
        3.0
    )
    p = sigmoid(z)
    df["opened_deposit_30d"] = rng.binomial(1, p)

    df.to_csv(f"{OUT_DIR}/feature_mart.csv", index=False)
    print("Saved feature mart to data/feature_mart.csv")

if __name__ == "__main__":
    main()
