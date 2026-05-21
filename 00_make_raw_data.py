import numpy as np
import pandas as pd
from datetime import datetime, timedelta

OUT_DIR = "data"
SEED = 42

def main(n_clients=20000, months=6):
    rng = np.random.default_rng(SEED)

    # --- Клиенты ---
    clients = pd.DataFrame({
        "client_id": np.arange(1, n_clients + 1),
        "age_group": rng.choice(
            ["18-25", "26-35", "36-45", "46-60", "60+"],
            size=n_clients,
            p=[0.18, 0.28, 0.24, 0.20, 0.10]
        ),
        "region": rng.choice(
            ["СПб", "Москва", "ЦФО", "СЗФО", "ЮФО", "ДФО"],
            size=n_clients,
            p=[0.12, 0.18, 0.28, 0.18, 0.14, 0.10]
        )
    })

    # --- Продукты ---
    products = pd.DataFrame({
        "client_id": clients["client_id"],
        "has_deposit": rng.binomial(1, 0.25, n_clients),
        "has_credit_card": rng.binomial(1, 0.35, n_clients),
        "has_loan": rng.binomial(1, 0.20, n_clients)
    })

    # --- Транзакции ---
    tx_rows = []
    start_date = datetime(2025, 1, 1)

    for m in range(months):
        month_start = start_date + timedelta(days=30 * m)
        n_ops = rng.integers(15, 31, size=n_clients)

        for idx, cid in enumerate(clients["client_id"].values):
            k = int(n_ops[idx])
            dates = [
                month_start + timedelta(days=int(rng.integers(0, 30)),
                                        minutes=int(rng.integers(0, 1440)))
                for _ in range(k)
            ]
            types = rng.choice(
                ["income", "purchase", "cash_withdraw", "utility", "credit_payment"],
                size=k,
                p=[0.08, 0.68, 0.07, 0.12, 0.05]
            )

            for t, dt in zip(types, dates):
                if t == "income":
                    a = float(rng.normal(80000, 30000))
                    a = max(15000, min(a, 250000))
                    cat, ch = "income", "inbank"
                elif t == "purchase":
                    a = float(rng.gamma(2.0, 1200))
                    cat = rng.choice(
                        ["supermarket", "transport", "fun", "health", "travel", "other"],
                        p=[0.35, 0.15, 0.20, 0.10, 0.10, 0.10]
                    )
                    ch = rng.choice(["pos", "online"], p=[0.65, 0.35])
                elif t == "cash_withdraw":
                    a = float(rng.choice([1000, 2000, 5000, 10000, 15000]))
                    cat, ch = "cash", "atm"
                elif t == "utility":
                    a = float(rng.normal(2500, 1000))
                    a = max(200, min(a, 15000))
                    cat = "utility"
                    ch = rng.choice(["app", "web"], p=[0.7, 0.3])
                else:
                    a = float(rng.normal(12000, 5000))
                    a = max(1000, min(a, 60000))
                    cat, ch = "credit", "autopay"

                tx_rows.append((cid, dt, t, round(a, 2), cat, ch))

    tx = pd.DataFrame(tx_rows, columns=["client_id", "dt", "op_type", "amount", "category", "channel"])

    # --- События приложения ---
    ev_rows = []
    for cid in clients["client_id"].values:
        weekly_sessions = int(rng.poisson(4))
        weekly_sessions = max(0, min(weekly_sessions, 20))

        for _ in range(12 * weekly_sessions):
            dt = start_date + timedelta(days=int(rng.integers(0, 180)))
            ev_rows.append((cid, dt, "app_open", "main"))
            if rng.random() < 0.35: ev_rows.append((cid, dt, "view", "payments"))
            if rng.random() < 0.25: ev_rows.append((cid, dt, "view", "transfers"))
            if rng.random() < 0.18: ev_rows.append((cid, dt, "view", "deposits"))
            if rng.random() < 0.10: ev_rows.append((cid, dt, "view", "investments"))
            if rng.random() < 0.12: ev_rows.append((cid, dt, "banner_click", "promo"))

    events = pd.DataFrame(ev_rows, columns=["client_id", "dt", "event_type", "screen"])

    # --- Обращения ---
    tk_rows = []
    for cid in clients["client_id"].values:
        t = int(rng.poisson(0.8))
        for _ in range(t):
            dt = start_date + timedelta(days=int(rng.integers(0, 180)))
            channel = rng.choice(["call_center", "chat", "branch"], p=[0.55, 0.35, 0.10])
            topic = rng.choice(["tariffs", "tech", "complaint", "data_change", "other"],
                               p=[0.30, 0.25, 0.15, 0.10, 0.20])
            tk_rows.append((cid, dt, channel, topic, "closed"))

    tickets = pd.DataFrame(tk_rows, columns=["client_id", "dt", "channel", "topic", "status"])

    # --- Сохранение ---
    clients.to_csv(f"{OUT_DIR}/clients.csv", index=False)
    products.to_csv(f"{OUT_DIR}/products.csv", index=False)
    tx.to_csv(f"{OUT_DIR}/transactions.csv", index=False)
    events.to_csv(f"{OUT_DIR}/events.csv", index=False)
    tickets.to_csv(f"{OUT_DIR}/tickets.csv", index=False)

    print("Saved raw data to data/*.csv")

if __name__ == "__main__":
    main()
