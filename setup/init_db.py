"""
Initialize SQLite database with sample Korean corporate financial data.
Run this once: python setup/init_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/financial.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS financials")
    cursor.execute("""
        CREATE TABLE financials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            year INTEGER NOT NULL,
            revenue_billion_krw REAL,
            operating_profit_billion_krw REAL,
            net_profit_billion_krw REAL,
            capex_billion_krw REAL,
            employees INTEGER,
            UNIQUE(company, year)
        )
    """)

    # Sample data: Samsung Electronics, SK Hynix, LG Electronics (2020-2024)
    records = [
        # Samsung Electronics
        ("Samsung Electronics", 2020, 236800, 35994, 26407, 38533, 267937),
        ("Samsung Electronics", 2021, 279600, 51634, 39907, 48176, 270372),
        ("Samsung Electronics", 2022, 302200, 43376, 55404, 53110, 274400),
        ("Samsung Electronics", 2023, 258900, 6567,  15487, 53500, 270000),
        ("Samsung Electronics", 2024, 300900, 32400, 34500, 56000, 267800),
        # SK Hynix
        ("SK Hynix", 2020, 31930, 5008,  2430,  9766,  29490),
        ("SK Hynix", 2021, 42998, 12410, 9627,  14500, 30000),
        ("SK Hynix", 2022, 44649, 7055,  2071,  19000, 30400),
        ("SK Hynix", 2023, 32766, -7733, -7732, 16000, 30300),
        ("SK Hynix", 2024, 66194, 23468, 19793, 18000, 30800),
        # LG Electronics
        ("LG Electronics", 2020, 63262, 2387,  1118,  2600,  75000),
        ("LG Electronics", 2021, 74701, 3899,  1370,  3100,  74000),
        ("LG Electronics", 2022, 83467, 3549,  1200,  3500,  75000),
        ("LG Electronics", 2023, 84227, 3549,  900,   3800,  74500),
        ("LG Electronics", 2024, 87000, 3800,  1100,  4000,  75000),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO financials
        (company, year, revenue_billion_krw, operating_profit_billion_krw,
         net_profit_billion_krw, capex_billion_krw, employees)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
