"""SQLite transactions and hash chained per-agent journal."""
import sqlite3
import hashlib
import json
from .schema import canonical


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS agents(id TEXT PRIMARY KEY, state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS journal(agent TEXT NOT NULL, seq INTEGER NOT NULL,
          operation TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL,
          previous TEXT NOT NULL, hash TEXT NOT NULL, wall_time TEXT NOT NULL,
          PRIMARY KEY(agent,seq), FOREIGN KEY(agent) REFERENCES agents(id));
        ''')

    def read(self, agent):
        row = self.db.execute("SELECT state FROM agents WHERE id=?", (agent,)).fetchone()
        if row is None:
            raise KeyError(agent)
        return json.loads(row[0])

    def entries(self, agent):
        columns = ("seq", "operation", "payload", "state", "previous", "hash", "wall_time")
        rows = self.db.execute("SELECT seq,operation,payload,state,previous,hash,wall_time FROM journal WHERE agent=? ORDER BY seq", (agent,)).fetchall()
        result=[]
        previous=""
        for row in rows:
            item=dict(zip(columns,row))
            for key in ("payload", "state"):
                item[key]=json.loads(item[key])
            check={k:item[k] for k in ("seq", "operation", "payload", "state", "previous")}
            if item["previous"] != previous or digest(check) != item["hash"]:
                raise ValueError("journal integrity failure")
            previous=item["hash"]
            result.append(item)
        return result

    def close(self):
        self.db.close()
