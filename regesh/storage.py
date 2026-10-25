"""Atomic SQLite state and lossless compact, backward-compatible audit journals."""
import sqlite3
import hashlib
import json
import os
from .schema import canonical
from .native_memory import DeltaKernel, apply_patch


DELTA_MARKER='__regesh_delta_v1__'


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Store:
    def __init__(self,path):
        self.kernel=DeltaKernel(os.environ.get('REGESH_MEMORY_BACKEND','python'))
        self.db=sqlite3.connect(str(path),timeout=30,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS agents(id TEXT PRIMARY KEY, state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS journal(agent TEXT NOT NULL, seq INTEGER NOT NULL,
          operation TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL,
          previous TEXT NOT NULL, hash TEXT NOT NULL, wall_time TEXT NOT NULL,
          PRIMARY KEY(agent,seq), FOREIGN KEY(agent) REFERENCES agents(id));
        ''')

    def read(self,agent):
        row=self.db.execute('SELECT state FROM agents WHERE id=?',(agent,)).fetchone()
        if row is None:raise KeyError(agent)
        return json.loads(row[0])

    def last_hash(self,agent):
        row=self.db.execute('SELECT hash FROM journal WHERE agent=? ORDER BY seq DESC LIMIT 1',(agent,)).fetchone()
        if row is None:raise KeyError(agent)
        return row[0]

    def append(self,agent,operation,payload,state,wall_time):
        """Called within Agent's existing transaction; preserve the v1 logical hash.

        The state column contains a full initial checkpoint, followed by compact
        patches. `agents.state` remains one complete current state. Legacy full
        journals and mixed legacy/compact histories need no destructive migration.
        """
        row=self.db.execute('SELECT seq,hash FROM journal WHERE agent=? ORDER BY seq DESC LIMIT 1',(agent,)).fetchone()
        seq=row[0]+1 if row else 0
        previous=row[1] if row else ''
        state_json=canonical(state)
        payload_json=canonical(payload)
        encoded=state_json
        if row and self.kernel.backend!='snapshot':
            old_json=self.db.execute('SELECT state FROM agents WHERE id=?',(agent,)).fetchone()[0]
            old=None if self.kernel.backend=="rust" else json.loads(old_json)
            patch=self.kernel.diff(old,state,old_json=old_json,new_json=state_json)
            compact=canonical({DELTA_MARKER:patch})
            # A restore may replace almost everything; store whichever is smaller.
            if len(compact)<len(state_json):encoded=compact
        # Compose the exact sorted-key canonical v1 entry without serializing the
        # large state again. No float formatting or hash semantics change.
        entry_json=('{"operation":'+canonical(operation)+',"payload":'+payload_json+
                    ',"previous":'+canonical(previous)+',"seq":'+str(seq)+',"state":'+state_json+'}')
        hash_value=hashlib.sha256(entry_json.encode()).hexdigest()
        self.db.execute('INSERT INTO journal VALUES(?,?,?,?,?,?,?,?)',
                        (agent,seq,operation,payload_json,encoded,previous,hash_value,wall_time))
        self.db.execute('UPDATE agents SET state=? WHERE id=?',(state_json,agent))

    def iter_entries(self,agent):
        """Stream logical full entries, retaining only previous/current states.

        Callers retaining every returned full state intentionally use O(history²)
        space; use this iterator for long streaming audits rather than entries().
        """
        columns=('seq','operation','payload','state','previous','hash','wall_time')
        rows=self.db.execute('SELECT seq,operation,payload,state,previous,hash,wall_time FROM journal WHERE agent=? ORDER BY seq',(agent,))
        previous='';state=None
        for row in rows:
            item=dict(zip(columns,row))
            item['payload']=json.loads(item['payload'])
            encoded=json.loads(item['state'])
            if isinstance(encoded,dict) and set(encoded)=={DELTA_MARKER}:
                if state is None:raise ValueError('journal delta lacks initial checkpoint')
                try:state=apply_patch(state,encoded[DELTA_MARKER])
                except (KeyError,IndexError,TypeError,ValueError) as exc:raise ValueError('invalid journal delta') from exc
            else:state=encoded
            item['state']=state
            check={k:item[k] for k in ('seq','operation','payload','state','previous')}
            if item['previous']!=previous or digest(check)!=item['hash']:
                raise ValueError('journal integrity failure')
            previous=item['hash']
            yield item

    def entries(self,agent):
        return list(self.iter_entries(agent))

    def close(self):
        self.db.close()
