# Node Advisor — draft di migrazione schema

**Stato**: implementato in Fase 1 (PR LP-02/LP-03) esattamente come
descritto qui, incluso lo scostamento di `tree_heads` dalla bozza iniziale
(vedi nota sotto).
**Riferimento**: `docs/node-advisor-operational-plan.md` §5.

## Tabelle nuove (additive, `CREATE TABLE IF NOT EXISTS`)

Tutte vivono nello stesso file SQLite di `trees`/`runs`/`run_artifacts`
(`lazyportfolio.v2.db`), con `PRAGMA foreign_keys=ON`, WAL e busy timeout
abilitati su ogni connessione (`db.connect()`).

```
trees               (esistente, invariata: name TEXT PRIMARY KEY, config, created_at, updated_at)

tree_revisions      (revision_id UUID PK, tree_id UUID, parent_revision_id UUID NULL,
                      config_json TEXT, config_hash TEXT, created_at TEXT,
                      actor_type TEXT, actor_id TEXT, reason TEXT NULL)

tree_heads          (tree_id UUID PK, head_revision_id UUID -> tree_revisions.revision_id)
                      -- tabella nuova invece di una colonna aggiunta a `trees`: `trees`
                      -- resta byte-per-byte invariata, zero revisione richiesta ai
                      -- chiamanti esistenti di list_saved_models/read_model/write_model.

legacy_tree_names   (tree_id UUID, name TEXT UNIQUE)  -- mapping riga-legacy -> tree_id

agent_conversations (conversation_id UUID PK, tree_id UUID, node_id TEXT,
                      user_id TEXT, created_at TEXT, updated_at TEXT)

agent_messages      (message_id UUID PK, conversation_id UUID, role TEXT,
                      content_json TEXT, revision_id UUID NULL,
                      data_fingerprint TEXT NULL, created_at TEXT)

agent_jobs          (job_id UUID PK, conversation_id UUID, request_message_id UUID,
                      kind TEXT, status TEXT, checkpoint_key TEXT,
                      session_db_path TEXT NULL, budget_json TEXT,
                      started_at TEXT NULL, heartbeat_at TEXT NULL,
                      finished_at TEXT NULL, error_json TEXT NULL)

change_proposals    (proposal_id UUID PK, batch_id UUID NULL,
                      supersedes_proposal_id UUID NULL, tree_id UUID,
                      base_revision_id UUID, node_id TEXT,
                      kind TEXT, producer_kind TEXT, producer_id TEXT,
                      payload_json TEXT, content_hash TEXT, status TEXT,
                      expires_at TEXT, created_at TEXT)

proposal_approvals  (approval_id UUID PK, proposal_id UUID UNIQUE,
                      approved_by TEXT, approved_at TEXT, approved_hash TEXT,
                      idempotency_key TEXT UNIQUE, applied_revision_id UUID NULL,
                      result_json TEXT NULL)

proposal_evidence   (proposal_id UUID, evidence_id UUID, metadata_json TEXT,
                      excerpt TEXT, content_hash TEXT NULL)

outbox_events       (event_id UUID PK, aggregate_type TEXT, aggregate_id UUID,
                      event_type TEXT, payload_json TEXT, created_at TEXT,
                      delivered_at TEXT NULL)
```

Note sulle colonne denormalizzate su `change_proposals` (`kind`,
`producer_kind`, `producer_id`, oltre a `batch_id`): duplicano campi già
presenti in `payload_json` (il `ChangeProposal` serializzato) per permettere
filtri/indici SQL diretti (es. "tutte le proposte pending del committee")
senza deserializzare JSON riga per riga. `payload_json` resta la fonte di
verità; le colonne denormalizzate sono una cache di lettura scritta nella
stessa transazione, mai aggiornate indipendentemente.

Indici minimi: `idx_change_proposals_tree_status (tree_id, status)`,
`idx_change_proposals_batch (batch_id)` (nullable, per raggruppare le run
del committee), `idx_agent_messages_conversation (conversation_id,
created_at)`, `idx_outbox_undelivered (delivered_at)` parziale su
`delivered_at IS NULL` se il dialetto SQLite in uso lo supporta.

## Vincoli essenziali (§5.1)

- unique su `(tree_id, name)` o, in locale, su `name` mantenendo `tree_id`
  stabile;
- unique su `proposal_approvals.proposal_id` e su `idempotency_key`;
- foreign key `tree_revisions.tree_id -> trees` (via `legacy_tree_names` per
  compatibilità nome→id durante la transizione), `change_proposals.tree_id`,
  `agent_conversations.tree_id`;
- `tree_heads.head_revision_id` referenzia una revision dello stesso tree
  (foreign key verso `tree_revisions.revision_id`); che sia la revision del
  *proprio* `tree_id` e non di un altro è verificato dal repository
  (`save_revision`'s CAS legge la head corrente per quello specifico
  `tree_id` prima di scrivere), non dalla sola foreign key — SQLite non
  supporta un check cross-column così espressivo.

## Passi di migrazione (§5.2)

1. Creare le nuove tabelle senza modificare `trees` esistente (additivo, `IF
   NOT EXISTS`).
2. Migrare ogni riga legacy di `trees` in: un `tree_id` nuovo (`uuid4()`),
   una `tree_revisions` iniziale con quel `config`, una riga
   `legacy_tree_names(tree_id, name)`. Idempotente: un `name` già presente in
   `legacy_tree_names` viene saltato, non ri-migrato.
3. Aggiornare `list_saved_models`/`read_model` (oggi in
   `lazyportfolio.v2.store`) per leggere dalla head revision tramite
   `legacy_tree_names`, preservando il contratto esterno keyed by name — i
   chiamanti esistenti (Tree Studio, LazyTools' `PortfolioTreeTools`) non
   cambiano firma.
4. Aggiornare `write_model` perché ogni salvataggio umano crei una nuova
   `tree_revisions` invece di un `UPDATE` in-place su `trees.config`, con
   `expected_head` opzionale per un CAS best-effort anche dal path legacy.
5. Mantenere per una release la compatibilità di lettura con un database non
   ancora migrato (chiamare la migrazione lazily al primo accesso, non
   richiedere un passo manuale separato).
6. Solo dopo test di round-trip byte-equivalente e backup documentato,
   valutare se rimuovere il payload legacy da `trees.config` (probabilmente
   mai in questo progetto: `trees` resta l'indice nome→tree_id, non viene
   svuotata).

**Non fare dual-write prolungato** tra `trees.config` e `tree_revisions`:
dopo il passo 4, `trees.config` smette di essere scritto direttamente da
`write_model` — resta leggibile per compatibilità finché non si verifica che
nessun chiamante lo legge più direttamente, ma la fonte di verità per un
salvataggio umano è sempre e solo `tree_revisions` da quel punto in poi.

## Backup

Prima di ogni passo che tocca dati esistenti (2, 4): copia del file
`reports/tree_studio/tree_studio.sqlite3` (path risolto da
`lazyportfolio.v2.db.resolve_db_path`) con timestamp nel nome, conservata
finché il passo successivo non è verificato in produzione locale.

## Rollback

Nessuna delle tabelle nuove viene letta da `list_saved_models`/`read_model`/
`write_model` esistenti finché non è esplicitamente cablata nel passo 3/4
sopra — quindi il rollback dei passi 1-2 è **drop delle tabelle nuove**,
mai tocca `trees`/`runs`/`run_artifacts`. Il rollback dei passi 3-4 (dopo
che `write_model` scrive `tree_revisions` come fonte di verità) richiede
invece un passo esplicito di ri-sincronizzazione `trees.config` dalla head
revision prima del drop, per non perdere i salvataggi fatti nel frattempo
solo su `tree_revisions`.
