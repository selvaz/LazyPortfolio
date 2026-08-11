# Node Advisor — runbook operativo

**Riferimento:** `docs/node-advisor-operational-plan.md` (piano finalizzato), `docs/adr/0001-node-advisor-architecture.md`.
**Ambito:** operare e recuperare il Node Advisor in locale (Tree Studio, single-user). Non copre deployment multiutente/RBAC (fuori scope, §15 del piano).

## 1. Avviare il servizio

```bash
python project/tree_studio.py 8768
```

oppure tramite `.claude/launch.json`, voce `lazyportfolio-tree-studio` (porta 8768). Il worker del Node Advisor (`_start_advisor_worker`) parte automaticamente in un thread daemon dentro `main()` -- non serve un processo separato.

Il database condiviso è un unico file SQLite, risolto da `lazyportfolio.v2.db.resolve_db_path`:

1. argomento esplicito (se il chiamante lo passa);
2. env var `LAZYPORTFOLIO_TREE_DB`;
3. default: `<repo>/reports/tree_studio/tree_studio.sqlite3`.

Le tabelle del Node Advisor (`tree_revisions`, `tree_heads`, `agent_conversations`, `agent_messages`, `agent_jobs`, `change_proposals`, `proposal_approvals`, `proposal_evidence`, `outbox_events`, `legacy_tree_names`) sono create additivamente (`CREATE TABLE IF NOT EXISTS`) da `lazyportfolio.v2.db.connect()` a ogni connessione -- niente comando di migrazione separato da eseguire.

## 2. I due percorsi di un messaggio

`POST /api/advisor/conversations/{id}/messages` accetta **esattamente uno** dei due corpi:

- `{"node_id": ..., "text": "..."}` -- il percorso reale: instrada al job `advisor_turn`, che chiama `advisor.agent.run_advisor_turn` (chiamata LLM vera, modello di default `deepseek-v4-flash`). Nessun tool a disposizione dell'LLM può scrivere nulla (§7.2) -- l'unico output possibile è una risposta testuale (`route="explain"`) o una proposta `pending_approval` dopo validazione deterministica (`route="propose"`).
- `{"node_id": ..., "views": [...]}` -- il percorso a fixture (Fase 3, senza LLM): instrada al job `fixture_proposal`, usato da test e da qualunque chiamante che voglia una proposta deterministica senza costo LLM. Nessun controllo UI lo espone più (rimosso in Fase 5), ma resta un percorso di prima classe della API/servizi, non deprecato.

Un corpo con entrambe le chiavi, o nessuna delle due, risponde `400`.

## 3. Diagnosticare un job bloccato

```bash
sqlite3 reports/tree_studio/tree_studio.sqlite3 \
  "SELECT job_id, kind, status, started_at, heartbeat_at, error_json FROM agent_jobs ORDER BY started_at DESC LIMIT 20;"
```

- `status='queued'` da più di qualche secondo con il server in esecuzione → il worker thread non sta girando (controlla i log del processo per un'eccezione in `_start_advisor_worker`'s `_reap_then_run`).
- `status='running'` con `heartbeat_at` fermo da oltre 60s → un worker precedente è morto a metà job; il reaper (`reap_orphaned_jobs(heartbeat_timeout_seconds=60)`, chiamato ad ogni iterazione del loop, ~5 volte al secondo) lo rimette `queued` automaticamente entro pochi secondi dal prossimo riavvio del server. Se il server è ancora in esecuzione e il job resta `running` per minuti, il worker thread stesso è probabilmente morto (eccezione non catturata) -- riavvia il processo.
- `status='failed'` → leggi `error_json`; per `advisor_turn` un fallimento tipico è una `ValueError` di `node_universe.validate_view_set` (la view proposta dall'LLM ha fallito la validazione -- comportamento atteso, non un bug) o un errore di rete/quota verso il provider LLM.

## 4. Ricostruire l'audit trail di una proposta

L'audit primario è nei repository di dominio (`lazyportfolio.advisor.*`), non nel `Session`/`EventLog` di LazyBridge (quello è un log di osservabilità LLM secondario, opzionale, mai la fonte di verità sul dominio):

```python
from uuid import UUID
from lazyportfolio.advisor import conversation_repository as conversations
from lazyportfolio.advisor import proposal_repository as proposals
from lazyportfolio.advisor.repository import get_head

# 1. la domanda/richiesta originale
messages = conversations.list_messages(conversation_id)

# 2. la proposta e il suo hash di contenuto
record = proposals.get(UUID(proposal_id))
record.proposal.content_hash          # hash mostrato all'utente prima dell'approvazione
record.proposal.evidence              # EvidenceRef -- vuoto nell'MVP, vedi §5 sotto
record.proposal.counterfactual        # baseline vs variant

# 3. l'approvazione (se avvenuta)
# tabella proposal_approvals: approved_by, approved_at, idempotency_key, new_revision_id

# 4. la nuova revisione del tree
head = get_head(record.proposal.tree_id)
head.revision_id, head.parent_revision_id, head.actor_id, head.reason
```

Ogni passo è raggiungibile a partire dal solo `proposal_id` (o `conversation_id`) -- non serve alcun log esterno. `tests/advisor/test_redteam.py`'s hash-tampering tests provano che il `content_hash` cambia per qualunque singolo campo alterato, quindi confrontare l'hash mostrato all'utente con `record.proposal.content_hash` rileva un payload manomesso tra visualizzazione e approvazione.

## 5. Gap noti (dichiarati, non nascosti)

- **Evidence fetching non cablato.** `EvidenceRef.locator` esiste nel contratto ma nessun codice in `src/lazyportfolio/advisor` o `project/advisor` lo dereferenzia come path -- non c'è ancora una pipeline che recuperi il contenuto di una fonte a partire dal suo locator. `tests/advisor/test_redteam.py` prova strutturalmente che questo è vero oggi; un futuro PR che aggiunge il fetch **deve** validare/sandboxare `locator` (allowlist di schema, no path traversal) prima che quel test venga aggiornato per permetterlo.
- **Reviewer disabilitato di default.** `project/advisor/reviewer.py`'s `review_proposal(..., enabled=False)` è il default -- nessuna chiamata a `claude_code` avviene a meno che un chiamante non passi esplicitamente `enabled=True`. Riabilitarlo esplicitamente quando serve una seconda opinione read-only su una proposta; non influisce mai sullo stato della proposta da solo.
- **`POST /api/advisor/proposals/{id}/revise` non implementato.** Il piano originale (§9.1) lo elenca; l'MVP realizza una revisione creando una *nuova* conversazione/proposta invece di un endpoint dedicato -- semanticamente equivalente (ogni proposta è comunque immutabile, §4.3), ma non c'è un endpoint con quel nome esatto.
- **Run di conferma non implementato.** Il contratto (`ProposalStatus`) e la state machine dichiarano gli stati `confirmation_pending`/`confirmed`/`confirmation_failed` (§4.5), e §1 del piano finalizzato descrive l'intento ("il run di conferma parte dopo il commit come job separato"), ma nessuna fase (0-5) ha mai incluso un task concreto per implementarlo -- `approval_service.apply_proposal` si ferma allo stato `applied`. Una proposta applicata oggi non transiterà mai automaticamente a `confirmed`. Gap del piano originale, non introdotto da questa fase; da valutare per Fase 6 se ancora rilevante.
- **`Session` del Node Advisor è in-memory.** `project/advisor/agent.py`'s `_advisor_session()` costruisce un `Session(redact=...)` senza `db=`: la redazione (segreti + PII) è cablata da subito, ma nessun log LLM viene persistito su disco per ora. Se in futuro serve osservabilità persistente delle chiamate LLM stesse (non l'audit di dominio, già persistito -- vedi §4), va passato un `db=` esplicito, mantenendo lo stesso redattore.

## 6. scheduled batch workflow (secondo producer, Fase 6)

`project/advisor/batch_producer.py`'s `run_proposal_batch(tree_id, node_views, ...)` è un secondo producer di `ChangeProposal`, non instradato da nessuna API HTTP -- va chiamato direttamente (script, shell Python, o un futuro trigger schedulato):

```python
from project.advisor.batch_producer import run_proposal_batch

result = run_proposal_batch(
    tree_id,
    {"equity": [...view dicts...], "bond": [...view dicts...]},
    db_path=db_path,
)
result.batch_id     # UUID condiviso da ogni proposta di questa run
result.proposals    # proposte create con successo (pending_approval)
result.errors       # {node_id: messaggio} per i nodi che hanno fallito la validazione
```

`node_views` è fornito interamente dal chiamante -- questo modulo non decide quali nodi toccare né quali view proporre (nessun ragionamento macro/market, vedi `docs/node-advisor-operational-plan.md` Fase 6 per il perché). Ogni proposta passa dalla stessa `services.create_proposal` del Node Advisor conversazionale: stessa validazione (una view su un financing instrument o fuori universo viene rifiutata identicamente, finisce in `errors`, non in `proposals`), stesso hash, stessa state machine. Un nodo fallito non blocca gli altri nodi della stessa batch.

## 7. Costo LLM

Ogni messaggio `text` instradato a `advisor_turn` è una chiamata LLM reale (default `deepseek-v4-flash`, tier economico). Budget dichiarato (§9.3, non ancora enforced a livello di codice in questa fase): 20 tool call/job, 5 fetch artifact, 5 fonti web, 1 reviewer esterno. Il percorso `fixture_proposal` (`views` esplicite) resta a costo zero per test/demo che non hanno bisogno di un LLM reale.
