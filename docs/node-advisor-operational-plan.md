# Node Advisor per Tree Studio — valutazione architetturale e piano operativo

**Stato:** finale, approvato per implementazione  
**Data di analisi:** 2026-08-09  
**Ambito iniziale:** proposte Black–Litterman `replace_node_views`  
**Ambito strategico:** primo incremento concreto del più ampio *Investment Committee* (`investment-process-top-down-etf.md`, root del workspace); i contratti definiti qui (proposta, snapshot, validazione, approvazione) devono restare **producer-agnostic** fin dall'MVP, in modo che il committee possa in seguito propagare le proprie view sugli stessi nodi senza un secondo sistema — vedi §3.4.  
**Principio non negoziabile:** l'LLM ricerca e formula ipotesi; LazyPortfolio valida e calcola; solo un comando umano, vincolato a una proposta immutabile, può creare una nuova revisione del tree.

## 1. Decisione esecutiva

La direzione di prodotto è valida e coerente con l'ecosistema: un Advisor contestuale per il nodo selezionato è più utile e più controllabile di un agente generale sul portafoglio. Anche la scelta di un `Plan` LazyBridge deterministico, invece di uno swarm autonomo, è corretta.

La proposta non è però implementabile in sicurezza come semplice aggiunta di una chat all'attuale Tree Studio. Prima servono quattro fondazioni:

1. revisioni append-only del tree e identità stabili indipendenti dal nome;
2. un contratto pubblico di snapshot/fingerprint dei dati, condiviso da Tree Studio e LazyTools;
3. un servizio quantitativo di controfattuale che carichi i dati una sola volta e confronti baseline e variante sullo stesso snapshot;
4. persistenza di dominio per conversazioni, job, proposte e approvazioni, distinta da `Memory`, `Store` e `Session` LazyBridge.

L'MVP deve consentire soltanto `replace_node_views`. Vincoli, topologia, proxy e benchmark restano fuori dal primo perimetro mutante. La modifica viene applicata in una transazione SQLite con compare-and-swap sulla head revision; il run di conferma parte dopo il commit come job separato.

La stima puntuale baseline/proposta è obbligatoria. Un backtest con una view formulata usando informazioni correnti va etichettato come **scenario storico non causale** e non come prova OOS. Un backtest causale delle view richiede una futura pipeline point-in-time che rigeneri la view a ogni signal date usando esclusivamente evidenze disponibili allora.

## 2. Evidenze dall'ecosistema attuale

Questa sezione separa capacità già disponibili, riuso con adattamento e lavoro nuovo.

### 2.1 LazyPortfolio e Tree Studio

Capacità già presenti:

- motore V2 gerarchico con modalità `flat`, `forward` e `forward_backward`;
- view Black–Litterman node-scoped (`V2View`) con `view_tau` e `view_covariance_policy`;
- validazione di obiettivi, valori finiti, confidence, `view_tau` e divieto di view su strumenti di financing;
- audit per nodo con pesi locali/terminali, expected return, rischio, TEV, financing e dettagli delle view;
- estimate puntuale e walk-forward backtest;
- store SQLite condiviso tra Tree Studio e LazyTools;
- run history con `config_hash`, `data_as_of`, `data_fingerprint`, pesi, metriche e artefatti;
- export Audit ZIP e report HTML, con registrazione best-effort nel registry.

Limiti reali:

- `trees` è oggi una tabella mutabile keyed by `name`; `write_model` esegue un upsert e non conserva revisioni;
- il `tree_id` operativo coincide di fatto con il nome sanitizzato, non con un'identità immutabile;
- `_data_fingerprint` e `_config_hash` sono funzioni private di `project/tree_studio.py`, quindi non sono contratti riusabili;
- la validazione delle view verifica forma e financing, ma non verifica esplicitamente che ogni strumento appartenga all'universo risolto del nodo;
- Tree Studio è una SPA in un singolo file HTML servita da `ThreadingHTTPServer`; estimate e backtest vengono eseguiti nel request thread;
- non esistono conversazioni, job persistenti, proposte, approval o una state machine applicativa;
- `portfolio_tree_estimate` e `portfolio_tree_backtest` duplicano parte dell'assembly applicativo di Tree Studio e non restituiscono lo stesso snapshot descriptor/run record;
- LazyPortfolio core non dipende da LazyBridge, e questo confine deve restare intatto.

Riferimenti principali: `src/lazyportfolio/v2/db.py`, `store.py`, `run_history.py`, `validation.py`, `contracts.py`, `project/tree_studio.py` e `project/tree_studio.html`.

### 2.2 LazyBridge

Capacità già presenti:

- `Agent` per il dialogo;
- `Plan` e `Step` per un workflow dichiarativo, tipizzato, con routing e resume;
- `Store(db=...)` SQLite con checkpoint, blackboard e compare-and-swap per singola chiave;
- `Session(db=...)` come event log tecnico con tool calls, tempi, token, costi ed errori;
- streaming dei passi sequenziali;
- redazione automatica dei formati comuni di segreti;
- `HumanEngine` per HIL sincrono e adattatori UI custom.

Correzioni rispetto alla proposta iniziale:

- `Memory` è in-process e non durevole; non può essere il sistema di persistenza delle chat;
- `Store` è un key-value store con commit per singola write, non è adatto a transazioni multi-tabella di dominio;
- `Session` è telemetria, non il registro autorevole di proposte o approvazioni;
- `HumanEngine` bloccante non è il meccanismo corretto per un'approvazione web che può arrivare ore dopo;
- l'agente persistente va inteso come **identità logica persistente**, non come processo LLM sempre vivo: a ogni job si ricostruisce il contesto conversazionale necessario dal database.

### 2.3 LazyTools, Market Data Hub, LazyStats e LazyCrawler

Capacità riusabili:

- `PortfolioTreeTools` espone validate/list/load/save/estimate/backtest sullo stesso store di Tree Studio;
- `DataHubTools` offre discovery, coverage e price summary bounded, con serie raw opt-in e capped;
- `RegistryTools` implementa correttamente search metadata-first e `artifact_get` selettivo;
- `RegimeTools` e gli strumenti statistici coprono regimi, volatilità e correlazioni;
- `macro_views_plan` dimostra un `Plan` deterministico macro + market → synthesis → report;
- LazyCrawler offre cache/news locale interrogabile;
- i connector `claude_code` e `codex` sono read-only e marcano l'output `content_is_untrusted`.

Gap da chiudere:

- `PortfolioTreeTools` usa `allow_write` anche per estimate/backtest, benché siano compute read-only; vanno separati `allow_compute` e `allow_persist`;
- il pipeline `macro_views` è universe-wide, scrive regimi, produce report e può inviare Telegram: non va collegato direttamente al Node Advisor; si riusano i blocchi di ricerca e i modelli, non la pipeline completa;
- manca un `NodeContext` canonico costruito da LazyPortfolio;
- manca un tool controfattuale che garantisca lo stesso dataset in memoria per i due solve;
- l'attuale view synthesis chiede copertura di tutti i ticker: per un Advisor conversazionale è una policy sbagliata; deve produrre zero o più view giustificate, senza forzare una view per ogni componente;
- il Codex Engine via app-server è ancora dichiarato non verificato live: nell'MVP usare i connector CLI read-only già consolidati o ClaudeCodeEngine, mantenendo il reviewer opzionale.

## 3. Architettura target e ownership

### 3.1 Confini tra repository

| Responsabilità | Owner | Decisione |
|---|---|---|
| Modello quantitativo, risoluzione universo nodo, snapshot, counterfactual | LazyPortfolio | API Python deterministica, nessuna dipendenza da LLM |
| Revisioni, proposte, approval, apply atomico | LazyPortfolio | dominio e repository SQLite |
| API/worker/UI Tree Studio | LazyPortfolio `project/` inizialmente | integration layer opzionale con LazyBridge/LazyTools |
| Runtime Agent/Plan/Store/Session | LazyBridge | riuso senza modifiche core nell'MVP |
| Provider data/registry/regimi/code-review | LazyTools | nuovi provider stretti e privilege-separated |
| Prezzi e freshness di origine | market-data-hub | source of record; nessuna write implicita |
| Statistiche/regimi | LazyStats/LazyTools | read-only nell'MVP del Advisor |
| News/cache locale | LazyCrawler | evidenza non fidata e point-in-time quando disponibile |

LazyPortfolio core non deve importare LazyBridge o LazyTools. Il composition root del Advisor vive nell'app Tree Studio o, se cresce oltre il locale single-user, in un futuro package applicativo separato.

### 3.2 Componenti logici

```text
Tree Studio UI
  ├─ editor e risultati V2 esistenti
  └─ pannello Node Advisor
       ├─ REST: conversazioni, messaggi, proposte, approval
       └─ SSE: eventi del job e token di risposta

Tree Studio application service
  ├─ NodeContextService
  ├─ AdvisorJobService + worker
  ├─ ProposalService
  └─ ApprovalApplicationService

LazyBridge runtime (per job)
  ├─ chat agent con history ricostruita
  ├─ Plan deterministico di proposal preparation
  ├─ Store per checkpoint namespaced per job
  └─ Session per trace tecnico namespaced per run

LazyPortfolio domain/quant
  ├─ TreeRepository revisionato
  ├─ NodeUniverseResolver
  ├─ SnapshotService
  ├─ ViewValidator
  ├─ CounterfactualEvaluator
  └─ apply atomico con CAS
```

### 3.3 Identità del Advisor

La chiave logica è:

```text
tree_id + node_id + user_id
```

`revision_id` e `data_fingerprint` appartengono al contesto di un singolo messaggio/job/proposta, non all'identità della conversazione. Se fossero parte dell'identità, ogni salvataggio spezzerebbe artificialmente la chat. Ogni messaggio assistant registra invece la revisione e lo snapshot effettivamente consultati.

Per installazione locale single-user, `user_id` può essere un valore stabile `local-user`, ma resta nel contratto per non rendere impossibile una futura modalità multiutente. Un job schedulato (es. una run giornaliera del committee) usa allo stesso modo un `user_id` stabile di tipo servizio (es. `committee-scheduler`), non una sessione umana — vedi §3.4.

### 3.4 Producer-agnostic contracts (predisposizione per l'Investment Committee)

Il Node Advisor è il primo produttore di `ChangeProposal`, non l'unico previsto. Il futuro Investment Committee (processo giornaliero, portfolio-wide, descritto in `investment-process-top-down-etf.md`) dovrà produrre proposte sugli stessi nodi usando lo stesso contratto di validazione/snapshot/approvazione, senza un secondo sistema parallelo. Questo impone quattro vincoli, tutti a costo marginale nullo se fissati ora e costosi da retrofittare dopo:

1. **`ChangeProposal.kind` è una stringa validata contro un registro di validator, non un `Literal` chiuso.** L'MVP registra un solo validator (`replace_node_views`); aggiungere in futuro un `kind` per proposte del committee (es. `committee_tilt_proposal`) è una nuova voce nel registro, non una migrazione di schema o un cambio del tipo di colonna.
2. **Le proposte hanno un `batch_id` opzionale (nullable) che le raggruppa.** Il Node Advisor conversazionale genera tipicamente proposte con `batch_id = NULL` (un nodo, una richiesta umana). Il committee genera in una singola run più proposte correlate su nodi diversi (es. un pillar equity e un pillar bond nello stesso giro decisionale): `batch_id` le lega senza forzare un apply atomico multi-nodo, che resta fuori scope MVP. La UI di approvazione può mostrare "queste N proposte appartengono alla stessa run" già dalla Fase 1, anche se il primo produttore reale del campo arriva solo con il committee.
3. **`ModelProvenance` distingue esplicitamente il tipo di produttore** (`producer_kind: Literal["interactive_chat", "scheduled_batch"]`, più `producer_id` libero, es. `node-advisor` o `investment-committee`). Non è solo un campo di audit: la UI e le policy di budget (§9.3) possono trattare diversamente una proposta nata da una conversazione umana e una nata da un job notturno, senza dover ispezionare l'origine per euristica.
4. **Il service layer (`NodeContextService`, `AdvisorJobService`, `ProposalService`) non deve assumere una richiesta HTTP con utente davanti.** Va scritto come funzioni/servizi richiamabili sia da un handler REST sia da un job schedulato (es. da LazyPulse), con l'identità del chiamante passata esplicitamente come parametro — mai letta da un contesto di sessione web implicito. Questo è un vincolo di dependency injection nella Fase 3, non un problema di calcolo.

Un quinto punto è di verifica, non di contratto: la Fase 0 deve includere una golden fixture con un nodo "pillar" (es. equity/bond/commodity a livello root, coerente con `investment-process-top-down-etf.md`) oltre alle fixture multi-livello già previste, per confermare che `NodeUniverseResolver` e `NodeContext` funzionano identicamente a quel livello di albero. Se emergono differenze semantiche, è meglio scoprirle qui che alla Fase 4.

Questi vincoli non allargano lo scope MVP: `replace_node_views`, un solo nodo per proposta, nessun LLM nel primo incremento restano invariati. Rendono solo aperti, invece che impliciti, i punti che il committee toccherà per primi.

## 4. Contratti canonici

### 4.1 NodeContext

Il contesto deve essere prodotto da LazyPortfolio, non ricostruito dall'LLM o duplicato in LazyTools.

```python
class NodeComponent(BaseModel):
    component_id: str
    kind: Literal["direct", "child"]
    label: str
    candidate_instrument: str
    child_node_id: str | None = None

class NodeContext(BaseModel):
    schema_version: Literal["1.0"]
    tree_id: UUID
    revision_id: UUID
    node_id: str
    node_name: str
    objective: str
    mode: Literal["flat", "forward", "forward_backward"]
    solved_components: list[NodeComponent]
    allowed_view_instruments: list[str]
    direct_instruments: list[str]
    child_node_ids: list[str]
    parent_node_id: str | None
    parent_candidate_instrument: str | None
    constraints: dict[str, Any]
    current_views: list[dict[str, Any]]
    snapshot: SnapshotDescriptor | None
    recent_run: RunSummary | None
```

`allowed_view_instruments` è la lista autorevole per le view. Per un figlio visto dal padre contiene il proxy/candidate del figlio, non i terminal ticker interni. La risoluzione deve riusare le identità `V2Component`/`V2SolveContext`, senza inventare una seconda semantica.

### 4.2 SnapshotDescriptor

Un semplice hash opaco non basta per audit e freshness.

```python
class SnapshotDescriptor(BaseModel):
    schema_version: Literal["1.0"]
    source: Literal["market-data-hub"]
    database_identity: str
    universe: list[str]
    start: date | None
    end: date | None
    data_as_of: date | None
    field: str
    currency: str
    frequency: str
    coverage: list[CoverageEntry]
    source_run_ids: list[str]
    fingerprint: str
```

Il fingerprint è SHA-256 della serializzazione canonica dei campi che influenzano il dataset. `coverage_report` è utile ma non sufficiente se una correzione storica cambia valori senza modificare `last_date` o `obs_count`: includere l'identità/versione di ingestione disponibile oppure un content digest calcolato dal backend sui dati effettivamente caricati.

### 4.3 Proposal e hashing

La proposta è immutabile. Una richiesta come “abbassa confidence a 0,25” crea una nuova proposta con `supersedes_proposal_id`; non aggiorna la precedente.

Campi minimi aggiuntivi rispetto alla bozza iniziale:

```python
class ModelProvenance(BaseModel):
    producer_kind: Literal["interactive_chat", "scheduled_batch"]
    producer_id: str  # es. "node-advisor", "investment-committee"
    model: str
    model_version: str | None = None
    prompt_version: str | None = None

class ChangeProposal(BaseModel):
    id: UUID
    schema_version: Literal["1.0"]
    kind: str  # validato contro un registro di validator; MVP registra solo "replace_node_views"
    batch_id: UUID | None  # raggruppa proposte correlate della stessa run; NULL per il Advisor conversazionale MVP
    supersedes_proposal_id: UUID | None
    tree_id: UUID
    base_revision_id: UUID
    node_id: str
    snapshot: SnapshotDescriptor
    information_cutoff: datetime
    patch: list[JsonPatchOperation]
    proposed_views: list[ProposedView]
    rationale: str
    caveats: list[str]
    evidence: list[EvidenceRef]
    model_provenance: ModelProvenance
    validation: ValidationResult
    counterfactual: CounterfactualResult
    expires_at: datetime
    content_hash: str
```

`kind` è una stringa e non un `Literal` chiuso perché è validata a runtime contro un registro di validator (§3.4, punto 1): l'MVP registra un solo `kind`, ma il tipo non deve cambiare quando se ne aggiunge un secondo. `batch_id` è nullable e non entra nel `content_hash` (due proposte con lo stesso contenuto ma raggruppamento diverso restano proposte distinte per apply, ma il batch è puro raggruppamento UI/audit). `status` non fa parte del payload immutabile: è stato operativo separato. Il content hash include schema version, kind, tree/revision/node, snapshot fingerprint, cutoff, patch, view, evidenze per hash, provenance, validation, counterfactual ed expiry. Esclude `status`, `batch_id`, timestamp tecnici di inserimento e contenuto UI derivato.

La serializzazione deve essere unica e testata: chiavi ordinate, UTF-8, timezone UTC, tickers normalizzati, nessun `NaN/Infinity`, numeri normalizzati e array in ordine semanticamente definito. Non basta `json.dumps(sort_keys=True)` se produttori diversi possono serializzare float in modo differente; adottare JSON Canonicalization Scheme (RFC 8785) o un equivalente interno documentato.

### 4.4 EvidenceRef

Ogni evidenza deve distinguere origine, contenuto recuperato e validità temporale:

```python
class EvidenceRef(BaseModel):
    id: UUID
    kind: Literal["artifact", "web", "datahub", "crawler", "agent_review"]
    locator: str
    title: str
    publisher: str | None
    retrieved_at: datetime
    published_at: datetime | None
    as_of: date | None
    content_hash: str | None
    excerpt: str
    supports_claims: list[str]
```

L'evidence card è dato non fidato. Non può determinare routing, privilegi o tool selection. Il reviewer esterno è una valutazione, non una fonte primaria: `agent_review` non deve da solo sostenere una view.

### 4.5 State machine

```text
drafting
  ├─ failed
  └─ pending_approval
       ├─ rejected
       ├─ expired
       ├─ superseded
       └─ applying
            ├─ apply_failed
            └─ applied
                 ├─ confirmation_pending
                 ├─ confirmed
                 └─ confirmation_failed
```

Le transizioni sono gestite da codice, con `UPDATE ... WHERE status = ?`. L'approvazione one-shot non è riutilizzabile e un retry idempotente dello stesso comando restituisce lo stesso risultato senza creare una seconda revisione.

## 5. Persistenza e migrazione

### 5.1 Schema target

Usare lo stesso file SQLite locale per semplicità operativa, ma con repository di dominio separati. Abilitare foreign keys, WAL e busy timeout su ogni connessione.

Tabelle:

- `trees` (esistente, **mai modificata**: `name TEXT PRIMARY KEY, config, created_at, updated_at` — vedi nota sotto);
- `tree_revisions(revision_id, tree_id, parent_revision_id, config_json, config_hash, created_at, actor_type, actor_id, reason)`;
- `tree_heads(tree_id, head_revision_id)` — la testa corrente di ogni tree, bersaglio del CAS; tabella nuova invece di una colonna `head_revision_id` aggiunta a `trees`, così `trees` resta byte-per-byte invariata e nessun chiamante esistente di `list_saved_models`/`read_model`/`write_model` richiede revisione (implementato in Fase 1, PR LP-02; vedi `docs/node-advisor-schema-migration-draft.md`);
- `agent_conversations(conversation_id, tree_id, node_id, user_id, created_at, updated_at)`;
- `agent_messages(message_id, conversation_id, role, content_json, revision_id, data_fingerprint, created_at)`;
- `agent_jobs(job_id, conversation_id, request_message_id, kind, status, checkpoint_key, session_db_path, budget_json, started_at, heartbeat_at, finished_at, error_json)`;
- `change_proposals(proposal_id, batch_id, supersedes_proposal_id, tree_id, base_revision_id, node_id, kind, producer_kind, producer_id, payload_json, content_hash, status, expires_at, created_at)` — `batch_id` nullable, nessuna foreign key verso una tabella batch nell'MVP (raggruppamento debole, vedi §3.4); `kind`/`producer_kind`/`producer_id` denormalizzati dal payload per poter filtrare/indicizzare senza deserializzare il JSON;
- `proposal_approvals(approval_id, proposal_id, approved_by, approved_at, approved_hash, idempotency_key, applied_revision_id, result_json)`;
- `proposal_evidence(proposal_id, evidence_id, metadata_json, excerpt, content_hash)`;
- `outbox_events(event_id, aggregate_type, aggregate_id, event_type, payload_json, created_at, delivered_at)`.

Vincoli essenziali:

- unique su `(tree_id, name)` o, in locale, su `name` mantenendo `tree_id` stabile;
- unique su `(conversation_id, node_id)` non necessario: una conversazione è già node-scoped;
- unique su `content_hash` solo se si vuole dedup esplicito, non come requisito globale;
- unique su `proposal_approvals.proposal_id` e su `idempotency_key`;
- foreign keys e check sugli status;
- `tree_heads.head_revision_id` referenzia una revision dello stesso tree (foreign key verso `tree_revisions.revision_id`), verificato anche dal repository/service.

### 5.2 Migrazione senza rotture

1. Creare le nuove tabelle senza modificare `trees` esistente.
2. Migrare ogni row legacy in un tree con UUID e una revisione iniziale; salvare una mapping table `legacy_tree_names`.
3. Aggiornare read/list/load per leggere la head revision, preservando il contratto esterno keyed by name.
4. Aggiornare `write_model` perché ogni salvataggio umano crei una revisione, con expected head opzionale.
5. Mantenere per una release una compatibilità di lettura con eventuali database non migrati.
6. Solo dopo test di round-trip e backup documentato, rinominare/rimuovere il payload legacy dalla vecchia tabella.

Non fare dual-write prolungato tra due rappresentazioni autorevoli: introdurrebbe divergenza. La migrazione deve avere un'unica source of truth per fase.

### 5.3 Ruolo di Memory, Store e Session

- **Conversazione:** `agent_messages`; il worker seleziona gli ultimi turni e un summary versionato per costruire una `Memory` temporanea.
- **Workflow:** `Store(db=...)` con chiavi `advisor/{job_id}/...` e checkpoint key unica.
- **Audit tecnico:** `Session(db=...)`; `agent_jobs` conserva il riferimento al file/session run.
- **Stato autorevole:** tabelle revision/proposal/approval; mai dedotto dagli eventi Session.

## 6. Servizi quantitativi da aggiungere a LazyPortfolio

### 6.1 NodeUniverseResolver

API proposta:

```python
resolve_node_context(config, node_id, *, mode) -> NodeContext
validate_view_set(config, node_id, views, *, mode) -> ValidationResult
```

Regole:

- ticker normalizzati con la funzione canonica `ticker()`;
- ogni chiave di pick deve essere in `allowed_view_instruments`;
- vietati cash lend, cash borrow e alias equivalenti;
- coefficienti finiti e almeno uno non nullo;
- confidence in `(0, 1]`, expected return finito, `view_tau > 0`;
- rilevazione deterministica di duplicati esatti e pick opposti con stesso orizzonte;
- warning, non decisione LLM, per scale non comparabili o expected return estremo;
- per `min_risk` e `hrp` con `prior_risk`, risultato `no_effect_on_weights` e nessuna proposal pending approval;
- il validator restituisce error code machine-readable e messaggio UI.

### 6.2 SnapshotService

Spostare la logica privata di fingerprint da Tree Studio nel package. Tree Studio e LazyTools devono chiamare lo stesso servizio. Il loader deve restituire insieme dataset e descriptor; non deve essere possibile eseguire un counterfactual con dataset A e dichiarare fingerprint B.

### 6.3 CounterfactualEvaluator

API proposta:

```python
evaluate_view_counterfactual(
    base_config,
    node_id,
    proposed_views,
    dataset,
    snapshot,
) -> CounterfactualResult
```

Sequenza:

1. valida base config e patch;
2. crea una copia canonica in memoria, senza persistere;
3. verifica che la patch tocchi solo `/nodes/{node}/constraints/views` nell'MVP;
4. esegue baseline e variante usando lo stesso frame di ritorni già caricato;
5. produce delta locali e terminali con chiavi esplicite;
6. calcola variazioni di expected return, volatilità, TEV, cash/financing, gross/net exposure;
7. calcola turnover one-way stimato come `0.5 * sum(abs(w_new - w_old))`, dichiarando la convenzione;
8. registra versioni solver, seed e parametri rilevanti.

Il risultato deve contenere sia valori assoluti sia delta, e distinguere `0`, `null` e metrica non applicabile.

### 6.4 Semantica del backtest

Tre modalità, nominate senza ambiguità:

- `current_estimate_counterfactual`: obbligatoria e causalmente corretta rispetto allo snapshot corrente;
- `historical_static_view_sensitivity`: opzionale, applica retroattivamente la view corrente e porta un warning `uses_future_formulated_view=true`;
- `point_in_time_view_backtest`: futuro, rigenera/reperisce una view per ogni signal date con cutoff antecedente; unica modalità che può sostenere claim OOS sulla strategia di view generation.

L'MVP non deve mostrare la seconda come “backtest della proposta” senza il warning prominente. La terza è fuori scope finché registry e crawler non garantiscono versioni point-in-time delle evidenze.

## 7. Tool surface e policy di privilegi

### 7.1 Profili

Creare profili dichiarativi, non assemblati ad hoc dal prompt:

| Profilo | Tool |
|---|---|
| `explain` | node context, parent/children summary, recent runs, artifact search/get, bounded data summaries |
| `research` | `explain` + web/crawler + statistiche/regimi read-only |
| `prepare_view_proposal` | `research` + validate views + estimate counterfactual + create proposal |
| `review` | proposta/evidenze read-only + Claude/Codex read-only |

Nessun profilo agentico contiene save/delete/apply, refresh data, Telegram/email o write shell.

### 7.2 Modifiche a LazyTools

Refactor di `PortfolioTreeTools`:

```python
PortfolioTreeTools(
    allow_compute=False,
    allow_persist=False,
    allow_delete=False,
)
```

- validate/list/load sono sempre read-only;
- estimate/backtest dipendono da `allow_compute`;
- save dipende da `allow_persist`;
- delete è un privilegio distinto e non viene mai dato al Advisor.

Nuovo provider `NodeAdvisorReadTools`:

- `tree_get_node_context`;
- `tree_get_parent_context`;
- `tree_get_child_summaries`;
- `tree_get_revision`;
- `tree_get_recent_runs`;
- `portfolio_tree_validate_views`;
- `portfolio_tree_estimate_counterfactual`;
- opzionale `portfolio_tree_static_view_sensitivity` con warning nel payload.

`create_change_proposal` appartiene al servizio applicativo Tree Studio e registra soltanto una proposta già validata/counterfattuale. Non accetta patch arbitrarie generate dall'LLM nell'MVP: riceve `node_id` e `proposed_views`, poi il server costruisce la JSON Patch canonica.

### 7.3 Reviewer esterno

Ordine consigliato:

1. reviewer assente per il primo vertical slice;
2. connector `claude_code(mode="read")` o `codex()` come second opinion opzionale;
3. ClaudeCodeEngine con `file_roots` espliciti se serve una conversazione reviewer più ricca;
4. CodexEngine solo dopo smoke test live dell'app-server e pin della versione.

Il reviewer restituisce un modello `ReviewResult` con finding, severity, claim/evidence mismatch e raccomandazione. Non riscrive direttamente le view e non promuove una proposta a pending approval.

## 8. Workflow operativo

### 8.1 Risposta informativa

Per domande come “Perché siamo sottopesati in Europa?” non serve il Plan completo:

1. caricare `NodeContext`, ultimo run e revision;
2. recuperare solo i dettagli audit necessari;
3. rispondere con citazioni interne al run/config;
4. salvare messaggio e provenance;
5. nessuna proposal.

### 8.2 Preparazione di una proposta BL

Il `Plan` dichiarativo contiene callable deterministiche e passi LLM confinati:

1. `load_context` — callable LazyPortfolio;
2. `clarify_or_continue` — LLM con output tipizzato; route a risposta utente se manca una scelta materiale;
3. `retrieve_evidence` — ricerca metadata-first, fetch selettivo, budget e cutoff;
4. `synthesize_candidate_views` — LLM → `CandidateViewSet`;
5. `validate_candidate_views` — callable deterministica;
6. `review_candidate_views` — opzionale, read-only;
7. `load_snapshot_once` — callable, dataset custodito fuori dal prompt;
8. `run_counterfactual` — callable LazyPortfolio;
9. `create_proposal` — servizio di dominio, proposta immutabile;
10. `render_proposal_message` — presentazione, senza alterare il payload.

Le serie raw non entrano nel prompt. Gli step quantitativi si scambiano riferimenti interni a dataset/job o oggetti in-process, non matrici JSON.

### 8.3 Approvazione e apply

Endpoint suggerito:

```text
POST /api/advisor/proposals/{proposal_id}/approve
{
  "proposal_hash": "sha256:...",
  "idempotency_key": "...",
  "approved_by": "local-user"
}
```

Nella singola transazione:

1. leggere proposta e status;
2. verificare hash con confronto constant-time;
3. verificare expiry;
4. rileggere `tree_heads.head_revision_id` e confrontarla con `base_revision_id`;
5. ricostruire e validare la patch server-side;
6. ricalcolare il fingerprint sul medesimo contratto di snapshot (Fase 1: hook `recompute_fingerprint` con default che si fida del fingerprint salvato — un vero ricalcolo richiede il `SnapshotService` di Fase 2);
7. applicare patch su una copia e validare `V2Model` completo;
8. inserire nuova `tree_revision`;
9. aggiornare `tree_heads` con CAS `WHERE tree_id = ? AND head_revision_id = base_revision_id`;
10. inserire approval/result e outbox event;
11. commit.

Dopo il commit, il worker consuma l'outbox e avvia estimate di conferma. Un fallimento del run non annulla silenziosamente la revisione già applicata: la UI mostra `confirmation_failed` e offre retry o rollback tramite nuova revisione.

Se revision o dati sono cambiati, rispondere `409 Conflict` con codice `stale_revision` o `stale_data`; la proposta passa a `expired` e non può essere riattivata.

### 8.4 Rifiuto e modifica

- reject registra attore, timestamp e motivo opzionale;
- modifica genera un nuovo job e una nuova proposta `supersedes`;
- la proposta precedente passa a `superseded` solo quando la nuova raggiunge `pending_approval`;
- nessuna approvazione eredita hash o stato dalla precedente.

## 9. API, worker e streaming

### 9.1 API minima

- `GET /api/trees/{tree_id}/nodes/{node_id}/advisor/context`;
- `GET/POST /api/advisor/conversations`;
- `GET /api/advisor/conversations/{id}/messages`;
- `POST /api/advisor/conversations/{id}/messages` → crea job; il corpo determina quale (`{"text": ...}` → job `advisor_turn`, LLM reale via `advisor.agent.run_advisor_turn`; `{"views": [...]}` → job `fixture_proposal`, deterministico senza LLM -- esattamente uno dei due, mai entrambi, mai nessuno; vedi `docs/node-advisor-runbook.md` §2);
- `GET /api/advisor/jobs/{job_id}`;
- `GET /api/advisor/jobs/{job_id}/events` → SSE;
- `GET /api/advisor/proposals/{proposal_id}`;
- `POST /api/advisor/proposals/{id}/approve`;
- `POST /api/advisor/proposals/{id}/reject`;
- `POST /api/advisor/proposals/{id}/revise`.

### 9.2 Worker locale MVP

Non eseguire ricerca, LLM o backtest nel `BaseHTTPRequestHandler`.

Per il deployment locale iniziale è sufficiente:

- tabella `agent_jobs` come coda durevole;
- un worker process/thread dedicato che claimi job con transizione atomica `queued → running`;
- heartbeat e recovery dei job orfani;
- un solo job mutante/apply per tree, più concorrenza configurabile per job read-only;
- eventi persistiti e inoltrati via SSE;
- timeout e cancellation cooperativa;
- nessun `await_approval` che occupi un worker.

Prima del Advisor conviene estrarre dal monolite `tree_studio.py` moduli `api`, `services`, `repositories` e `jobs`, mantenendo lo stesso entry point. Non serve introdurre subito un framework web, ma SSE, routing e test diventano più semplici con FastAPI/Starlette; la scelta va presa con un breve ADR nella Fase 0.

### 9.3 Budget di default

Configurazione iniziale, esplicita in UI:

- massimo 20 tool call per proposal job;
- massimo 5 artifact full fetch;
- massimo 5 fonti web/crawler;
- massimo 1 reviewer esterno;
- estimate baseline/proposta obbligatoria;
- backtest disattivato di default;
- timeout per tool e timeout globale;
- massimo un retry per structured output e per reviewer;
- cost estimate prima di un job classificato `expensive`.

## 10. UX finalizzata

Il pannello laterale resta legato al nodo selezionato e ha quattro tab:

- **Chat** — conversazione e streaming;
- **Fonti** — evidence card, data/cutoff, stale/missing flags;
- **Proposte** — pending, superseded, expired, applied;
- **Audit** — tool event sintetici e link alla trace tecnica autorizzata.

Eventi UI ammessi:

- contesto caricato;
- ricerca artefatti completata;
- coverage verificata;
- view candidata validata/rifiutata;
- counterfactual completato;
- review indipendente completata;
- proposta pronta o fallita.

Non mostrare chain-of-thought, prompt raw, segreti o payload di serie. Mostrare invece decision record e motivazioni sintetiche.

La proposal card include sempre:

- ID breve e hash abbreviato;
- tree/revision/node;
- data snapshot e information cutoff;
- view prima/dopo;
- delta pesi locali e terminali;
- metriche assolute e delta;
- caveat, warning causali e fonti;
- expiry;
- azioni `Apri diff`, `Apri evidenze`, `Approva e applica`, `Rifiuta`, `Modifica`.

Il pulsante approve richiede una conferma che ripeta nodo e sintesi della patch. Se topologia o dati cambiano mentre la card è aperta, il server rifiuta comunque l'apply; la UI non è il security boundary.

## 11. Sicurezza e invarianti

Invarianti da codificare come test, non solo prompt:

- l'Agent non possiede alcun tool che possa aggiornare la head revision;
- l'endpoint approve non accetta una patch client-side;
- la patch MVP può toccare un solo nodo e solo `constraints.views`;
- proposta, revision e fingerprint devono coincidere al momento dell'apply;
- una proposal approvata/applicata non è riutilizzabile;
- tutte le write sono idempotenti o protette da unique constraint/CAS;
- web, crawler, artifact content e reviewer sono untrusted data;
- tool selection e privilege profile derivano dal server, non dal contenuto recuperato;
- `Session` usa redazione custom per PII oltre alla redazione segreti default;
- excerpt e contenuto completo hanno limiti di dimensione;
- i path per Claude/Codex sono allowlist esplicite;
- nessuna write su Market Data Hub è implicita;
- nessun invio esterno è disponibile al Advisor MVP;
- SQLite apre `PRAGMA foreign_keys=ON`, WAL e busy timeout;
- audit e log non sono l'unica copia della configurazione applicata.

## 12. Strategia di test ed eval

### 12.1 Unit e contract test

- risoluzione corretta dell'universo per direct instrument, child proxy, flat/forward/backward;
- view fuori universo rifiutata prima del solve;
- financing e coefficienti nulli/non finiti rifiutati;
- `min_risk`/`hrp` + `prior_risk` produce `no_effect_on_weights` e nessuna proposta;
- canonical hash stabile su processi diversi e sensibile a ogni campo vincolante;
- patch allowlist e ricostruzione server-side;
- state transition illegali rifiutate;
- idempotency e CAS concorrente;
- migrazione store legacy round-trip byte/equivalence;
- stesso config + stesso dataset + stessi seed produce stesso counterfactual entro tolleranze numeriche dichiarate.

Non richiedere che due chiamate LLM indipendenti producano la stessa view: la deterministica ripetibilità riguarda validazione, payload canonico a input fissato, calcolo e apply. Per testare synthesis usare fixture o response cache.

### 12.2 Integration test

- proposta creata su revision R1, salvataggio umano crea R2, apply restituisce 409;
- nuovi dati o correzione storica cambiano fingerprint e fanno scadere la proposta;
- approvazione P1 non può applicare P2;
- due approve concorrenti producono una sola revision;
- crash dopo commit e prima del confirmation job viene recuperato dall'outbox;
- restart del server conserva conversazione, job, proposta e trace;
- baseline e variante usano lo stesso oggetto snapshot;
- report malevolo con prompt injection non altera tool profile né genera proposal non richiesta;
- fallimento reviewer non impedisce una proposta se la policy lo dichiara opzionale;
- fallimento confirmation run è visibile e retryable.

### 12.3 Financial/causal eval

- delta locale e terminale riconciliati con l'audit V2;
- financing, gross/net exposure e cash coerenti prima/dopo;
- turnover coerente con la convenzione documentata;
- static sensitivity sempre marcata hindsight;
- point-in-time backtest, quando implementato, rifiuta evidenze con `published_at/retrieved_at` successivi al signal cutoff;
- artefatti senza timestamp affidabile non sono eleggibili per claim causali;
- nessun linguaggio UI promette performance futura o presenta engineering validation come superiorità finanziaria.

### 12.4 Red-team

- injection in HTML/report/web page;
- evidence locator che tenta path traversal;
- prompt che chiede save/delete/refresh/invio esterno;
- proposal payload manomesso tra visualizzazione e approve;
- replay di approval;
- hash collision test fixture/manomissione di float/order;
- database locked, worker duplicato e job orphan;
- session log con token, email e identificatori sensibili.

## 13. Piano operativo per fasi

Le stime sono in **giorni ingegnere** e servono per sequencing, non come promessa di calendario. Un team di due persone può parallelizzare UI e dominio solo dopo la chiusura dei contratti di Fase 0.

### Fase 0 — ADR e contratti eseguibili (3–5 giorni)

Deliverable:

- ADR su confini repository e scelta HTTP/SSE;
- Pydantic model v1 per NodeContext, Snapshot, Proposal (incl. `kind` come registro di validator e `ModelProvenance.producer_kind`), Evidence, Validation e Counterfactual;
- state machine e JSON Patch allowlist;
- golden fixtures per tree multi-livello, view BL e un nodo pillar-level (§3.4, punto 5) per la compatibilità futura con l'Investment Committee;
- schema migration draft e backup/rollback procedure, incl. le colonne `batch_id`/`kind`/`producer_kind`/`producer_id` di `change_proposals`.

Exit criteria:

- nessun campo semantico ambiguo;
- ownership approvata;
- hash contract con golden vectors;
- decisione esplicita sulla semantica backtest;
- `NodeContext` risolto correttamente anche sulla fixture pillar-level, senza differenze semantiche rispetto a un nodo interno.

### Fase 1 — Revisioni e dominio proposal (8–12 giorni)

Repository: LazyPortfolio.

Deliverable:

- nuove tabelle e migration idempotente;
- `TreeRepository` append-only;
- compatibilità list/load/save keyed by name;
- `ProposalRepository` e state transitions, con registro `kind` estensibile e `batch_id` nullable fin dalla prima migration (§3.4);
- `ApprovalApplicationService` con transaction, CAS e idempotency;
- outbox per confirmation job;
- test di concorrenza e migration.

Exit criteria:

- ogni save manuale crea una revision;
- nessun overwrite perde storia;
- due apply concorrenti non possono entrambi riuscire;
- suite V2 e Tree Studio esistente resta verde.

### Fase 2 — Contesto, snapshot e controfattuale (8–12 giorni)

Repository: LazyPortfolio, poi wrapper LazyTools.

Deliverable:

- `NodeUniverseResolver` e validator node-scoped;
- `SnapshotService` pubblico usato da Tree Studio;
- `CounterfactualEvaluator` one-load/two-solve;
- delta schema e turnover convention;
- refactor `PortfolioTreeTools` dei privilegi;
- `NodeAdvisorReadTools` con bounded outputs.

Exit criteria:

- impossibile validare una view sul livello economico sbagliato;
- Tree Studio e LazyTools emettono lo stesso fingerprint;
- baseline/variant condividono lo snapshot;
- `min_risk`/`hrp` non produce proposal cosmetiche.

### Fase 3 — Job, conversazioni e vertical slice senza LLM (7–10 giorni)

Repository: LazyPortfolio application layer.

Deliverable:

- moduli API/service/repository estratti dal monolite, con identità del chiamante passata esplicitamente ai service (mai letta da sessione HTTP implicita), così da restare richiamabili anche da un job schedulato futuro (§3.4, punto 4);
- conversazioni e messaggi persistenti;
- job queue locale, worker, heartbeat/recovery e SSE;
- UI panel/tab base;
- creazione di una proposta da fixture deterministica;
- approve/reject/apply end-to-end e confirmation run.

Exit criteria:

- demo completa senza LLM: fixture → card → approve → nuova revision → conferma;
- restart durante ogni stato critico recuperabile;
- nessun lavoro lungo nel request thread.

### Fase 4 — Node Advisor conversazionale e ricerca (8–12 giorni)

Repository: Tree Studio integration + LazyTools; LazyBridge riusato.

Deliverable:

- composition root Agent/Plan;
- history reconstruction e summary versionato;
- profili `explain`, `research`, `prepare_view_proposal`;
- artifact search/get metadata-first;
- DataHub/statistics/crawler/web bounded;
- structured synthesis e retry limitato;
- proposal rendering con caveat/citazioni.

Exit criteria:

- domande informative non avviano il proposal Plan;
- “prepara ma non applicare” produce solo pending approval;
- injection test non amplia privilegi;
- budget/timeout/costi sono visibili e auditati.

**Nota di implementazione (Fase 5):** `project/advisor/agent.py`'s `run_advisor_turn` è stato scritto e testato in Fase 4 (PR LP/LT-09) come funzione standalone, ma non era ancora cablato nel worker/API live né nella UI di Tree Studio -- il worker registrava solo il job kind `fixture_proposal` (Fase 3) e la chat era ancora una textbox per JSON di view esplicite. Il cablaggio reale (nuovo job kind `advisor_turn`, routing API per forma del corpo `views` vs `text`, riscrittura della chat UI) è stato completato in Fase 5 dopo essere stato individuato durante il Deep Audit di Fase 5 -- vedi `docs/node-advisor-runbook.md` §2.

### Fase 5 — Reviewer, hardening ed eval (5–8 giorni)

Deliverable:

- reviewer read-only opzionale;
- custom Session redactor e retention policy;
- red-team suite e failure injection;
- metriche operative: success rate, stale rate, validation reject rate, apply latency, confirmation failure, cost per proposal;
- manuale operativo e recovery runbook.

Exit criteria:

- tutti gli eval critici passano;
- nessuna P0/P1 aperta;
- audit ricostruisce domanda → evidenze → proposal hash → approval → revision → confirmation run.

**Nota di implementazione:** il "confirmation run" descritto al §1 non è mai stato implementato in nessuna fase (0-5) -- `approval_service.apply_proposal` si ferma allo stato `applied`; gli stati `confirmation_pending`/`confirmed`/`confirmation_failed` esistono nel contratto e nella state machine (§4.5) ma non sono mai raggiunti. `tests/advisor/test_audit_reconstruction.py` ricostruisce la catena end-to-end fino ad `applied` (l'ultimo stato realmente raggiungibile), non oltre. Gap del piano originale, dichiarato in `docs/node-advisor-runbook.md` §5, da valutare per Fase 6 se ancora rilevante.

### Fase 6 — Estensioni successive (non MVP)

- **Investment Committee come secondo producer** di `ChangeProposal` -- l'obiettivo strategico dichiarato all'avvio di questo progetto (§1: "primo incremento concreto del più ampio Investment Committee");
- patch di vincoli non strutturali con schema separato;
- proposte di topologia/proxy/benchmark ad alto impatto;
- confronto e ranking di proposte;
- rollback come nuova revisione, mai riscrittura della storia;
- vista aggregata delle view nell'albero;
- pipeline point-in-time per backtest causale delle view;
- deployment multiutente con autenticazione e RBAC;
- notifiche esterne con approval separata.

**Nota di implementazione (Investment Committee):** `project/advisor/committee.py`'s `run_committee_batch` implementa solo la **prova strutturale** -- un secondo producer non interattivo (`producer_kind="scheduled_batch"`, `producer_id="investment-committee"`) che crea proposte `pending_approval` su più nodi in un'unica run condividendo un `batch_id`, attraverso la stessa `services.create_proposal` (stessa validazione, stesso hash, stessa state machine) usata dal Node Advisor conversazionale e dal percorso fixture -- zero modifiche a schema o state machine, la prova che la scelta producer-agnostic di Fase 0/§3.4 ha retto. `node_views` (quali nodi toccare, quali view proporre) è fornito dal chiamante -- **non c'è ancora un vero ragionamento del committee** (sintesi multi-specialist macro/market, decisione autonoma su quali nodi toccare): quello resta un lavoro sostanziale a sé, non ancora pianificato in dettaglio, così come il trigger schedulato (LazyPulse) reale -- entrambi rimandati a quando servirà davvero un committee che ragiona, non solo un producer che scrive.

## 14. Ordine dei pull request

PR piccoli e verificabili:

1. **LP-01:** ADR, contratti Pydantic e hash golden vectors.
2. **LP-02:** schema revisioni + migration + repository compatibile.
3. **LP-03:** proposal/approval/outbox + state machine.
4. **LP-04:** node universe resolver e validation error codes.
5. **LP-05:** snapshot service centralizzato e refactor caller.
6. **LP-06:** counterfactual evaluator e delta contract.
7. **LT-01:** split privilegi `PortfolioTreeTools` + tool read-only Advisor.
8. **LP-07:** job/conversation API e worker.
9. **LP-08:** pannello UI e vertical slice deterministico.
10. **LP/LT-09:** LazyBridge Agent/Plan, ricerca e synthesis.
11. **LP-10:** approval UI, confirmation, audit completo.
12. **LT-02:** reviewer opzionale e hardening.

Ogni PR deve includere migration/rollback notes, test contract e aggiornamento di questo documento se cambia un'invariante.

## 15. Definition of Done dell'MVP

L'MVP è completo soltanto quando un utente può:

1. selezionare un nodo e aprire una conversazione che sopravvive al restart;
2. chiedere spiegazioni sui pesi usando config, snapshot e run reali;
3. chiedere una proposta BL senza modificare il tree;
4. vedere fonti, cutoff, limiti, view, diff e impatto locale/finale;
5. modificare la richiesta ottenendo una nuova proposta/hash;
6. approvare una sola proposta specifica;
7. ricevere un rifiuto sicuro se revision o dati sono cambiati;
8. ottenere una nuova revision applicata atomicamente e un run di conferma;
9. ricostruire l'intera catena dall'audit.

Sono esplicitamente fuori dalla Definition of Done: modifica topologica, refresh dati, invio esterno, auto-approval, backtest causale della strategia di view generation e processi agentici sempre residenti.

## 16. Rischi principali e mitigazioni

| Rischio | Impatto | Mitigazione |
|---|---|---|
| View applicata al livello errato | Alto | `allowed_view_instruments` canonico dal solve context |
| Proposal stale | Alto | base revision + snapshot fingerprint + expiry + CAS |
| Hindsight presentato come OOS | Alto | tre modalità di test nominate; point-in-time fuori MVP |
| Divergenza Tree Studio/LazyTools | Alto | servizi condivisi LazyPortfolio per snapshot e compute |
| Apply duplicato/concorrenza SQLite | Alto | transazione, unique idempotency, CAS, WAL/busy timeout |
| Prompt injection da fonti | Alto | dati untrusted, profili server-side, nessun write tool |
| Job perso al restart | Medio/alto | job table, heartbeat, recovery e transactional outbox |
| Context growth | Medio | history window + summary versionato, contenuti fetch selettivi |
| Costo/backtest eccessivo | Medio | estimate obbligatoria, backtest opt-in, budget visibile |
| Complessità del monolite UI/server | Medio | estrazione moduli prima dell'integrazione agentica |
| Reviewer esterno instabile | Medio | opzionale, timeout, output tipizzato, nessuna autorità |

## 17. Primo incremento consigliato

Il primo incremento non deve chiamare alcun LLM. Deve dimostrare la catena di sicurezza con una view fixture:

```text
tree legacy → revision R1
→ NodeContext canonico
→ view fixture validata
→ stesso snapshot per baseline/variant
→ proposal P1 immutabile
→ card UI
→ approval(hash P1, R1, fingerprint F1)
→ CAS e revision R2
→ confirmation estimate
→ audit ricostruibile
```

Se questo vertical slice non è solido, aggiungere chat e ricerca rende soltanto più difficile osservare i difetti. Una volta chiuso, LazyBridge può essere innestato come produttore controllato della stessa `CandidateViewSet`, senza cambiare il security model né il percorso di apply.

## 18. Conclusione

Il Node Advisor deve essere un sistema di **proposal preparation**, non un portfolio manager autonomo. Il disegno finale mantiene tre autorità separate:

- LazyBridge orchestra conversazione, ricerca e sintesi;
- LazyPortfolio definisce universo, validazione, snapshot, calcolo e revisioni;
- Tree Studio presenta le prove e raccoglie l'unico comando umano mutante.

La qualità del prodotto dipende meno dal numero di agenti e più dalla precisione dei contratti: identità del nodo, snapshot, patch allowlist, hash canonico, state machine e controfattuale sullo stesso dataset. Queste fondazioni rendono possibile aggiungere in seguito reviewer, vincoli e modifiche strutturali senza indebolire il principio centrale: nessuna modifica economica silenziosa, non ricostruibile o applicata fuori contesto.
