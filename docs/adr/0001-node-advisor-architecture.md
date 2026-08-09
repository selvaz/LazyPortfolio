# ADR 0001 — Node Advisor: confini di repository, trasporto e contratti producer-agnostic

**Stato**: accettata
**Data**: 2026-08-09
**Riferimento**: `docs/node-advisor-operational-plan.md` (piano finalizzato)

## Contesto

Il Node Advisor introduce un LLM che propone modifiche (inizialmente solo
view Black-Litterman) a un nodo dell'albero gerarchico di LazyPortfolio.
Prima di scrivere qualunque contratto o tabella serve fissare tre decisioni
architetturali che altrimenti verrebbero prese implicitamente file per file,
rendendo costoso tornare indietro in una fase successiva.

## Decisione 1 — Confini tra repository

`LazyPortfolio` (core quantitativo: modello, universo del nodo, snapshot,
controfattuale, revisioni) **non dipende** da `LazyBridge` né da `LazyTools`.
Il composition root dell'agente (Plan/Agent LazyBridge, profili tool) vive
nell'application layer sotto `project/` (Tree Studio) o in un futuro package
applicativo separato, mai in `src/lazyportfolio/`. `LazyTools` espone i
provider LLM-facing (`NodeAdvisorReadTools`, refactor privilegi
`PortfolioTreeTools`) come proprio pacchetto, chiamando LazyPortfolio come
libreria Python, non il contrario.

Motivazione: LazyPortfolio deve restare utilizzabile (e testabile) senza
alcuna dipendenza LLM — è il motore quantitativo, non l'agente. Invertire la
dipendenza renderebbe impossibile usare LazyPortfolio in un contesto
batch/non conversazionale (esattamente il caso d'uso del futuro Investment
Committee, vedi Decisione 3).

## Decisione 2 — Trasporto HTTP/SSE minimale, non un framework nuovo subito

Nell'MVP (Fase 3), l'API REST e lo streaming SSE vengono aggiunti
all'infrastruttura esistente di Tree Studio (`BaseHTTPRequestHandler`
estratto in moduli `api`/`services`/`repositories`/`jobs`), non introducendo
FastAPI/Starlette da subito. Motivazione: il volume (single-user locale) non
giustifica la dipendenza aggiuntiva finché SSE/routing manuale restano
gestibili; la scelta va rivalutata con un breve ADR dedicato se la Fase 3
mostra che il routing manuale è diventato un collo di bottiglia reale (non
anticipato).

## Decisione 3 — Contratti producer-agnostic fin dall'MVP

Il Node Advisor è il primo produttore di `ChangeProposal`, non l'unico
previsto: il futuro Investment Committee (`investment-process-top-down-etf.md`)
deve poter produrre proposte sugli stessi nodi usando lo stesso contratto di
validazione/snapshot/approvazione, senza un secondo sistema parallelo. Questo
impone, fin dalla Fase 0:

1. `ChangeProposal.kind` è una stringa validata contro un registro di
   validator, non un `Literal` chiuso a un solo valore.
2. Le proposte hanno un `batch_id` opzionale (nullable) che le raggruppa —
   il Node Advisor conversazionale lo lascia `None`, un futuro producer
   batch (il committee) lo popola per raggruppare le proposte multi-nodo di
   una singola run.
3. `ModelProvenance` distingue esplicitamente `producer_kind`
   (`"interactive_chat"` vs `"scheduled_batch"`) e `producer_id` libero.
4. Il service layer (`NodeContextService`, `AdvisorJobService`,
   `ProposalService`, introdotti in Fase 3) riceve l'identità del chiamante
   come parametro esplicito, mai da una sessione HTTP implicita — così resta
   richiamabile da un job schedulato (LazyPulse) tanto quanto da una
   richiesta REST.

Motivazione: retrofittare questi campi dopo aver già scritto schema e
validator del Node Advisor richiederebbe una migrazione di schema e la
riscrittura dei service layer; fissarli ora costa quattro campi/parametri in
più, oggi inutilizzati dal solo Node Advisor ma già pronti per il secondo
producer.

## Decisione 4 — Semantica del backtest, dichiarata esplicitamente

Tre modalità nominate senza ambiguità (§6.4 del piano finalizzato):
`current_estimate_counterfactual` (obbligatoria, causalmente corretta),
`historical_static_view_sensitivity` (opzionale, marcata
`uses_future_formulated_view=true`, mai presentata come backtest della
proposta), `point_in_time_view_backtest` (futura, fuori MVP finché
registry/crawler non garantiscono versioni point-in-time delle evidenze).
Questa distinzione è un invariante di prodotto, non solo tecnico: previene
che un backtest con view formulata con informazioni correnti venga letto
come prova out-of-sample.

## Decisione 5 — Fase 4: una singola chiamata LLM strutturata, non un `Plan` a 10 step

Il piano finalizzato (§8.2) descrive la preparazione di una proposta come un
`Plan` LazyBridge a 10 step dichiarativi. L'implementazione (`project/advisor/agent.py`)
usa invece un solo `Agent` con `output=AdvisorTurnResult` (schema Pydantic
vincolato: `route: "explain"|"propose"`, `message`, `proposed_views`) seguito
da controllo di flusso Python semplice (if/else), non un oggetto `Plan`
multi-step.

Motivazione: i 10 step del piano descrivono l'invariante che conta — l'LLM
propone tramite uno schema vincolato, ogni passo successivo è validazione,
calcolo e persistenza deterministici, nessun tool dell'LLM può mai scrivere
nulla — non un numero di step imposto. Una singola chiamata Agent con output
strutturato mantiene esattamente lo stesso invariante (dimostrato dai test:
`route="explain"` non tocca mai `validate_view_set`/counterfactual/
create_proposal; una view malevola prodotta dall'LLM viene comunque
rifiutata dal validator deterministico) con una frazione della complessità
implementativa. `clarify_or_continue` (step 2 del piano) è collassato nel
campo `route` dello stesso output strutturato; `retrieve_evidence` (step 3)
è delegato al normale tool-calling dell'Agent all'interno della sua unica
chiamata, non un passo separato.

## Conseguenze

- Ogni contratto Pydantic scritto in Fase 0 include `kind: str` aperto e
  `batch_id`/`producer_kind` fin da subito (nessun campo aggiunto in una
  migration successiva per questo scopo specifico).
- Il resolver/servizi di Fase 2-3 devono essere scritti come funzioni pure
  con parametri espliciti, non come metodi legati a un contesto di richiesta
  HTTP.
- Qualunque futura richiesta di "backtest della proposta" nella UI deve
  dichiarare quale delle tre modalità sta mostrando.
- Se una fase successiva ha davvero bisogno di step multipli con routing
  complesso (es. un ciclo di chiarimento multi-turno), la Decisione 5 va
  rivista con un `Plan` reale — non è preclusa, solo non necessaria per
  l'MVP di Fase 4.
- (Fase 5) `run_advisor_turn` è stato scritto e testato in Fase 4 ma cablato
  nel worker/API/UI live solo in Fase 5, dopo essere stato individuato come
  gap durante il Deep Audit — la Decisione 3 punto 4 (identità del chiamante
  come parametro esplicito, mai da un contesto HTTP implicito) ha reso quel
  cablaggio un routing aggiuntivo in `project/advisor/api.py`/`services.py`
  (nuovo job kind `advisor_turn` accanto a `fixture_proposal`), non una
  riscrittura del service layer — la scelta architetturale ha retto al primo
  vero utilizzo end-to-end.
