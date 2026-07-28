# Ottimizzatore gerarchico V2: esempio step by step

Questo documento descrive esattamente cosa esegue il motore V2 usando il
modello Tree Studio `Global allocation_3pct_TEV_father`. I nomi, i ticker e
l'ordine dei solve corrispondono al JSON e al codice corrente.

## 1. Albero dell'esempio

```text
Global allocation
|- Equity, rappresentata nel root da ACWI
|  |- ticker diretti: VGK, EWJ, VWO, ACWI
|  `- SPY sleeve, rappresentata in Equity da SPY
|     `- XLB, XLE, XLF, XLI, XLP, XLU, XLV, XLY, SPY
|- Bonds, rappresentata nel root da AGG
|  `- SHY, IEF, TLT, CEMB, EMLC, EUHY, AGG
`- Commodities, rappresentata nel root da DBC
   `- DBA, DBB, GLD, UGA, DBC
```

Il benchmark globale `B0` e:

```text
70% ACWI + 30% AGG + 0% DBC
```

Configurazione locale dei nodi:

| Nodo | Obiettivo configurato | Target volatilita | TEV |
|---|---|---|---|
| Global allocation | max return | volatilita di B0 | massimo 3% vs B0 |
| Equity | max return | volatilita di ACWI | massimo 3% vs ACWI |
| SPY sleeve | min risk | volatilita di SPY | massimo 3% vs SPY |
| Bonds | max return | volatilita di AGG | massimo 3% vs AGG |
| Commodities | min risk | volatilita di DBC | massimo 3% vs DBC |

Con un target di volatilita attivo, il problema effettivo ordina le soluzioni
ammissibili per rendimento atteso massimo al target. Quindi `min risk` resta
l'obiettivo configurato, ma non distingue due portafogli che devono avere la
stessa volatilita: in quel caso il rendimento atteso e il criterio economico
secondario.

## 2. Cosa fa ogni solve locale

Ogni ottimizzatore riceve esclusivamente i ticker o le serie sintetiche del
nodo corrente. Father e benchmark sono riferimenti esterni e non vengono
aggiunti automaticamente ai candidati.

Per ogni finestra di stima il solve locale:

1. allinea le serie sulle stesse osservazioni complete;
2. stima covarianza e rendimento atteso — mai sui valori campionari grezzi.
   La covarianza e sempre Ledoit-Wolf shrunk. Il rendimento atteso segue
   `constraints.mean_estimator` (default `auto`): equilibrium (reverse-optimized
   dal riferimento padre/benchmark del nodo) quando quel riferimento e pienamente
   investito — il caso di ogni nodo in questo esempio, avendo tutti un target di
   volatilita relativo — altrimenti Bayes-Stein verso la media di gruppo. Vedi
   [`hierarchical-v2.md`](hierarchical-v2.md#moment-estimation) per le altre
   opzioni (`james_stein`, `bodnar_okhrin`, `empirical`);
3. applica pieno investimento, long-only, minimi, massimi e per-asset cap;
4. calcola target-vol, vol-cap e TEV in unita periodiche partendo dai valori
   annualizzati mostrati nello Studio;
5. tenta target-vol e TEV esatti;
6. se non sono compatibili con l'universo dichiarato, minimizza la somma degli
   scostamenti normalizzati: distanza dal target-vol e solo l'eccesso sopra il
   limite TEV;
7. a parita della minima violazione applica l'obiettivo economico;
8. ricalcola indipendentemente somma pesi, bounds, volatilita e TEV;
9. registra `raggiunto`, `entro limite` o `piu vicino ottenibile`.

Il volatility cap resta sempre hard. Se pieno investimento, bounds o cap sono
incompatibili, il solve fallisce: questi vincoli non vengono rilassati.

## 3. Approccio Flat: Forward disattivato

Flat produce un portafoglio globale diretto. Non moltiplica i pesi lungo
l'albero.

### Step 3.1 - Diagnostica dei nodi

V2 esegue prima gli stessi solve locali del Forward, descritti nella sezione 4,
per mostrare cosa farebbe ogni sleeve. Questi pesi sono diagnostici e non
determinano il risultato finale Flat.

### Step 3.2 - Universo terminale globale

Viene costruito un unico universo senza duplicati:

```text
VGK, EWJ, VWO, ACWI,
XLB, XLE, XLF, XLI, XLP, XLU, XLV, XLY, SPY,
SHY, IEF, TLT, CEMB, EMLC, EUHY, AGG,
DBA, DBB, GLD, UGA, DBC
```

Sono 25 strumenti. Non esistono `SPY_SYNTH`, `ACWI_SYNTH`, `AGG_SYNTH` o
`DBC_SYNTH` in questo solve.

### Step 3.3 - Ottimizzazione globale Flat

Un solo ottimizzatore calcola direttamente:

```text
g_VGK, g_EWJ, ..., g_SPY, ..., g_AGG, ..., g_DBC
```

con somma pari a 100%. Usa l'obiettivo e i vincoli del root: target-vol di B0 e
TEV 3% vs B0 nell'esempio.

### Step 3.4 - Risultato Flat

Il peso finale di ogni ETF e semplicemente `g_ticker`. Per esempio:

```text
peso_finale(XLB) = g_XLB
peso_finale(VGK) = g_VGK
peso_finale(AGG) = g_AGG
```

Non esiste un peso intermedio di ACWI o SPY che moltiplica questi valori.

## 4. Approccio Forward

Forward mantiene i proxy nei livelli superiori e ricompone i pesi lungo ogni
ramo. L'ordine dei solve e root-first.

### Step 4.1 - Root: Global allocation

Il root non contiene ticker diretti nel JSON. I suoi tre figli forniscono i
proxy candidati:

```text
ACWI, AGG, DBC
```

Il solve produce:

```text
r_ACWI + r_AGG + r_DBC = 100%
```

Target-vol e TEV sono calcolati contro `B0 = 70% ACWI + 30% AGG`.
Con questi pesi viene anche congelata la serie Forward del root:

```text
ROOT_FWD(t) = r_ACWI*R_ACWI(t) + r_AGG*R_AGG(t) + r_DBC*R_DBC(t)
```

### Step 4.2 - Equity

Equity viene ottimizzata sui suoi ticker diretti e sul proxy del figlio SPY:

```text
VGK, EWJ, VWO, ACWI, SPY
```

SPY e ancora SPY, non i settori. Il solve produce:

```text
e_VGK + e_EWJ + e_VWO + e_ACWI + e_SPY = 100%
```

Target-vol e TEV sono confrontati con la serie esterna ACWI. Se ACWI viene
rimosso dai candidati, resta il riferimento ma non riceve un peso.

### Step 4.3 - SPY sleeve

Solo adesso viene ottimizzato il livello settoriale:

```text
XLB, XLE, XLF, XLI, XLP, XLU, XLV, XLY, SPY
```

Il solve produce `s_XLB, ..., s_XLY, s_SPY`, con target-vol e TEV vs SPY.

### Step 4.4 - Bonds

Il solve usa:

```text
SHY, IEF, TLT, CEMB, EMLC, EUHY, AGG
```

e produce `b_SHY, ..., b_EUHY, b_AGG`, con target-vol e TEV vs AGG.

### Step 4.5 - Commodities

Il solve usa:

```text
DBA, DBB, GLD, UGA, DBC
```

e produce `c_DBA, c_DBB, c_GLD, c_UGA, c_DBC`, con target-vol e TEV vs DBC.

### Step 4.6 - Composizione Forward

I pesi finali sono prodotti lungo la catena:

```text
peso(VGK)  = r_ACWI * e_VGK
peso(ACWI) = r_ACWI * e_ACWI

peso(XLB) = r_ACWI * e_SPY * s_XLB
peso(XLE) = r_ACWI * e_SPY * s_XLE
peso(SPY) = r_ACWI * e_SPY * s_SPY

peso(IEF) = r_AGG * b_IEF
peso(AGG) = r_AGG * b_AGG

peso(GLD) = r_DBC * c_GLD
peso(DBC) = r_DBC * c_DBC
```

La stessa regola vale per tutti gli altri ticker. I settori compaiono nel
portafoglio finale, ma non nel solve Equity: entrano tramite il peso `e_SPY`.

## 5. Approccio Forward + Backward

Questa modalita esegue prima tutto il Forward della sezione 4 e ne conserva
pesi, serie e audit. Poi risale l'albero dal livello piu profondo.

### Step 5.1 - Forward completo e congelato

Vengono calcolati `r`, `e`, `s`, `b` e `c`. Questi risultati sono mostrati
nello Studio come `Passaggio Forward congelato`.

### Step 5.2 - Costruzione di SPY_SYNTH

Sulla stessa finestra storica di stima del rebalance corrente:

```text
SPY_SYNTH(t) =
    s_XLB*R_XLB(t) + s_XLE*R_XLE(t) + ... +
    s_XLY*R_XLY(t) + s_SPY*R_SPY(t)
```

Si usano i pesi appena ottimizzati e i rendimenti precedenti nella finestra
corrente. Non si usano pesi del rebalance precedente e non si usano dati OOS.

### Step 5.3 - Secondo solve Equity

SPY viene sostituito da SPY_SYNTH. I candidati diventano:

```text
VGK, EWJ, VWO, ACWI, SPY_SYNTH
```

Il riferimento di target-vol e TEV resta ACWI. Il solve produce i pesi backward:

```text
eb_VGK, eb_EWJ, eb_VWO, eb_ACWI, eb_SPY_SYNTH
```

`eb_SPY_SYNTH` viene poi ricondotto agli ETF della SPY sleeve usando i pesi `s`.

### Step 5.4 - Costruzione di ACWI_SYNTH

La serie Equity backward viene ricostruita come:

```text
ACWI_SYNTH(t) =
    eb_VGK*R_VGK(t) + eb_EWJ*R_EWJ(t) +
    eb_VWO*R_VWO(t) + eb_ACWI*R_ACWI(t) +
    eb_SPY_SYNTH*SPY_SYNTH(t)
```

### Step 5.5 - AGG_SYNTH e DBC_SYNTH

Bonds e Commodities sono foglie: non hanno figli da sostituire e non vengono
risolte una seconda volta. I loro pesi Forward costruiscono:

```text
AGG_SYNTH(t) = b_SHY*R_SHY(t) + ... + b_AGG*R_AGG(t)
DBC_SYNTH(t) = c_DBA*R_DBA(t) + ... + c_DBC*R_DBC(t)
```

### Step 5.6 - Secondo solve del root

I proxy del root vengono sostituiti dalle serie appena costruite:

```text
ACWI_SYNTH, AGG_SYNTH, DBC_SYNTH
```

Il benchmark di riferimento non viene sostituito. Target-vol, volatility cap e
TEV restano ancorati al benchmark ufficiale raw:

```text
B0_RAW(t) = 70%*R_ACWI(t) + 30%*R_AGG(t)
```

Il root backward produce:

```text
rb_ACWI_SYNTH + rb_AGG_SYNTH + rb_DBC_SYNTH = 100%
```

Separatamente viene costruito un portafoglio diagnostico che mantiene i pesi
strategici di B0 ma implementa le asset class tramite le sleeve:

```text
B0_SYNTH(t) = 70%*ACWI_SYNTH(t) + 30%*AGG_SYNTH(t)
```

`B0_SYNTH` non entra nei vincoli e non sostituisce il benchmark. Serve a
separare l'effetto dell'implementazione delle sleeve dall'effetto della
riottimizzazione dei pesi del root.

### Step 5.7 - Composizione finale backward

Esempi di pesi terminali:

```text
peso(VGK) = rb_ACWI_SYNTH * eb_VGK

peso(XLB) = rb_ACWI_SYNTH * eb_SPY_SYNTH * s_XLB
peso(SPY) = rb_ACWI_SYNTH * eb_SPY_SYNTH * s_SPY

peso(IEF) = rb_AGG_SYNTH * b_IEF
peso(AGG) = rb_AGG_SYNTH * b_AGG

peso(GLD) = rb_DBC_SYNTH * c_GLD
peso(DBC) = rb_DBC_SYNTH * c_DBC
```

Il report distingue `FORWARD_FINAL`, `B0_SYNTH` e `FINAL`: il primo usa i pesi
della sezione 4, il secondo mantiene i pesi strategici di B0 sulle sleeve, il
terzo usa i pesi ottenuti dopo la risalita backward. Tutti i confronti di
mandato restano contro `B0_RAW`.

Nello Studio, la tabella e il grafico dei pesi di ciascun nodo mostrano sempre
l'universo locale del relativo solve: Equity mostra `SPY` nel Forward e
`SPY_SYNTH` nel backward. I settori appaiono nel solve `SPY sleeve` e nei pesi
terminali finali, non nei pesi dell'ottimizzatore Equity. La performance di
Equity resta invece valorizzata sui terminali composti, perche e la strategia
effettivamente detenibile.

## 6. Caso senza father tra i candidati

Esempio: Equity contiene solo `VGK, EWJ, VWO, SPY`, mentre ACWI resta il suo
proxy di riferimento.

Il solve Equity:

1. non aggiunge ACWI alla matrice dei candidati;
2. calcola comunque volatilita target e TEV sulla serie ACWI esterna;
3. tenta target-vol e TEV richiesti con `VGK, EWJ, VWO, SPY`;
4. se non sono raggiungibili, restituisce il punto con minima violazione;
5. mostra nell'audit `piu vicino ottenibile`.

Lo stesso vale rimuovendo SPY dalla SPY sleeve, AGG da Bonds o DBC da
Commodities. Il riferimento continua a esistere, ma non puo ricevere un peso.

## 7. Walk-forward backtest

Per ogni modalita e per ogni rebalance mensile:

1. seleziona le 104 osservazioni settimanali precedenti;
2. esegue tutti i solve richiesti dalla modalita scelta;
3. salva pesi locali, composti, riferimenti e audit del fold;
4. applica i pesi solo al periodo giornaliero successivo;
5. lascia derivare i pesi giornalmente con i rendimenti;
6. al rebalance successivo applica i nuovi target e gli eventuali costi;
7. calcola tutte le serie sullo stesso calendario OOS.

Il report confronta `FINAL` con `B0`, `FORWARD_FINAL` con `B0` nella modalita
backward, e ogni `NODE:<nome>` con il suo diretto `FATHER:<nome>`.

## 8. Modalita iterativa

La modalita iterativa non e ancora implementata nel V2. Non viene simulata
riutilizzando Forward+Backward. Sara attivata solo con un coordinatore separato,
regole di convergenza, storia delle iterazioni e gate esterno dedicato.

Il suo contratto manterra immutabili `SPY`, `ACWI`, `AGG`, `DBC` e `B0_RAW`
come riferimenti economici. Le serie `_SYNTH` cambieranno soltanto sul lato dei
candidati. La distanza dalla precedente iterazione sara una diagnostica di
convergenza, non un nuovo benchmark per target-vol o TEV.
