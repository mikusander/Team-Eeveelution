# Team Eeveelution: Pokémon Battle Prediction

Questo repository contiene il codice per una pipeline di machine learning progettata per predire l'esito delle battaglie Pokémon.

Il framework è costruito attorno a un modello **CatBoost**, ma è esplicitamente progettato per produrre output (file `.npy`) adatti per l'**ensembling** (sovrapposizione) con altri modelli, come LightGBM o Reti Neurali.

## Guida al Repository

Di seguito è riportata una breve descrizione dei file e delle cartelle principali e del loro scopo.

### Cartelle Principali

* **/Input**:
    Contiene i dati grezzi della competizione, `train.jsonl` e `test.jsonl`.

* **/CatBoost_Data_Pipeline**:
    Questa è la cartella di output per tutti i file `.csv` intermedi. Contiene i dati processati, le feature ingegnerizzate e i set di dati suddivisi (train/validation/holdout) pronti per essere usati dal modello CatBoost.

* **/CatBoost_Model_Outputs**:
    Salva tutti i risultati dell'analisi del modello CatBoost. Include grafici diagnostici (es. Feature Importance, ROC/AUC, Matrice di Confusione), report di classificazione e i file `.json` con i parametri e le iterazioni ottimali.

* **/OOF_Predictions**:
    Cartella cruciale per l'ensembling. Contiene le previsioni finali sia per il set di training (Out-of-Fold) che per il set di test, salvate come file `.npy`. Questi array possono essere usati come feature di "livello 2" per un modello di stacking.

* **/Submissions**:
    Contiene il file `submission.csv` finale (basato sulle probabilità) generato dalla pipeline CatBoost, pronto per essere caricato.

### File Principali

* `utils_catboost.py`:
    Il "cervello" dell'intera pipeline. Questo file contiene **tutta la logica**, le funzioni di utility, i percorsi e le definizioni di ogni fase (dalla 00 alla 11) per il caricamento dei dati, la feature engineering, la validazione e l'addestramento.

* `run_training_pipeline.py` (Esempio di Esecuzione):
    Script principale utilizzato per eseguire le fasi finali (Fase 5 e 6) della pipeline. Carica i dati processati, gestisce l'addestramento e la validazione del modello e genera tutti i file di output.

## Workflow e Configurazione

1.  **Fase Dati (Fasi 0-4)**: La logica in `utils_catboost.py` (da `run_00a_load_data` a `run_09_data_splitter`) viene eseguita per prima. Converte i file grezzi `.jsonl` (dalla cartella `/Input`) nei file `.csv` finali (nella cartella `/CatBoost_Data_Pipeline`).

2.  **Fase Training (Fasi 5-6)**: Lo script `run_training_pipeline.py` utilizza i dati processati per addestrare e validare il modello CatBoost, salvando i risultati e i file finali nelle rispettive cartelle.

### Configurazione Chiave

All'interno di `run_training_pipeline.py` (o script simile) è possibile controllare il comportamento della pipeline:

* **`RUN_GRID_SEARCH`**:
    * `True`: Esegue una `GridSearchCV` completa per trovare nuovi iperparametri ottimali (molto lento).
    * `False`: Carica gli iperparametri esistenti da `best_catboost_params.json`.

* **`RECALCULATE_ITERATIONS`**:
    * `True`: Esegue l'early stopping per trovare il numero ottimale di iterazioni (alberi) e lo salva.
    * `False`: Carica il numero di iterazioni già salvato da `best_catboost_iteration.json`.
