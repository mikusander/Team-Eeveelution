# Team Eeveelution: Pokémon Battle Prediction

Questo repository contiene il codice per una pipeline di machine learning progettata per predire l'esito delle battaglie Pokémon.

Il framework è costruito attorno a un modello **CatBoost**, ma è esplicitamente progettato per produrre output (file `.npy`) adatti per l'**ensembling** (sovrapposizione) con altri modelli, cioè LightGBM e XGBoost.

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
    Cartella cruciale per l'ensembling. Contiene le previsioni finali sia per il set di training (Out-of-Fold) che per il set di test, salvate come file `.npy`. 

* **/Submissions**:
    Contiene i file di `submission.csv`, sia quelli utili per il blending (basati sulle probabilità) sia quelli finali blended e stacking.

### File Principali

* `utils_*.py`:
    Il "cervello" dell'intera pipeline. Questi file contengono **tutta la logica**, le funzioni di utility, i percorsi e le definizioni di ogni fase per il caricamento dei dati, la feature engineering, la validazione, l'addestramento e il blending/stacking. 

