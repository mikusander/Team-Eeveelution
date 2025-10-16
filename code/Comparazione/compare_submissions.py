import pandas as pd

def analyze_differences(file1_path: str, file2_path: str, model1_name: str = 'CatBoost', model2_name: str = 'LightGBM'):
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        print(f"ERRORE: File non trovato. Controlla il percorso: {e.filename}")
        print("Assicurati che i file CSV siano nella stessa cartella dello script o modifica i percorsi.")
        return

    merged_df = pd.merge(
        df1, 
        df2, 
        on='battle_id', 
        suffixes=(f'_{model1_name}', f'_{model2_name}')
    )

    prediction_col1 = f'player_won_{model1_name}'
    prediction_col2 = f'player_won_{model2_name}'
    
    disagreements = merged_df[merged_df[prediction_col1] != merged_df[prediction_col2]].copy()

    total_predictions = len(merged_df)
    num_disagreements = len(disagreements)
    agreement_percentage = (1 - (num_disagreements / total_predictions)) * 100
    disagreement_percentage = 100 - agreement_percentage

    print("\n" + "="*60)
    print("---      ANALISI DELLE DIFFERENZE TRA I MODELLI      ---")
    print("="*60)
    print(f"File 1 ({model1_name}): {file1_path}")
    print(f"File 2 ({model2_name}): {file2_path}")
    print("-"*60)
    print(f"Numero totale di previsioni analizzate: {total_predictions}")
    print(f"Previsioni su cui i modelli sono d'accordo: {total_predictions - num_disagreements}")
    print(f"Previsioni su cui i modelli NON sono d'accordo: {num_disagreements}")
    print("-"*60)
    print(f"Percentuale di Accordo:     {agreement_percentage:.2f}%")
    print(f"Percentuale di Disaccordo:  {disagreement_percentage:.2f}%  <-- POTENZIALE PER L'ENSEMBLE!")
    print("="*60)

    if num_disagreements > 0:
        print("\nEsempi di battaglie su cui i modelli non sono d'accordo:")
        disagreements['Voto'] = disagreements.apply(
            lambda row: f"{model1_name}: {row[prediction_col1]}, {model2_name}: {row[prediction_col2]}", 
            axis=1
        )
        print(disagreements[['battle_id', 'Voto']].head(10).to_string(index=False))
    else:
        print("\nIncredibile! I due modelli sono perfettamente d'accordo su tutte le previsioni.")

print("catboost vs lightgbm")
analyze_differences('/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Comparazione/submission_submitted - 08113.csv', '/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Comparazione/submission.csv')
print("catboost vs ensemble 60 - 40")
analyze_differences('/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Comparazione/submission_submitted - 08113.csv', '/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Comparazione/submission_GRAND_ENSEMBLE.csv')
print("catboost vs ensemble 50 - 50")
analyze_differences('/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Comparazione/submission_submitted - 08113.csv', '/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Comparazione/submission_GRAND_ENSEMBLE 2.csv')



    
