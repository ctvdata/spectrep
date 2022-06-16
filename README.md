Para generar Diccionario de documentos

```
python3 DocDictionary.py -a <carpeta del corpus>
```

Para entrenar la red 
```
python3 NeuralNetwork.py -a <carpeta del corpus>
```

Para generar respuestas
```
python3 PredictionsTest.py -i <carpeta del corpus>/testinput -o testoutput/ -a <carpeta del corpus>
```

Para evaluar 
```
python3 verif_evaluator.py -i <carpeta del corpus>/testinput/truth.jsonl -a testoutput/ -o testoutput/
```

Resultados

| Corpus | Train size | Test size | F1    | AUC   | Brier | c@1   | f_05_u | overall|
|--------|------------|-----------|-------|-------|-------|-------|--------|--------|
| PAN15  | 100        | 500       | 0.559 | 0.567 | 0.636 | 0.562 | 0.561  | 0.577  |
| PAN22  | 15732      | 1070      | 0.611 | 0.592 | 0.756 | 0.593 | 0.593  | 0.63   |
