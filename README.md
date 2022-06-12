Para generar respuestas
```
python3 PredictionsTest.py -i testinput -o testoutput/
```

Para evaluar 
```
python3 pan22_verif_evaluator.py -i pan22-authorship-verification-training-dataset/particiones/test_truth.jsonl -a testoutput/ -o testoutput/
```