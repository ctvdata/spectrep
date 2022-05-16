import argparse
import DocDictionaryTest as dd
import TextTransformTest as tt
import json
import pandas as pd
import numpy as np
from myGenerator import TestGenerator
from keras.models import load_model
from CustomLayers import ResidualLayer, AbsoluteResidual

def spectraLoader(inputPath: str) -> pd.DataFrame:
    df = pd.DataFrame()
    with open(inputPath, encoding='utf-8') as f:
        for line in f:
            jsonline = json.loads(line)
            lineDf = pd.DataFrame({'id':jsonline['id'], 'spectra':[np.array(jsonline['spectra'])]})
            df = pd.concat([df, lineDf])
    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    print("Leyendo archivo de configuracion")
    with open('config.json') as f:
        cfg = json.loads(f.read())

    parser = argparse.ArgumentParser(description='Siamese network AA@PAN')
    parser.add_argument('-i', type=str,
        help='Path to the input directory')
    parser.add_argument('-o', type=str,
        help='Path to the output directory')
    args = parser.parse_args()

    if not args.i:
        raise ValueError('The input directory is required')
    if not args.o:
        raise ValueError('The output directory is required')

    # Transformacion de texto
    
    print("Inicia la transformacion de texto")
    print("Creando diccionario de documentos unicos")
    dd.createDocDictionary(args.i + '/pairs.jsonl', cfg['tmp'])

    print("Realizando preprocesamiento")
    tt.preProcessStage(cfg['tmp'] + '/PanUniqueDocs.jsonl', cfg['tmp'] + '/panPreprocessed.jsonl')

    print("Extrayendo vectores de caracteristicas")
    tt.featureExtractionStage(cfg['tmp'] + '/panPreprocessed_lex.jsonl',
                                cfg['tmp'] + '/LexicVectors.jsonl',
                                cfg['LexicVectorizer'],
                                cfg['tmp'] + '/panPreprocessed_syn.jsonl',
                                cfg['tmp'] + '/SyntacticVectors.jsonl',
                                cfg['SyntacticVectorizer'],
                                cfg['tmp'] + '/panPreprocessed_sem.jsonl',
                                cfg['tmp'] + '/SemanticVectors.jsonl',
                                cfg['SemanticVectorizer'])

    print("Obteniendo espectros de contenido")
    tt.spaceMappingStage(cfg['tmp'] + '/LexicVectors.jsonl',
                            cfg['tmp'] + '/LexicSpectra.jsonl',
                            cfg['LexicProjector'],
                            cfg['tmp'] + '/SyntacticVectors.jsonl',
                            cfg['tmp'] + '/SyntacticSpectra.jsonl',
                            cfg['SyntacticProjector'],
                            cfg['tmp'] + '/SemanticVectors.jsonl',
                            cfg['tmp'] + '/SemanticSpectra.jsonl',
                            cfg['SemanticProjector'])

    print("Consolidando capas")
    tt.layerConsolidation(cfg['tmp'] + '/LexicSpectra.jsonl',
                            cfg['tmp'] + '/SyntacticSpectra.jsonl',
                            cfg['tmp'] + '/SemanticSpectra.jsonl',
                            cfg['tmp'] + '/FullSpectra.jsonl')

    # Clasificacion
    print("Cargando modelo")
    model = load_model(cfg['Classifier'], custom_objects={"Residual": ResidualLayer, "AbsoluteResidual": AbsoluteResidual})

    test = pd.read_pickle(cfg['tmp'] + "/PanTest.plk")
    full_spectra = spectraLoader(cfg['tmp'] + '/FullSpectra.jsonl')
    list_IDs = test.id.unique().tolist()

    print("Realizando predicciones")
    params = {'dim': (1200,),
            'batch_size': 32}
    x = TestGenerator(list_IDs, test, full_spectra, **params)
    predictions = model.predict(x)

    with open(args.o + "/answers.jsonl", "w", encoding="utf8") as f:
        for id, prediction in zip(list_IDs, predictions):
            answer = {"id": id, "value":prediction}
            f.write(json.dumps(answer) + "\n")