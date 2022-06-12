import pandas as pd
import json

def loadDf(inputPath: str) -> pd.DataFrame:
    df = pd.DataFrame()
    with open(inputPath, encoding='utf-8') as f:
        for line in f:
            lineDf = pd.DataFrame(json.loads(line))
            df = pd.concat([df, lineDf])
    df = df.reset_index(drop=True)

    return df

def createDocDictionary(input: str, output: str) -> None:
    """
    input: Archivo de entrada
    output: Ruta para archivos de salida
    """
    pan = loadDf(input)

    uniquePanDocs = pd.DataFrame(pan.pair.unique()).reset_index().rename(columns={'index':'idtext', 0:'pair'})

    testId2text = pd.merge(pan, uniquePanDocs, on='pair').sort_values('id').drop(columns=['pair','discourse_types']) \
        .reset_index(drop=True)

    testId2text.to_pickle(output + '/PanTest.plk')
    # uniquePanDocs.to_pickle(output + '/UniquePanDocs.plk')

    with open(output + '/PanUniqueDocs.jsonl', 'w', encoding='utf-8') as f:
        for idx, row in uniquePanDocs.iterrows():
            f.write(json.dumps({'id':row.idtext, 'text':row.pair}) + "\n")