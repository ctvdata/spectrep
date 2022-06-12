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

train = loadDf('pan22-authorship-verification-training-dataset/particiones/train.jsonl')
val = loadDf('pan22-authorship-verification-training-dataset/particiones/val.jsonl')
test = loadDf('pan22-authorship-verification-training-dataset/particiones/test.jsonl')

pan = pd.concat([train, val, test])
uniquePanDocs = pd.DataFrame(pan.pair.unique()).reset_index().rename(columns={'index':'idtext', 0:'pair'})

trainId2text = pd.merge(train, uniquePanDocs, on='pair').sort_values('id').drop(columns=['pair','discourse_types']) \
    .reset_index(drop=True)

valId2text = pd.merge(val, uniquePanDocs, on='pair').sort_values('id').drop(columns=['pair','discourse_types']) \
    .reset_index(drop=True)

testId2text = pd.merge(test, uniquePanDocs, on='pair').sort_values('id').drop(columns=['pair','discourse_types']) \
    .reset_index(drop=True)

trainId2text.to_pickle('pan22-authorship-verification-training-dataset/particionesXid/PanTrain.plk')
valId2text.to_pickle('pan22-authorship-verification-training-dataset/particionesXid/PanVal.plk')
testId2text.to_pickle('pan22-authorship-verification-training-dataset/particionesXid/PanTest.plk')
uniquePanDocs.to_pickle('pan22-authorship-verification-training-dataset/particionesXid/UniquePanDocs.plk')

with open('pan22-authorship-verification-training-dataset/particionesXid/PanUniqueDocs.jsonl', 'w', encoding='utf-8') as f:
    for idx, row in uniquePanDocs.iterrows():
        f.write(json.dumps({'id':row.idtext, 'text':row.pair}) + "\n")