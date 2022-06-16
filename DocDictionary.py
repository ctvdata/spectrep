import argparse
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

parser = argparse.ArgumentParser(description='Generate Document Diccionary')
parser.add_argument('-a', type=str, help='Path name to dataset JSONL file')
args = parser.parse_args()

if not args.a:
    print('ERROR: Corpus is required')
    parser.exit(1)


train = loadDf(args.a + '/particiones/train.jsonl')
val = loadDf(args.a + '/particiones/val.jsonl')
test = loadDf(args.a + '/particiones/test.jsonl')

pan = pd.concat([train, val, test])
uniquePanDocs = pd.DataFrame(pan.pair.unique()).reset_index().rename(columns={'index':'idtext', 0:'pair'})

trainId2text = pd.merge(train, uniquePanDocs, on='pair').sort_values('id').drop(columns=['pair']) \
    .reset_index(drop=True)

valId2text = pd.merge(val, uniquePanDocs, on='pair').sort_values('id').drop(columns=['pair']) \
    .reset_index(drop=True)

testId2text = pd.merge(test, uniquePanDocs, on='pair').sort_values('id').drop(columns=['pair']) \
    .reset_index(drop=True)

trainId2text.to_pickle(args.a + '/particionesXid/PanTrain.plk')
valId2text.to_pickle(args.a + '/particionesXid/PanVal.plk')
testId2text.to_pickle(args.a + '/particionesXid/PanTest.plk')
uniquePanDocs.to_pickle(args.a + '/particionesXid/UniquePanDocs.plk')

with open(args.a + '/particionesXid/PanUniqueDocs.jsonl', 'w', encoding='utf-8') as f:
    for idx, row in uniquePanDocs.iterrows():
        f.write(json.dumps({'id':row.idtext, 'text':row.pair}) + "\n")