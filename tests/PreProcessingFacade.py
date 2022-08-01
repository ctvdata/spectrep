import sys
sys.path.append('..')
from spectraltrep.preProcessing import PreProcessingFacade
import time

def main():
    inputPath = '../data/data_sample.jsonl' # input file
    outputPath = './outputs/SalidaPipelinePreProcesamiento.jsonl' # output file
    preProcessingType = ['lex','syn','sem'] # Applicable preprocessing type ['lex', 'syn', 'sem']
    numThreads = 1 # Number of preprocessing threads
    batchSize = 10 # Number of documents delivered by Corpus Reader per batch
    sortedOutput = True

    init = time.time()
    ppf = PreProcessingFacade()
    ppf.preProcess(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput)
    print('%d Threads, %.02f seconds' % (numThreads,time.time() - init))

if __name__ == '__main__':
    main()