import time
import sys
sys.path.append('..')
from spectraltrep.preProcessing import PreProcessingFacade
from spectraltrep.featureExtraction import VectorizerFactory, Doc2VecCorpusReader
from spectraltrep.utils import CorpusReader, DocumentSink
from spectraltrep.spaceUnification import VectorReader, Projector
from spectraltrep.utils import DocumentSink

def preProcessStage():
    """Pre-processing stage"""

    print("Pre-processing stage")

    inputPath = 'data_sample.jsonl' # input file
    outputPath = 'outputs/Preprocessed.jsonl' # output file
    preProcessingType = ['lex','syn','sem'] # Applicable preprocessing type ['lex', 'syn', 'sem']
    numThreads = 1 # Number of preprocessing threads
    batchSize = 500 # Number of documents delivered by CorpusReader per batch
    sortedOutput = True

    ppf = PreProcessingFacade()
    ppf.preProcess(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput)

    del(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput, ppf)

def featureExtractionStage() -> int:
    """Feature Extraction Stage"""

    print("Feature Extraction Stage")

    # Extraction of lexical vectors

    print('Extraction of lexical vectors')
    cr = CorpusReader('outputs/Preprocessed_lex.jsonl', 500)
    vw = DocumentSink('outputs/LexicVectors.jsonl', False)
    vf = VectorizerFactory()
    lv = vf.createLexicVectorizer(vw, cr)

    print('Training model - lexical')
    lv.fit()

    print('Performing vectorization')
    lv.transform()

    print('Save results')
    lv.saveModel('outputs/lexicModel.json')
    vw.saveCorpus()

    lexicVectorLenght = lv.model.vocabularyLength

    del(cr, vw, vf, lv)

    # Extraction of syntactic vectors

    print('Extraction of syntactic vectors')
    cr = Doc2VecCorpusReader('outputs/Preprocessed_syn.jsonl')
    vw = DocumentSink('outputs/SyntacticVectors.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw, cr, 300)

    print('Training model - syntactic')
    sv.fit()

    print('Performing vectorization')
    sv.transform()

    print('Save results')
    sv.saveModel('outputs/dv2SynModel')
    vw.saveCorpus()

    del(cr, vw, vf, sv)

    # Semantic vector extraction

    print('Semantic vector extraction')
    cr = Doc2VecCorpusReader('outputs/Preprocessed_sem.jsonl')
    vw = DocumentSink('outputs/SemanticVectors.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw, cr, 300)

    print('Training model - semantic')
    sv.fit()

    print('Performing vectorization')
    sv.transform()

    print('Save results')
    sv.saveModel('outputs/dv2SemModel')
    vw.saveCorpus()

    del(cr, vw, vf, sv)

    return lexicVectorLenght

def spaceMappingStage(lexicVectorLenght: int):
    """Space unification"""

    print("Space unification stage")

    # Obtaining lexical spectra

    vr = VectorReader('outputs/LexicVectors.jsonl')
    proj = Projector(20,lexicVectorLenght,learningRate=0.5) # We indicate the size of the output layer and the input dimensions
    sink = DocumentSink('outputs/LexicSpectra.jsonl', False)

    print("Reading vectors - lexical")
    data = vr.readFeatureVectors()

    print("Training SOM")
    proj.fit(data, 1000)

    print("Getting spectra")
    proj.getProjection(data, sink)

    print('Save model')
    proj.saveSomModel('outputs/LexicModel.som')

    del(vr, proj, sink, data, lexicVectorLenght)

    # Obtaining syntactic spectra

    vr = VectorReader('outputs/SyntacticVectors.jsonl')
    proj = Projector(20,300,learningRate=0.5) # We indicate the size of the output layer and the input dimensions
    sink = DocumentSink('outputs/SyntacticSpectra.jsonl', False)

    print("Reading vectors - syntactic")
    data = vr.readFeatureVectors()

    print("Training SOM")
    proj.fit(data, 1000)

    print("Getting spectra")
    proj.getProjection(data, sink)

    print('Save model')
    proj.saveSomModel('outputs/SyntacticModel.som')

    del(vr, proj, sink, data)

    # Obtaining semantic spectra

    vr = VectorReader('outputs/SemanticVectors.jsonl')
    proj = Projector(20,300,learningRate=0.1) # We indicate the size of the output layer and the input dimensions
    sink = DocumentSink('outputs/SemanticSpectra.jsonl', False)

    print("Reading vectors - semantic")
    data = vr.readFeatureVectors()

    print("Training SOM")
    proj.fit(data, 1000)

    print("Getting spectra")
    proj.getProjection(data, sink)

    print('Save model')
    proj.saveSomModel('outputs/SemanticModel.som')

    del(vr, proj, sink, data)

def layerConsolidation():
    """LayerConsolidation"""

    print("Layer consolidation stage")

    from spectraltrep.layerConsolidation import SpectraAssembler

    assembler = SpectraAssembler("./outputs/FullSpectra.jsonl")
    path = "outputs/"
    assembler.assemble(path+"LexicSpectra.jsonl", path+"SyntacticSpectra.jsonl", path+"SemanticSpectra.jsonl")

    del(assembler, path)

    print("Ensemble finished")

if __name__ == "__main__":
    init = time.time()

    preProcessStage()
    lexicVectorLenght = featureExtractionStage()
    spaceMappingStage(lexicVectorLenght)
    layerConsolidation()

    print("Proceso de transformacion realizado en %.02f segundos" % (time.time() - init))