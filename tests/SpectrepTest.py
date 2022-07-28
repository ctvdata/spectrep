import time
import sys
sys.path.append('..')
from spectraltrep.preProcessing import PreProcessingFacade
from spectraltrep.featureExtraction import VectorizerFactory, Doc2VecCorpusReader
from spectraltrep.utils import CorpusReader, DocumentSink
from spectraltrep.spaceUnification import VectorReader, Projector
from spectraltrep.utils import DocumentSink

def preProcessStage():
    """Etapa de pre-procesamiento"""

    print("Etapa de pre-procesmiento")

    inputPath = '../data/data_sample.jsonl' # Archivo de entrada
    outputPath = 'outputs/Preprocessed.jsonl' # Archivo de salida
    preProcessingType = ['lex','syn','sem'] # Tipo de preprocesamiento aplicable ['lex', 'syn', 'sem']
    numThreads = 1 # Numero de hilos de preprocesamiento
    batchSize = 500 # Numero de documentos que entrega CorpusReader por batch
    sortedOutput = True

    ppf = PreProcessingFacade()
    ppf.preProcess(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput)

    del(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput, ppf)

def featureExtractionStage() -> int:
    """Etapa de extraccion de caracteristicas"""

    print("Etapa de extraccion de caracteristicas")

    # Extraccion de vectores léxicos

    print('Extraccion de vectores lexicos')
    cr = CorpusReader('outputs/Preprocessed_lex.jsonl', 500)
    vw = DocumentSink('outputs/LexicVectors.jsonl', False)
    vf = VectorizerFactory()
    lv = vf.createLexicVectorizer(vw, cr)

    print('Entrenando modelo lexico')
    lv.fit()

    print('Realizando vectorizacion')
    lv.transform()

    print('Guardando resultados')
    lv.saveModel('outputs/lexicModel.json')
    vw.saveCorpus()

    lexicVectorLenght = lv.model.vocabularyLength

    del(cr, vw, vf, lv)

    # Extraccion de vectores sintácticos

    print('Extraccion de vectores sintacticos')
    cr = Doc2VecCorpusReader('outputs/Preprocessed_syn.jsonl')
    vw = DocumentSink('outputs/SyntacticVectors.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw, cr, 300)

    print('Entrenando modelo sintactico')
    sv.fit()

    print('Realizando vectorizacion')
    sv.transform()

    print('Guardando resultados')
    sv.saveModel('outputs/dv2SynModel')
    vw.saveCorpus()

    del(cr, vw, vf, sv)

    # Extraccion de vectores semanticos

    print('Extraccion de vectores semanticos')
    cr = Doc2VecCorpusReader('outputs/Preprocessed_sem.jsonl')
    vw = DocumentSink('outputs/SemanticVectors.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw, cr, 300)

    print('Entrenando modelo semantico')
    sv.fit()

    print('Realizando vectorizacion')
    sv.transform()

    print('Guardando resultados')
    sv.saveModel('outputs/dv2SemModel')
    vw.saveCorpus()

    del(cr, vw, vf, sv)

    return lexicVectorLenght

def spaceMappingStage(lexicVectorLenght: int):
    """Unificacion del espacio"""

    print("Etapa de unificacion del espacio")

    # Obtencion de espectros lexicos

    vr = VectorReader('outputs/LexicVectors.jsonl')
    proj = Projector(20,lexicVectorLenght,learningRate=0.5) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
    sink = DocumentSink('outputs/LexicSpectra.jsonl', False)

    print("Leyendo vectores lexicos")
    data = vr.readFeatureVectors()

    print("Entrenando som")
    proj.fit(data, 1000)

    print("Obteninendo espectros")
    proj.getProjection(data, sink)

    print('Guardando modelo')
    proj.saveSomModel('outputs/LexicModel.som')

    del(vr, proj, sink, data, lexicVectorLenght)

    # Obtencion de espectros sintacticos

    vr = VectorReader('outputs/SyntacticVectors.jsonl')
    proj = Projector(20,300,learningRate=0.5) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
    sink = DocumentSink('outputs/SyntacticSpectra.jsonl', False)

    print("Leyendo vectores sintacticos")
    data = vr.readFeatureVectors()

    print("Entrenando som")
    proj.fit(data, 1000)

    print("Obteninendo espectros")
    proj.getProjection(data, sink)

    print('Guardando modelo')
    proj.saveSomModel('outputs/SyntacticModel.som')

    del(vr, proj, sink, data)

    # Obtencion de espectros semanticos

    vr = VectorReader('outputs/SemanticVectors.jsonl')
    proj = Projector(20,300,learningRate=0.1) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
    sink = DocumentSink('outputs/SemanticSpectra.jsonl', False)

    print("Leyendo vectores semanticos")
    data = vr.readFeatureVectors()

    print("Entrenando som")
    proj.fit(data, 1000)

    print("Obteninendo espectros")
    proj.getProjection(data, sink)

    print('Guardando modelo')
    proj.saveSomModel('outputs/SemanticModel.som')

    del(vr, proj, sink, data)

def layerConsolidation():
    """Consolidacion de capas"""

    print("Etapa de consolidacion de capas")

    from spectraltrep.layerConsolidation import SpectraAssembler

    assembler = SpectraAssembler("./outputs/FullSpectra.jsonl")
    path = "outputs/"
    assembler.assemble(path+"LexicSpectra.jsonl", path+"SyntacticSpectra.jsonl", path+"SemanticSpectra.jsonl")

    del(assembler, path)

    print("Ensamble terminado")

if __name__ == "__main__":
    init = time.time()

    preProcessStage()
    lexicVectorLenght = featureExtractionStage()
    spaceMappingStage(lexicVectorLenght)
    layerConsolidation()

    print("Proceso de transformacion realizado en %.02f segundos" % (time.time() - init))