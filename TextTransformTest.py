# import time
from spectraltrep.preProcessing import PreProcessingFacade
from spectraltrep.featureExtraction import VectorizerFactory, Doc2VecCorpusReader
from spectraltrep.utils import CorpusReader, DocumentSink
from spectraltrep.spaceUnification import VectorReader, Projector
from spectraltrep.utils import DocumentSink

def preProcessStage(inputPath, outputPath):
    """Etapa de pre-procesamiento"""

    print("Etapa de pre-procesmiento")

    # inputPath = 'pan22-authorship-verification-training-dataset/particionesXid/PanUniqueDocs.jsonl' # Archivo de entrada
    # outputPath = 'outputs/panPreprocessed.jsonl' # Archivo de salida
    preProcessingType = ['lex','syn','sem'] # Tipo de preprocesamiento aplicable ['lex', 'syn', 'sem']
    numThreads = 1 # Numero de hilos de preprocesamiento
    batchSize = 500 # Numero de documentos que entrega CorpusReader por batch
    sortedOutput = True

    ppf = PreProcessingFacade()
    ppf.preProcess(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput)

    del(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput, ppf)

def featureExtractionStage(lexInput, lexOutput, lexModel, synInput, synOutput, synModel, semInput, semOutput, semModel):
    """Etapa de extraccion de caracteristicas"""

    print("Etapa de extraccion de caracteristicas")

    # Extraccion de vectores léxicos

    print('Extraccion de vectores lexicos')
    cr = CorpusReader(lexInput, 500)
    vw = DocumentSink(lexOutput, False)
    vf = VectorizerFactory()
    lv = vf.createLexicVectorizer(vw)
    lv.model = lexModel

    print('Realizando vectorizacion')
    lv.transform(cr)

    print('Guardando resultados')
    vw.saveCorpus()

    del(cr, vw, vf, lv)

    # Extraccion de vectores sintácticos

    print('Extraccion de vectores sintacticos')
    cr = Doc2VecCorpusReader(synInput)
    vw = DocumentSink(synOutput, False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw)
    sv.model = synModel
    print('Realizando vectorizacion')
    sv.transform(cr)

    print('Guardando resultados')
    vw.saveCorpus()

    del(cr, vw, vf, sv)

    # Extraccion de vectores semanticos

    print('Extraccion de vectores semanticos')
    cr = Doc2VecCorpusReader(semInput)
    vw = DocumentSink(semOutput, False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw)
    sv.model = semModel

    print('Realizando vectorizacion')
    sv.transform(cr)

    print('Guardando resultados')
    vw.saveCorpus()

    del(cr, vw, vf, sv)

def spaceMappingStage(lexInput, lexOutput, lexModel, synInput, synOutput, synModel, semInput, semOutput, semModel):
    """Unificacion del espacio"""

    print("Etapa de unificacion del espacio")

    # Obtencion de espectros lexicos

    vr = VectorReader(lexInput)
    proj = Projector(0,0) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
    proj.loadSomModel(lexModel)
    sink = DocumentSink(lexOutput, False)

    print("Leyendo vectores lexicos")
    data = vr.readFeatureVectors()

    print("Obteninendo espectros")
    proj.getProjection(data, sink)

    del(vr, proj, sink, data)

    # Obtencion de espectros sintacticos

    vr = VectorReader(synInput)
    proj = Projector(0,0) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
    proj.loadSomModel(synModel)
    sink = DocumentSink(synOutput, False)

    print("Leyendo vectores sintacticos")
    data = vr.readFeatureVectors()

    print("Obteninendo espectros")
    proj.getProjection(data, sink)

    del(vr, proj, sink, data)

    # Obtencion de espectros semanticos

    vr = VectorReader(semInput)
    proj = Projector(0,0) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
    proj.loadSomModel(semModel)
    sink = DocumentSink(semOutput, False)

    print("Leyendo vectores semanticos")
    data = vr.readFeatureVectors()

    print("Obteninendo espectros")
    proj.getProjection(data, sink)

    del(vr, proj, sink, data)

def layerConsolidation(lexInput, synInput, semInput, output):
    """Consolidacion de capas"""

    print("Etapa de consolidacion de capas")

    from spectraltrep.layerConsolidation import SpectraAssembler

    assembler = SpectraAssembler(output)
    assembler.assemble(lexInput, synInput, semInput)

    del(assembler)

    print("Ensamble terminado")

# if __name__ == "__main__":
#     init = time.time()

#     preProcessStage()
#     lexicVectorLenght = featureExtractionStage()
#     spaceMappingStage(lexicVectorLenght)
#     layerConsolidation()

#     print("Proceso de transformacion realizado en %.02f segundos" % (time.time() - init))