# Spectral Text Representation Documentation

<a id="index"></a>
## Index:
1. [**Documentation**](#documentation)

2. [**Text Pre-Processing Stage**](#preprocessing)
   
3. [**Feature Extraction Stage**](#extraction)

4. [**Unified Space Mapping Stage**](#unifiedSpace)

5. [**Layer Consolidation**](#consolidation)

<a id="documentation"></a>
## Documentation <small>[[Top](#index)]</small>
### Create documentation
```
cd documentation
sphinx-apidoc -o ./_modules ../spectraltrep/
make html
```

### See documentation
```
cd documentation
firefox _build/html/index.html
```

<a id="preprocessing"></a>
## Text Pre-Processing Stage <small>[[Top](#index)]</small>

Different pre-processing techniques are applied to the input corpus depending on the layer we work on (lexical, syntactic, or semantic layer). The only common pre-processing task for all layers are converting all text to lowercase and tokenizing it for all layers.

We do not remove stop words or punctuation symbols to preserve as much information as possible. We also preserve labels with named entities in the form of ```<entity>``` (for example, streets, people, organizations, etc.) in the tokenization process.

The syntactic layer applies the part-of-speech tagging (POS tagging) process. From this stage, we obtained three new versions of the corpus, one for each text component.

To carry out this process, it is possible to use threads, since each text within the corpus can be processed independently of other texts within the corpus, even between feature layers, since it is possible to use threads, locks are also used to prevent something wrong with them.

<p align="center">
  <img src="figs/preprocessing.png" alt="preprocessing"/>
</p>

For this stage, the user only interacts with the PreProcessingFacade class, this class is responsible for reading the corpus with the help of ```CorpusReader()``` found in the ```utils.py``` file, later according to which layers we want (lex, syn, sem) the corresponding preprocessors are created with the help of the ```PreprosessorFactory``` class, where each one makes use of its respective ```LexicPreprocessor```, ```SyntacticPreprocessor``` or ```SemanticPreprocessor```, finally each corpus that results from passing through a Preprocessor is saved using ```DocumentSink()```, which is also found in ```utils.py``` file. 

<p align="center">
  <img src="figs/preprocessingClass.png" alt="preprocessingClass"/>
</p>

The utils file only contains two classes ```CorpusReader``` and ```DocumentSink``` with their respective interfaces. The first one is in charge of reading the corpus but by blocks, since the corpus could be huge and the memory of the computer could not be enough. The second class is in charge of receiving the corpus blocks already pre-processed and here we have two options: saving the blocks as they are received or first receiving all those blocks, ordering them, and finally saving them.

<p align="center">
  <img src="figs/utilsClass.png" alt="utilsClass"/>
</p>

<a id="extraction"></a>
## Feature Extraction Stage <small>[[Top](#index)]</small>
After this stage, we obtain three feature vectors for the same text (lex, syn, sem) corresponding to each text component. For this, the extraction of each of these types of vectors is as follows:

- Lexical layer: Each entry in the vector corresponds to the amount of information (using Shannon's formula) provided by each word of the vocabulary, including punctuation marks and stop words if desired.

- Syntactic layer:  POS tagging process was applied in the pre-processing stage to obtain a POS tag sequence from the original text. In this way, using the ```Doc2Vec``` algorithm is expected to capture syntactic information about the content  to obtain the vector.

- Semantic layer: We want to obtain feature vectors that capture semantic information. Given this, we resort to the ```Doc2Vec``` algorithm once again to extract the corresponding feature vectors.

<p align="center">
  <img src="figs/extraction.png" alt="extraction"/>
</p>

To achieve this, we will once again use the ```CorpusReader``` and ```DocumentSink ``` classes from the ```utils.py``` file to read the text that resulted from the pre-processing stage and save the vectors once this stage is finished.

The class that interacts with the user is the ```VectorizerFactory()```, which is in charge of creating the Vectorizers for each layer and does what corresponds to its layer (mentioned above).

<p align="center">
  <img src="figs/extractionClass.png" alt="extractionClass"/>
</p>

<a id="unifiedSpace"></a>
## Unified Space Mapping Stage <small>[[Top](#index)]</small>
The sets of extracted feature vectors lex,  syn, and sem, can have a different number of dimensions. At this stage, we make use of the ```Self-Organizing Maps (SOM)``` to transfer vectors with a different number of dimensions to a space with the same dimensions where their similarity is preserved.

<p align="center">
  <img src="figs/unifiedSpace.png" alt="unifiedSpace"/>
</p>

The class in charge is called ```Projector``` and this in turn implements the following code [SOM](https://github.com/JustGlowing/minisom).

<p align="center">
  <img src="figs/unifiedSpaceClass.png" alt="unifiedSpaceClass"/>
</p>

<a id="consolidation"></a>
## Layer Consolidation <small>[[Top](#index)]</small>
This final stage of the text transformation consists only of taking the three spectra of each text to consolidate them into a single three-layer text representation containing lexical, syntactical, and semantic features about the content.

<p align="center">
  <img src="figs/consolidation.png" alt="consolidation"/>
</p>

This stage consists of only two classes, the first ```SpectraAssembler```, does the work mentioned in the previous paragraph, which needs the route of the 3 vectors as well as the final route.
The second is ```SpectraReader``` class, as its name indicates, returns the full spectrum as ```JSON```.

<p align="center">
  <img src="figs/consolidationClass.png" alt="consolidationClass"/>
</p>