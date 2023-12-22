# Toxic-Comments-Classifier-cpp



CodeSpace and Github was not allowing to upload files greater than 25MB, hence uploaded a .7z file less than 25MB and extracted the content via codespace terminal.

`sudo apt-get update` <br/>
`sudo apt-get install p7zip-full`

`7z x train.7z` <br/>
`7z x test.7z`

Once you have the .csv files, we will need to do some data process!<br/>
I have used the `DataPreProcessor.py` file, since the comments have a lot of new line and processing it via C++ was bit overwhelming!

Just run the py code and it will generate the new files that containts only the necessary files


`fitOnTexts` Method -> Builds a vocabulary (word index) from a collection of text samples.
- Assigns a unique integer ID to each word, starting from 1 (index 0 is reserved).
- Retains only the most frequent maxFeatures words.


`textToSequence` Method ->
- Converts a single text string into a sequence of integer IDs based on the vocabulary.
- Omitted words (not in the vocabulary) are excluded.


`textsToSequences` Method ->
- Tokenizes multiple text samples into sequences of integer IDs.


`padSequence` Method ->
- Ensures all sequences have the same length (maxlen) by:
- Truncating longer sequences.
- Padding shorter sequences with zeros.


g++ -g
valgrind ./main
valgrind --leak-check=full ./main

valgrind --tool=massif ./main
ms_print massif.out.13305
kcachegrind massif.out.13305

valgrind --tool=massif --massif-out-file=massif.out.13305 ./main
