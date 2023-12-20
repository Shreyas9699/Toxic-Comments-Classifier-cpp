# Toxic-Comments-Classifier-cpp



CodeSpace and Github was not allowing to upload files greater than 25MB, hence uploaded a .7z file less than 25MB and extracted the content via codespace terminal.

`sudo apt-get update` <br/>
`sudo apt-get install p7zip-full`

`7z x train.7z` <br/>
`7z x test.7z`

Once you have the .csv files, we will need to do some data process!<br/>
I have used the `DataPreProcessor.py` file, since the comments have a lot of new line and processing it via C++ was bit overwhelming!

Just run the py code and it will generate the new files that containts only the necessary files
