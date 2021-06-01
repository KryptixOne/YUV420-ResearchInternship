# YUV420-ResearchInternship
YUV420 Compatibility with CompressAI to the Hyperprior Model

To use this:

1) install the CompressAI library as instructed:
https://github.com/InterDigitalInc/CompressAI

2) Within CompressAI library location. Replace the "Utils.py" and "__init__.py" Files in the ./compressai/datasets directory with appropriate files in Git changes folder
3) Implement similar change in the CompressAI library location: ./compressai/models Directory for the files "__init__.py" and "priors.py" with appropriate files in Git changes folder

4) Use Python file QualityTest.py to implement models for single image compression -> reconstruction

OR

5) Use Python file TestWithOwnNetwork.py to train own model based on YUV Images.


********************************
How to run QualityTest.py
********************************

Run quality test with the following input parameters:

-d C:\Users\path\to\Compressable_Image_folder       #dataset directory. Be sure to create a folder with the name "test". In this folder put the SINGLE image you wish to compress
--cuda        #type of Device to be used. cpu or cuda
--training_type 1   # Training type/ Compression type. 1 = Separate Paths, 2 = PixelShuffle
--lambda 0.1 #Lambda that your checkpoint/model is associated with
--FinalFileName C:\Users\Path\to\Directory\ReconstructedImageName.yuv  #Full path and file name of the reconstructed Image
--checkpoint C:\Users\Path\To\Desired\Checkpoint\Directory\checkpoint_best_loss.pth.tar #Location of the checkpoint you wish to use.

********************************
How to run TestWithOwnNetwork.py
********************************

Same procedure as with the original compressAI documentation except for the following inclusion:
--training_type 1   # Training type/ Compression type. 1 = Separate Paths, 2 = PixelShuffle


