# CUB dataset

kaggle datasets download -d tarunkr/caltech-birds-2011-dataset
unzip -qq caltech-birds-2011-dataset.zip "CUB_200_2011/images/*" -d "./filelists/CUB/"
mkdir ./filelists/CUB/images
mv ./filelists/CUB/CUB_200_2011/images/* ./filelists/CUB/images/
rm -rf ./filelists/CUB/CUB_200_2011/
rm *.zip

# Aircraft 

kaggle datasets download -d seryouxblaster764/fgvc-aircraft
unzip -qq fgvc-aircraft.zip "fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/*" -d "./filelists/aircrafts/"
mkdir ./filelists/aircrafts/images
mv ./filelists/aircrafts/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/* ./filelists/aircrafts/images/
rm -rf ./filelists/aircrafts/fgvc-aircraft-2013b
rm *.zip

# Flowers

kaggle datasets download -d arjun2000ashok/vggflowers
unzip -qq vggflowers.zip "images/*" -d "./filelists/flowers/"
rm *.zip

# Dogs

kaggle datasets download -d jessicali9530/stanford-dogs-dataset
unzip -qq stanford-dogs-dataset.zip "images/Images/*" -d "./filelists/dogs/"
mkdir ./filelists/dogs/Images/
mv ./filelists/dogs/images/Images/* ./filelists/dogs/Images/
rm -rf ./filelists/dogs/images
rm *.zip

# Cars

kaggle datasets download -d hassiahk/stanford-cars-dataset-full
unzip -qq stanford-cars-dataset-full.zip "images/*" -d "./filelists/cars/"
rm *.zip