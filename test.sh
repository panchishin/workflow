echo ========================
echo python autoencode_spec.py
python autoencode_spec.py

echo ========================
echo git checkout meta-data
git checkout meta-data

echo ========================
echo python data_source.py
python data_source.py

echo ========================
echo python label_spec.py 
python label_spec.py 

echo ========================
echo Done