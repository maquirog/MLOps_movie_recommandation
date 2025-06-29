python src/model-selector/initialise_null_model.py  
sleep 5 

python src/models/train.py  
sleep 5 

python src/models/predict.py  
sleep 5 

python src/models/evaluate.py
sleep 5 

python src/model-selector/compare_and_promote.py
