1) Install all the dependencies in requirements.txt (replica of kaggle's environment with transformers upgraded to 3.0.2 and iterative-stratification==0.1.6)
2) Solution directory should contain the train and test csv for it to run successfully (incase it is not, please change the DATA_DIR variable for each notebook accordingly)
3) Final solution is an ensemble of three models, sequence of notebooks to generate the final submission is as follows:
    - run model1/scibertv1_uncased.ipynb
    - run model2/scibertv2_uncased.ipynb
    - run model3/scibertv2_uncased_mutlilayer.ipynb
    - run ensemble.ipynb
    - Running this sequence will generate final_submission.csv
    
4) Please note that the scripts were run on kaggle environment (with some libraries upgraded to the latest version)