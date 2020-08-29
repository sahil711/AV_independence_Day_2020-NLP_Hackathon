Link to the Competition: https://datahack.analyticsvidhya.com/contest/janatahack-independence-day-2020-ml-hackathon/#LeaderBoard

Model Details :

* Architecture : Bert
* Model Name: scibert_scivocab_uncased, bert base pre-trained on research papers from Semantic Scholar with a custom vocabulary (huggingface link: https://bit.ly/34syFQ1)
* Final solution an ensemble of three models:
    * Model 1: 
        Input: Title and abstract separated by [SEP] 
        Mean pooling of the final hidden states of all the tokens
        CV Score: 85.04 Pub LB: 85.57 Pvt LB: 86.07

    * Model 2:
        Input: Title and abstract simple concatenation (i.e. w/o the [SEP] token)
        Mean pooling of the final hidden states of all the tokens
        CV Score: 84.98 Pub LB: 85.44 Pvt LB: 86.13

    * Model 3:
        Input: Title and abstract simple concatenation (i.e. w/o the [SEP] token)
        Mean pooling of the last three layer’s hidden states of all the tokens
        CV Score: 84.99 Pub LB: 85.53 Pvt LB: 86.07
    
**All three models would score in the Top3 individually**

**Tip:
The  most important takeaway was to use a pre-trained model which was more suited to the task (in this case scientific papers, so SciBert). Using Scibert over Bert immediately gave a boost of more than 1 %points micro F1 score on the CV. **


Steps to replicate the solution:

1) Install all the dependencies in requirements.txt (replica of kaggle's environment with transformers upgraded to 3.0.2 and iterative-stratification==0.1.6)
2) Solution directory should contain the train and test csv for it to run successfully (incase it is not, please change the DATA_DIR variable for each notebook accordingly)
3) Final solution is an ensemble of three models, sequence of notebooks to generate the final submission is as follows:
    - run model1/scibertv1_uncased.ipynb
    - run model2/scibertv2_uncased.ipynb
    - run model3/scibertv2_uncased_mutlilayer.ipynb
    - run ensemble.ipynb
    - Running this sequence will generate final_submission.csv
    
4) Please note that the scripts were run on kaggle environment (with some libraries upgraded to the latest version)