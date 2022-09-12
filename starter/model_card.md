# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was created by Igor Pereira. It is a logistic regression model, created with the default hyperparameters of scikit-learn version 0.23.2.

## Intended Use
The model should be used to estimate whether a person has an income which exceeds $50K/year according to set of attributes 
such as race, education, sex and occupation. Its users could be people who work mainly at agencies such as the US Census Bureau, 
whose goal is to help in the funds allocation process, enabling more data-driven decisions on where to build/maintain schools, 
hospitals, among other public resources. Although a possible intended use is described here, it is important to note that 
this model was built in the context of a Machine Learning course project with the purpose of demonstrating model serving and deployment.
Hence, it should not be used directly in a real production scenario (for more information, see the sections "Ethical Considerations" 
and "Caveats and Recommendations" below).

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).
The full data set has 14 attributes, 48842 rows and 80% was used for training. The data was processed using one hot encoding
for the categorical features and a label binarizer for the label 'salary' ('>50K' and '<=50K').

## Evaluation Data
20% of the full data set was used for evaluation/test. As with the training data, one hot encoding for the categorical
features and a label binarizer were used for data processing.

## Metrics
The model was evaluated using precision, recall and F1 score. The value obtained for these metrics on the test set were 
0.572234273318872, 0.8395926161680458 and 0.6805985552115582, respectively.

## Ethical Considerations
Although a deeper analysis is necessary, a simple EDA on the data set reveals some information which can lead to bias/unfairness on the model. 
For instance, regarding the target variable 'salary' we have ~75% of the people on the data set earning $50K/year or less. 
Besides, 85% are white and 67% are male. The imbalance present in these attributes is very likely to influence on what the model learns and, therefore,
lead to biased predictions.

## Caveats and Recommendations
One should mainly consider that the used data set was built on the 1994 Census database, so it is definitely outdated and doesn't reflect the current 
picture of population income in the US. In terms of model building, using more recent data, testing more complex models and applying techniques 
such as cross-validation and hyperparameter tuning are some recommendations to further improve the model results.