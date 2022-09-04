# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was created by Igor Pereira. It is a logistic regression model, created with the default hyperparameters of scikit-learn version 0.23.2

## Intended Use
The model should be used to estimate whether a person has an income which exceeds $50K/year according to set of attributes such as race, education, sex and occupation.
Its users are people who work mainly at agencies such as the US Census Bureau, whose goal is to help in the funds allocation process, enabling more data-driven decisions on where to build/maintain schools, hospitals, among other public resources. 

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).
The full data set has 14 attributes, 48842 rows and 80% was used for training. The data was processed using one hot encoding
for the categorical features and a label binarizer for the label 'salary' ('>50K' and '<=50K').

## Evaluation Data
20% of the full data set was used for evaluation/test. As with the training data, one hot encoding for the categorical
features and a label binarizer were used for data processing.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model was evaluated using precision, recall and F1 score. The value obtained for these metrics were , respectively.

## Ethical Considerations

## Caveats and Recommendations
