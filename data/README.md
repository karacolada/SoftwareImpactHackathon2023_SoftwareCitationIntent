# Datasets

For training, use columns `text` and `label`. `label` is a numerical label with the following mapping:

```
num2label = {0: "created", 1: "used", 2: "mention", 3: "none"}
label2num = {"created": 0, "used": 1, "mention": 2, "none": 3}
```

To regenerate the CSV files from the Excel sheets, run the notebook `load_data.ipynb`. 

- `software_citation_intent_merged.csv`: dataset build by Josh, Martin and Kai from Softcite, SoMeSci and unlabelled examples. Also contains a column `context` which has the surrounding sentences. In the case of the unlabelled examples, this is currently equivalent to the `sentences` column but that will be updated soon.
- `software_citation_intent_czi.csv`: dataset curated by CZI, provided by Ana and mapped to our intent labels by Donghui. Does not contain `context` column. Removed rows with unclear/NaN citation intent.
