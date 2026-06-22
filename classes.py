"""
© 2026 Nicholas J. Calabro. All rights reserved.

This file is intended to be a top level dependency file.
Ideally no imports.

Concrete data about the problem I'm solving goes here.

Dataset, architecture, and training independent.

At some point I may opt to combine or drop one or more class(es).

"""


ALL_CLASSES = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Effusion","Emphysema","Fibrosis","Hernia",
    "Infiltration","Mass","No Finding","Nodule",
    "Pleural_Thickening","Pneumonia","Pneumothorax",
]

NO_FINDING_COL = ALL_CLASSES.index("No Finding")

NUM_CLASSES = len(ALL_CLASSES)

THRESHOLD_COUNT        = 32

SENS_THRESHOLD_ID = 0
SPEC_THRESHOLD_ID = THRESHOLD_COUNT - 1

# Minimum number of value needed to evaluate a catagory's AUC
# to avoid unreliable estimates and early stopping
MIN_VAL_POSITIVES = 50


def print_classes_parameters():
    print("Class Parameters")
    print("  ALL_CLASSES", ALL_CLASSES)
    print("  NUM_CLASSES", NUM_CLASSES)
    print("  MIN_VAL_POSITIVES", MIN_VAL_POSITIVES)

