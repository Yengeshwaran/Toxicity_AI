from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

def extract_features(smiles):
    """
    Dummy feature extractor that takes a SMILES string and returns simple RDKit features
    expected by our placeholder model.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")

    # Example placeholder features: Molecular Weight, LogP, TPSA, Number of Rotatable Bonds
    features = {
        "MolWt": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

    # Return as list of values (or format expected by model)
    return [
        features["MolWt"],
        features["LogP"],
        features["TPSA"],
        features["NumRotatableBonds"]
    ], list(features.keys())
