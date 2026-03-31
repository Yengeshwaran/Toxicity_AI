import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem


def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    features = {}

    # Basic molecular descriptors
    features["MolWt"] = Descriptors.MolWt(mol)
    features["LogP"] = Descriptors.MolLogP(mol)
    features["NumHDonors"] = Descriptors.NumHDonors(mol)
    features["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)

    # Morgan Fingerprints (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    for i, bit in enumerate(fp):
        features[f"FP_{i}"] = int(bit)

    return features


def build_dataset(df):
    data = []

    for _, row in df.iterrows():
        smiles = row["smiles"]
        label = row["toxicity"]

        feats = smiles_to_features(smiles)

        if feats is not None:
            feats["label"] = label
            data.append(feats)

    return pd.DataFrame(data)