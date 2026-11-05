'''
This script retrieves the Hansen solubility parameters of the solvents 
by their SMILES and saves them to a CSV file.
'''


from rdkit import Chem
import pandas as pd

from utils import DATA_DIR

def get_hansen_parameters_from_smiles(df, smiles):
    """
    Retrieve Hansen Solubility Parameters from a DataFrame using a SMILES string.

    Parameters:
    - df (pd.DataFrame): DataFrame with solvent data including InChIKey and Hansen parameters.
    - smiles (str): SMILES string of the solvent molecule.

    Returns:
    - dict: Hansen solubility parameters (δd, δp, δh, δt) as floats.

    Raises:
    - ValueError: If the SMILES string is invalid or not found in the DataFrame.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}'")

    inchikey = Chem.MolToInchiKey(mol)

    if inchikey not in df['InChIKey'].values:
        raise ValueError(f"InChIKey '{inchikey}' derived from SMILES not found in dataset.")

    row = df[df['InChIKey'] == inchikey].iloc[0]

    def extract_quantity(value):
        try:
            # Assumes format: Quantity[<float>, ...]
            return float(str(value).split("[")[1].split(",")[0])
        except:
            raise ValueError(f"Unexpected quantity format: {value}")

    return {
        'SMILES': smiles,
        'δd': extract_quantity(row["δd"]),
        'δp': extract_quantity(row["δp"]),
        'δh': extract_quantity(row["δh"]),
        'δt': extract_quantity(row["δt"]),
    }

solvents = {
    'DMSO': "CS(C)=O", # Dimethyl sulfoxide (DMSO)
    'DMA': "CC(=O)N(C)C", # Dimethylacetamide (DMA)
    "DMF": "CN(C)C=O", # N,N-Dimethylformamide (DMF)
    "Sulfolane": "C1CCS(=O)(=O)C1", # Sulfolane
    "Toluene": "CC1=CC=CC=C1", # Toluene
    "1,4-Dioxane": "O1CCOCC1", # 1,4-Dioxane
    "Diglyme" : "COCCOCCOC", # Diglyme
    "Triglyme": "COCCOCCOCCOC", # Triglyme
    "Isoamyl acetate": "O=C(OCCC(C)C)C", # Isoamyl acetate
    "CPME": "COC1CCCC1", # CPME
}

hansen_df = pd.read_csv(DATA_DIR / "JoshuaSchrier_Hansen-Solubility-Parameters.csv")
# Source of the dataset: https://datarepository.wolframcloud.com/resources/JoshuaSchrier_Hansen-Solubility-Parameters/
# Source Triglyme: https://www.hansen-solubility.com/HSP-examples/cleaning.php
# Source CPME: https://www.stevenabbott.co.uk/practical-solubility/hsp-basics.php
features = {}
for solvent, smiles in solvents.items():
    try:
        params = get_hansen_parameters_from_smiles(hansen_df, smiles)
        features[solvent] = params
        print(f"SMILES: {smiles} -> Hansen Parameters: {params}")
    except ValueError as e:
        print(f"Error for SMILES '{smiles}': {e}")
        continue

params_df = pd.DataFrame.from_dict(features, orient='index')
params_df.to_csv(DATA_DIR / "hansen_solubility_parameters.csv", index_label='Solvent')