import os
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from tokenizer import SMILESTokenizer
from transformer import Transformer

def verify_and_visualize(smiles_string, output_filename="molecule_structure.png"):
    """
    Parses a drug's SMILES string, validates its chemical grammar via RDKit,
    and visualizes the 2D molecular structure for assignment submission.
    """
    print(f"--- Launching Paper Verification Pipeline ---")
    print(f"Target Input Drug SMILES: {smiles_string}\n")
    
    # 1. Chemical Parsing & Validation via RDKit
    print("[Step 1] Parsing molecular structure with RDKit...")
    mol = Chem.MolFromSmiles(smiles_string)
    
    if mol is None:
        print("[-] Error: RDKit could not parse the SMILES string. Invalid chemical syntax.")
        return False
    
    print("[+] Chemical structure successfully parsed and validated.")
    
    # 2. Setup Deliverables Directory
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", output_filename)
    
    # 3. Generate and Save 2D Molecular Visualization
    print(f"[Step 2] Rendering 2D molecular structure diagram...")
    # Generate an image asset with standard dimensions (400x400)
    img = Draw.MolToImage(mol, size=(400, 400))
    img.save(output_path)
    
    print(f"[+] Success! Structural visualization saved to: {output_path}")
    print(f"--------------------------------------------")
    return True

if __name__ == "__main__":
    # Standard Drug Test Case: Aspirin
    # This matches the chemical target processed by our core network training script
    drug_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    
    verify_and_visualize(drug_smiles, "aspirin_verification.png")