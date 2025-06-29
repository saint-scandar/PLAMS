#!/usr/bin/env amspython
"""Batch‐generate COSMO-RS .coskf files from a SMILES list."""

import argparse
import os
from pathlib import Path

# Force PLAMS RKF settings for AMS2022 compatibility
try:
    from scm.plams.core.config import config
except ModuleNotFoundError:
    from scm.plams import config
config.kftools.intsize = 8
config.kftools.endian = "little"

from scm.plams import AMSJob, Molecule, from_smiles, init, finish

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def estimate_nconfs_from_smiles(
    smiles: str,
    per_rot: int = 20,
    min_conf: int = 10,
    max_conf: int = 100,
) -> int:
    """Return conformer count estimate based on rotatable bonds."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return min_conf
    nrot = rdMolDescriptors.CalcNumRotatableBonds(m)
    return max(min_conf, min(max_conf, nrot * per_rot))


# Default output directories
workdir_root = Path("/Volumes/S18/workdir2022").resolve()
coskf_dir = Path("/Volumes/S18/py_codes2022").resolve()


# ---------------------------------------------------------------------------
# Fallback implementations for AMS2022
# ---------------------------------------------------------------------------
try:
    from scm.plams.recipes.adfcosmorscompound import ADFCOSMORSCompoundJob

    adf_settings = ADFCOSMORSCompoundJob.adf_settings
    convert_to_coskf = ADFCOSMORSCompoundJob.convert_to_coskf
    get_compound_properties = ADFCOSMORSCompoundJob.get_compound_properties
except Exception:
    from scm.plams.core.settings import Settings
    from scm.plams.tools.kftools import KFFile

    def solvation_settings(elements=None, atomic_ion=False):
        s = Settings()
        s.input.adf.solvation.surf = "Delley"
        s.input.adf.solvation.solv = "name=CRS cav0=0.0 cav1=0.0"
        s.input.adf.solvation.charged = (
            "method=atom corr" if atomic_ion else "method=Conj corr"
        )
        s.input.adf.solvation["c-mat"] = "Exact"
        s.input.adf.solvation.scf = "Var All"
        s.input.adf.solvation.radii = {}
        return s

    def adf_settings(solvation, settings=None, elements=None, atomic_ion=False):
        s = settings.copy() if settings else Settings()
        s.input.adf.Basis.Type = "TZP"
        s.input.adf.Basis.Core = "Small"
        s.input.adf.XC.GGA = "BP86"
        s.input.adf.Symmetry = "NOSYM"
        s.input.adf.BeckeGrid.Quality = "Good"
        s.input.adf.SCF.Var = "All"
        s.input.adf.SCF.LinearScaling = "Off"
        s.input.adf.SCF.ScalableSCF = "Off"
        # Ensure AMS2022 writes history but PLAMS does not attempt to read it
        s.input.adf.Restart.WriteHistory = True
        s.input.adf.Restart.ReadHistory = False
        if solvation:
            s += solvation_settings(elements=elements, atomic_ion=atomic_ion)
        return s

    def get_compound_properties(mol, mol_info=None):
        info = (mol_info or {}).copy()
        info["Formula"] = mol.get_formula()
        return info, None

    def convert_to_coskf(
        rkf_path, coskf_name, plams_dir, coskf_dir=None, mol_info=None, densf_path=None
    ):
        dst = os.path.join(plams_dir, coskf_name)
        from shutil import copy2

        copy2(rkf_path, dst)
        kf = KFFile(dst)
        for k, v in (mol_info or {}).items():
            kf.write("Compound Data", k, float(v) if hasattr(v, "__float__") else v)
        if coskf_dir:
            os.makedirs(coskf_dir, exist_ok=True)
            final = os.path.join(coskf_dir, coskf_name)
            copy2(dst, final)
            return final
        return dst


# ---------------------------------------------------------------------------
# Workflow helpers
# ---------------------------------------------------------------------------

def generate_lowest_conformer(
    smiles: str, nconfs: int | None = None, prune_rms_thresh: float = 0.1
) -> Molecule:
    if nconfs is None:
        nconfs = estimate_nconfs_from_smiles(smiles)
    print(f"⚙️  Generating up to {nconfs} conformers for {smiles!r}")
    try:
        mols = from_smiles(
            smiles,
            nconfs=nconfs,
            forcefield="mmff",
            prune_rms_thresh=prune_rms_thresh,
            embed_params={"useExpTorsionAnglePrefs": True, "useBasicKnowledge": True},
        )
    except TypeError:
        mols = from_smiles(smiles, nconfs=nconfs, forcefield="mmff")

    if isinstance(mols, Molecule):
        print("⚙️  1 conformer generated")
        return mols

    count = len(mols)
    print(f"⚙️  {count} conformers generated (requested {nconfs})")
    if count == 0:
        raise RuntimeError(f"No conformers for {smiles!r}")
    mols.sort(key=lambda m: m.properties.get("energy", 0.0))
    return mols[0]


def run_adf_optimization(mol: Molecule, name: str) -> AMSJob:
    s = adf_settings(solvation=False)
    s.input.ams.Task = "GeometryOptimization"
    job = AMSJob(settings=s, molecule=mol, name=f"{name}_opt")
    try:
        job.run()
    except Exception as e:
        raise RuntimeError(f"Geom opt for {name} crashed inside ADF: {e}")
    if job.status != "successful":
        raise RuntimeError(f"Geom opt for {name} failed (status={job.status})")
    return job


def run_adf_cosmo(gas_job: AMSJob, name: str) -> AMSJob:
    mol = gas_job.results.get_main_molecule()
    elems = sorted({a.symbol for a in mol.atoms})
    s = adf_settings(solvation=True, elements=elems, atomic_ion=(len(mol.atoms) == 1))
    s.input.ams.Task = "SinglePoint"
    job = AMSJob(settings=s, molecule=mol, name=f"{name}_cosmo")
    try:
        job.run()
    except Exception as e:
        raise RuntimeError(f"COSMO SP failed for {name}: {e}")
    if job.status != "successful":
        raise RuntimeError(f"COSMO SP failed for {name}, status={job.status}")
    return job


def process_molecule(smiles: str, mol_name: str, override_n: int) -> None:
    auto_n = override_n if override_n > 0 else None
    mol0 = generate_lowest_conformer(smiles, nconfs=auto_n)

    wd = workdir_root / mol_name
    wd.mkdir(parents=True, exist_ok=True)
    mol0.write(wd / f"{mol_name}_lowest.xyz", outputformat="xyz")

    try:
        gas = run_adf_optimization(mol0, mol_name)
    except Exception as e:
        print(f"⚠️  {mol_name} geom-opt failed → {e}")
        return

    try:
        cosmo = run_adf_cosmo(gas, mol_name)
    except Exception as e:
        print(f"⚠️  {mol_name} COSMO SP failed → {e}")
        return

    props, _ = get_compound_properties(mol0)
    try:
        out = convert_to_coskf(
            cosmo.results.rkfpath(file="adf"),
            coskf_name=f"{mol_name}.coskf",
            plams_dir=cosmo.path,
            coskf_dir=str(coskf_dir),
            mol_info=props,
        )
        print(f"✅  Wrote: {out}")
    except Exception as e:
        print(f"⚠️  {mol_name} write-coskf failed → {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch COSKF from SMILES/name file")
    p.add_argument(
        "--smiles-file",
        default="smilesOG2.txt",
        help="SMILES<whitespace>Name per line",
    )
    p.add_argument(
        "-n",
        "--nconfs",
        type=int,
        default=0,
        help=">0: force that many conformers; 0=auto",
    )
    args = p.parse_args()

    workdir_root.mkdir(parents=True, exist_ok=True)
    coskf_dir.mkdir(parents=True, exist_ok=True)

    with open(args.smiles_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smiles = parts[0]
            name = (parts[1] if len(parts) > 1 else smiles).replace(" ", "_")
            init(path=str(workdir_root), folder=name)
            try:
                process_molecule(smiles, name, args.nconfs)
            finally:
                finish()
