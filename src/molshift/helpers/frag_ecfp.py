"""
frag_ecfp.py -- ECFP4 fingerprints for the fragment library, stored as a separate parquet keyed by frag_id.

    build_ecfp4_cache("fragment_features_organic_noalerts.parquet", "ecfp4_cache.parquet")
    lib = load_ecfp4("ecfp4_cache.parquet")
    sims = lib.tanimoto_to_smiles("*c1ccc(O)cc1")                 # whole library, vectorised
    sims = lib.tanimoto_to_smiles("*c1ccc(O)cc1", frag_ids=shortlist["frag_id"])   # only a shortlist

Design choices
  * ECFP4 = Morgan radius 2, folded to 2048 bits, stored as the list of ON bits (small: fragments set few bits).
  * By default the attachment dummy is KEPT as an atom (keep_dummy=True). With H-capping, isopropyl and
    n-propyl become the same molecule (propane) and get identical fingerprints; with the dummy kept, the
    environments next to the attachment point are part of the fingerprint, so they differ. BRICS labels
    are ignored (isotopes cleared), as everywhere else in the pipeline.
  * The settings are written into the parquet metadata, and `lib.bits(smi)` uses the same settings,
    so a reference fingerprint can never be computed differently from the library.
"""
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool
from scipy import sparse
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

RADIUS, N_BITS = 2, 2048
_GENS = {}


def _gen(radius, n_bits):
    if (radius, n_bits) not in _GENS:
        _GENS[(radius, n_bits)] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _GENS[(radius, n_bits)]


def _prep(mol, keep_dummy):
    """Keep the dummy as an atom (labels cleared), or cap it with H like the descriptors do."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetIsotope(0)
            if not keep_dummy:
                a.SetAtomicNum(1)
                a.SetNoImplicit(True)
    m = rw.GetMol()
    if keep_dummy:
        return m
    Chem.SanitizeMol(m)
    return Chem.RemoveHs(m)


def fingerprint_bits(smi, keep_dummy=True, radius=RADIUS, n_bits=N_BITS):
    """Sorted array of ON-bit indices (uint16), or None if the SMILES cannot be processed."""
    try:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            return None
        arr = _gen(radius, n_bits).GetFingerprintAsNumPy(_prep(mol, keep_dummy))
        return np.flatnonzero(arr).astype(np.uint16)
    except Exception:
        return None


def scaffold_to_dummy_smiles(scaff_smi, selected_atts):
    """
    Reference scaffold (attachment atoms are normal atoms) -> SMILES with a '*' on each selected attachment atom,
    i.e. written the same way as the library fragments, so its fingerprint is comparable.
    """
    mol = Chem.MolFromSmiles(scaff_smi)
    if mol is None:
        raise ValueError("Invalid scaffold SMILES")
    rw = Chem.RWMol(mol)
    for a in selected_atts:
        d = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(int(a), d, Chem.BondType.SINGLE)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)


# ───────────────────────── build the cache ─────────────────────────
def _worker(args):
    smi, keep_dummy, radius, n_bits = args
    return fingerprint_bits(smi, keep_dummy, radius, n_bits)


def build_ecfp4_cache(feature_parquet, out_parquet, smiles_col="frag_smiles", id_col="frag_id",
                      keep_dummy=True, radius=RADIUS, n_bits=N_BITS, chunk=200_000, n_jobs=1):
    """Streams the feature table in chunks and writes ecfp4_cache.parquet (frag_id, ecfp4 = list of ON bits)."""
    meta = {b"n_bits": str(n_bits).encode(), b"radius": str(radius).encode(),
            b"keep_dummy": str(int(keep_dummy)).encode()}
    schema = pa.schema([("frag_id", pa.int64()), ("ecfp4", pa.list_(pa.uint16()))]).with_metadata(meta)
    tmp = out_parquet + ".tmp"
    pf = pq.ParquetFile(feature_parquet)
    n_ok = n_bad = 0
    pool = Pool(n_jobs) if n_jobs and n_jobs > 1 else None
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=chunk, columns=[id_col, smiles_col]):
            ids = batch.column(0).to_numpy(zero_copy_only=False)
            smis = batch.column(1).to_pylist()
            jobs = [(s, keep_dummy, radius, n_bits) for s in smis]
            bits = pool.map(_worker, jobs, chunksize=2000) if pool else [_worker(j) for j in jobs]
            keep = [i for i, b in enumerate(bits) if b is not None]
            n_bad += len(bits) - len(keep)
            n_ok += len(keep)
            lengths = np.array([len(bits[i]) for i in keep], dtype=np.int64)
            values = np.concatenate([bits[i] for i in keep]) if keep else np.empty(0, np.uint16)
            offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
            arr = pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), pa.array(values, pa.uint16()))
            writer.write_table(pa.table({"frag_id": pa.array(ids[keep], pa.int64()), "ecfp4": arr}, schema=schema))
    if pool:
        pool.close()
    os.replace(tmp, out_parquet)
    print(f"ECFP4 cache: {n_ok:,} fingerprints written, {n_bad:,} fragments could not be processed -> {out_parquet}")


# ───────────────────────── load + search ─────────────────────────
class ECFPLibrary:
    def __init__(self, ids, X, n_bits, radius, keep_dummy):
        self.ids, self.X = ids, X
        self.n_bits, self.radius, self.keep_dummy = n_bits, radius, keep_dummy
        self.counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        self.index = pd.Index(ids)

    def bits(self, smi):
        """Fingerprint of a reference, computed with the SAME settings as the library."""
        b = fingerprint_bits(smi, self.keep_dummy, self.radius, self.n_bits)
        if b is None:
            raise ValueError(f"Could not fingerprint: {smi}")
        return b

    def tanimoto(self, query_bits, frag_ids=None):
        """Tanimoto of one query (ON-bit indices) against the library, or against the given frag_ids only.
        Fragments with no fingerprint in the cache get NaN."""
        q = np.zeros(self.n_bits, dtype=np.float32)
        q[np.asarray(query_bits, dtype=np.int64)] = 1.0
        if frag_ids is None:
            X, nx, ok = self.X, self.counts, None
        else:
            rows = self.index.get_indexer(np.asarray(frag_ids))
            ok = rows >= 0
            X, nx = self.X[rows[ok]], self.counts[rows[ok]]
        inter = X @ q
        den = nx + q.sum() - inter
        sim = np.divide(inter, den, out=np.zeros_like(inter), where=den > 0)
        if ok is None:
            return sim
        out = np.full(len(ok), np.nan, dtype=np.float32)
        out[ok] = sim
        return out

    def tanimoto_to_smiles(self, smi, frag_ids=None):
        return self.tanimoto(self.bits(smi), frag_ids)


def load_ecfp4(path):
    t = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    n_bits, radius, keep_dummy = int(meta["n_bits"]), int(meta["radius"]), bool(int(meta["keep_dummy"]))
    ids = t["frag_id"].to_numpy()
    arr = t["ecfp4"].combine_chunks()
    offsets = arr.offsets.to_numpy()
    offsets = offsets - offsets[0]
    values = arr.flatten().to_numpy().astype(np.int32)
    X = sparse.csr_matrix((np.ones(len(values), dtype=np.float32), values, offsets), shape=(len(ids), n_bits))
    return ECFPLibrary(ids, X, n_bits, radius, keep_dummy)


# ───────────────────────── option B: ECFP4 as a column of the feature table itself ─────────────────────────
def add_ecfp4_column(feature_parquet, out_parquet, smiles_col="frag_smiles", col="ecfp4", keep_dummy=True,
                     radius=RADIUS, n_bits=N_BITS, chunk=200_000, n_jobs=1):
    """
    Copies feature_parquet to out_parquet with one extra column `ecfp4` (list<uint16> of ON bits), streaming in
    chunks. All existing columns and the schema are untouched. The fingerprint settings are stored in the file's
    metadata. A SMILES that cannot be processed gets an EMPTY list (counted and reported).
    Run it AFTER stage 4: rerunning stage 4 rebuilds the table without this column.
    """
    pf = pq.ParquetFile(feature_parquet)
    if col in pf.schema_arrow.names:
        raise ValueError(f"{feature_parquet} already has a column called {col!r}")
    meta = dict(pf.schema_arrow.metadata or {})
    meta.update({b"ecfp4_n_bits": str(n_bits).encode(), b"ecfp4_radius": str(radius).encode(),
                 b"ecfp4_keep_dummy": str(int(keep_dummy)).encode()})
    schema = pf.schema_arrow.append(pa.field(col, pa.list_(pa.uint16()))).with_metadata(meta)
    tmp = out_parquet + ".tmp"
    n_rows = n_empty = 0
    pool = Pool(n_jobs) if n_jobs and n_jobs > 1 else None
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=chunk):
            tbl = pa.Table.from_batches([batch])
            smis = tbl[smiles_col].to_pylist()
            jobs = [(s, keep_dummy, radius, n_bits) for s in smis]
            bits = pool.map(_worker, jobs, chunksize=2000) if pool else [_worker(j) for j in jobs]
            bits = [b if b is not None else np.empty(0, np.uint16) for b in bits]
            n_empty += sum(len(b) == 0 for b in bits)
            lengths = np.array([len(b) for b in bits], dtype=np.int64)
            values = np.concatenate(bits) if len(bits) else np.empty(0, np.uint16)
            offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
            arr = pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), pa.array(values, pa.uint16()))
            writer.write_table(tbl.append_column(col, arr))
            n_rows += tbl.num_rows
    if pool:
        pool.close()
    os.replace(tmp, out_parquet)
    print(f"{n_rows:,} rows written with column {col!r} -> {out_parquet} | rows with an empty fingerprint: {n_empty:,}")


def ecfp4_settings(parquet):
    """Fingerprint settings stored in a feature table made by add_ecfp4_column."""
    m = {k.decode(): v.decode() for k, v in (pq.ParquetFile(parquet).schema_arrow.metadata or {}).items()}
    return dict(n_bits=int(m["ecfp4_n_bits"]), radius=int(m["ecfp4_radius"]), keep_dummy=bool(int(m["ecfp4_keep_dummy"])))


def tanimoto_column(bits_series, query_bits, n_bits=N_BITS):
    """
    Tanimoto of one query against a pandas Series / list of ON-bit arrays (e.g. df['ecfp4'] of a shortlist).
    Rows with an empty fingerprint give 0.
    """
    arrays = [np.asarray(b, dtype=np.int64) for b in bits_series]
    lengths = np.array([len(a) for a in arrays], dtype=np.int64)
    values = np.concatenate(arrays) if arrays else np.empty(0, np.int64)
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    X = sparse.csr_matrix((np.ones(len(values), dtype=np.float32), values.astype(np.int32), indptr),
                          shape=(len(arrays), n_bits))
    q = np.zeros(n_bits, dtype=np.float32)
    q[np.asarray(query_bits, dtype=np.int64)] = 1.0
    inter = X @ q
    den = lengths.astype(np.float32) + q.sum() - inter
    return np.divide(inter, den, out=np.zeros_like(inter), where=den > 0)
