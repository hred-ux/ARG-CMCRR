import torch
from utils import process_lsaa
from new_classifier_short import RNNClassifier
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier
import joblib
import tempfile
import subprocess
import os

AR_NAME_TO_LABEL = {'beta-lactam': 0, 'bacitracin': 1, 'multidrug': 2, 'MLS': 3, 'aminoglycoside': 4, 'polymyxin': 5,
                    'fosfomycin': 6,
                    'quinolone': 7, 'chloramphenicol': 8, 'tetracycline': 9, 'glycopeptide': 10, 'peptide': 11,
                    'sulfonamide': 12, 'trimethoprim': 13,
                    'novobiocin': 14, 'rifamycin': 15, 'nonarg': 16, 'others': 17}
ME_NAME_TO_LABEL = {
    'antibiotic inactivation': 0, 'antibiotic target alteration': 1, 'antibiotic efflux': 2,
    'antibiotic target replacement': 3,
    'antibiotic target protection': 4, 'others': 5
}

LABEL_TO_AR_NAME = {v: k for k, v in AR_NAME_TO_LABEL.items()}
LABEL_TO_ME_NAME = {v: k for k, v in ME_NAME_TO_LABEL.items()}

DEVICE = torch.device("cuda")
BATCH_SIZE = 32
BATCH_SIZE_RISK_EMD = 1
BATCH_SIZE_RISK = 16
VOCAB_SIZE = 22
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 18
NUM_MECHANISM = 6

model_class = RNNClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
)
model_mech = RNNClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_MECHANISM
)

# Load models
current_dir = os.path.dirname(os.path.abspath(__file__))
model_class_path = os.path.join(current_dir, "classptFile", "aa_long_class.pt")
model_class.load_state_dict(
    torch.load(model_class_path, map_location=DEVICE)
)
model_mech_path = os.path.join(current_dir, "mechptFile", "aa_long_mechanism.pt")
model_mech.load_state_dict(
    torch.load(model_mech_path, map_location=DEVICE)
)

model_class.to(DEVICE).eval()
model_mech.to(DEVICE).eval()

print("Loading ESM-2 model...")

esm_model_dir = os.path.join(current_dir, "models", "esm2_t33_650M")

if not os.path.exists(esm_model_dir):
    raise FileNotFoundError(
        f"ESM-2 model not found at {esm_model_dir}. "
        "Please download the model and place it in the 'models/' directory."
    )

tokenizer = AutoTokenizer.from_pretrained(esm_model_dir, do_lower_case=False)
model_esm = AutoModel.from_pretrained(esm_model_dir).to(DEVICE).eval()
model_esm = model_esm.half()

print("Loading XGBoost model...")
clf = XGBClassifier()
json_path = os.path.join(current_dir, "risk_modelfile", "xgb_esm2_650m.json")
pkl_path = os.path.join(current_dir, "risk_modelfile", "label_encoder.pkl")
clf.load_model(json_path)
le = joblib.load(pkl_path)


# Utils
@torch.no_grad()
def predict_with_model(model, features, lengths):
    logits = model(features, lengths)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return preds.cpu(), probs.cpu()


@torch.no_grad()
def get_embeddings_batch(seqs, batch_size=8):
    # Batch ESM-2 embedding
    all_embs = []

    for start in tqdm(
            range(0, len(seqs), batch_size),
            desc="ESM embedding",
            leave=False,
    ):
        batch_seqs = seqs[start:start + batch_size]
        batch_seqs = [" ".join(list(s)) for s in batch_seqs]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True
            # truncation=True
        ).to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model_esm(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
        all_embs.append(emb.cpu())

    del inputs, outputs, emb
    torch.cuda.empty_cache()
    X = torch.cat(all_embs).numpy()
    del all_embs
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return X


def write_multi_fasta(seqs, headers, out_fasta):
    with open(out_fasta, "w") as f:
        for h, s in zip(headers, seqs):
            f.write(f">{h}\n")
            f.write(s + "\n")


def parse_blast_best_hits(blast_out):
    # return dict: {qseqid: risk_label}
    result = {}

    with open(blast_out) as f:
        for line in f:
            qid, sid, *_ = line.strip().split("\t")
            if qid not in result:  # best hit
                parts = sid.split("|")
                result[qid] = parts[-1] if len(parts) > 1 else "NA"

    return result


def run_blast_multi(
        query_fasta,
        db_fasta,
        out_file,
        evalue=1e-3,
        max_target_seqs=1,
        max_hsps=1,
        num_threads=8,
):
    blastp_path = os.path.join(current_dir, "blast", "bin", "blastp")
  
    if not os.path.exists(blastp_path):
        raise FileNotFoundError(
            "BLAST executable not found.\n"
            "Please download NCBI BLAST+ and place it in:\n"
            "ARG_CMCRR/blast/bin/"
        )
    cmd = [
        blastp_path,
        "-query", query_fasta,
        "-db", db_fasta,
        "-outfmt", "6 qseqid sseqid bitscore evalue",
        "-evalue", str(evalue),
        "-max_target_seqs", str(max_target_seqs),
        "-max_hsps", str(max_hsps),
        "-num_threads", str(num_threads),
        "-out", out_file
    ]
    subprocess.run(cmd, check=True)


def blast_multi_fasta_risk(
        seqs,
        db_fasta,
        prefix="blast",
):
    # return seq risk label
    with tempfile.TemporaryDirectory() as tmpdir:
        query_fa = os.path.join(tmpdir, f"{prefix}_query.fasta")
        out_txt = os.path.join(tmpdir, f"{prefix}_out.txt")

        headers = [f"q_{i}" for i in range(len(seqs))]
        write_multi_fasta(seqs, headers, query_fa)

        run_blast_multi(
            query_fasta=query_fa,
            db_fasta=db_fasta,
            out_file=out_txt
        )

        hit_map = parse_blast_best_hits(out_txt)

        return [hit_map.get(f"q_{i}", "NA") for i in range(len(seqs))]


#  BLAST
def blast_in_batches(seqs, db_fasta, prefix, batch_size=BATCH_SIZE):
    blast_results = []
    for start in tqdm(range(0, len(seqs), batch_size),
                      desc=f"BLAST {prefix}", leave=False):
        batch_seqs = seqs[start:start + batch_size]
        batch_res = blast_multi_fasta_risk(batch_seqs,
                                           db_fasta=db_fasta,
                                           prefix=prefix)
        blast_results.extend(batch_res)
    return blast_results


def blast_all_at_once(arg_seqs, db_fasta, prefix):
    with tempfile.TemporaryDirectory() as tmpdir:
        query_fa = os.path.join(tmpdir, f"{prefix}.query.fasta")
        out_txt = os.path.join(tmpdir, f"{prefix}.out.txt")

        headers = [f"q_{i}" for i in range(len(arg_seqs))]
        write_multi_fasta(arg_seqs, headers, query_fa)

        run_blast_multi(
            query_fasta=query_fa,
            db_fasta=db_fasta,
            out_file=out_txt,
            num_threads=16
        )

        hit_map = parse_blast_best_hits(out_txt)
        return [hit_map.get(f"q_{i}", "NA") for i in range(len(arg_seqs))]


# Main prediction
def lsaa_predict_with_risk(input_file, outfile):
    stage1_file = outfile + ".stage1.csv"

    if os.path.exists(stage1_file):
        print(f"Found stage-1 file: {stage1_file}")
        print("Skipping ARG classification models, loading stage-1 results...")

        results_df = pd.read_csv(stage1_file)
        results = results_df.to_dict(orient="records")

        arg_indices = [
            i for i, r in enumerate(results)
            if r["Arg/non-Arg"] == "ARG"
        ]
        arg_seqs = [results[i]["sequence"] for i in arg_indices]

    else:

        print("Reading input file...")
        features, lengths, headers, seqs = process_lsaa(input_file)

        num_samples = features.size(0)

        all_preds_class, all_probs_class = [], []
        all_preds_mech, all_probs_mech = [], []

        print("Predicting ARG class...")
        for start in tqdm(
                range(0, num_samples, BATCH_SIZE),
                total=(num_samples + BATCH_SIZE - 1) // BATCH_SIZE,
                desc="Class inference"
        ):
            end = min(start + BATCH_SIZE, num_samples)

            feat_batch = features[start:end]
            len_batch = lengths[start:end]

            preds_c, probs_c = predict_with_model(
                model_class, feat_batch, len_batch
            )

            all_preds_class.append(preds_c)
            all_probs_class.append(probs_c)

        preds_class = torch.cat(all_preds_class)
        probs_class = torch.cat(all_probs_class)

        arg_indices = [
            i for i, pc in enumerate(preds_class)
            if LABEL_TO_AR_NAME[int(pc)] != "nonarg"
        ]

        arg_seqs = [seqs[i] for i in arg_indices]
        arg_features = features[arg_indices]
        arg_lengths = [lengths[i] for i in arg_indices]

        print(f"Predicting mechanism for {len(arg_indices)} ARG sequences...")

        all_preds_mech, all_probs_mech = [], []

        for start in tqdm(
                range(0, len(arg_indices), BATCH_SIZE),
                total=(len(arg_indices) + BATCH_SIZE - 1) // BATCH_SIZE,
                desc="Mechanism inference"
        ):
            end = min(start + BATCH_SIZE, len(arg_indices))

            feat_batch = arg_features[start:end]
            len_batch = arg_lengths[start:end]

            preds_m, probs_m = predict_with_model(
                model_mech, feat_batch, len_batch
            )

            all_preds_mech.append(preds_m)
            all_probs_mech.append(probs_m)

        preds_mech = torch.cat(all_preds_mech)
        probs_mech = torch.cat(all_probs_mech)

        results = []

        arg_pos = 0

        for i, (h, seq, pc, prob_c) in enumerate(
                zip(headers, seqs, preds_class, probs_class)
        ):
            pc = int(pc)
            class_name = LABEL_TO_AR_NAME[pc]

            if class_name == "nonarg":
                results.append({
                    "header": h,
                    "sequence": seq,
                    "Arg/non-Arg": "non-ARG",
                    "pred_class": "",
                    "probability_class": "",
                    "pred_mech": "",
                    "probability_mech": ""
                })
            else:
                pm = int(preds_mech[arg_pos])
                prob_m = probs_mech[arg_pos]
                mech_name = LABEL_TO_ME_NAME[pm]

                results.append({
                    "header": h,
                    "sequence": seq,
                    "Arg/non-Arg": "ARG",
                    "pred_class": class_name,
                    "probability_class": prob_c[pc].item(),
                    "pred_mech": mech_name,
                    "probability_mech": prob_m[pm].item()
                })

                arg_pos += 1

        # Save stage-1 results (before risk prediction)
        stage1_file = outfile + ".stage1.csv"

        pd.DataFrame(results).to_csv(stage1_file, index=False)
        print(f"Stage-1 results saved to {stage1_file}")

        del features, lengths
        del preds_class, probs_class
        del preds_mech, probs_mech
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Risk prediction (only ARG)
    print(f"Running risk prediction for {len(arg_seqs)} ARG sequences...")

    if len(arg_seqs) > 0:
        #  acquire esm embeddings
        X = get_embeddings_batch(arg_seqs, batch_size=BATCH_SIZE_RISK_EMD)

        #  predict
        all_pred_ids, all_pred_probs = [], []
        for start in tqdm(range(0, X.shape[0], BATCH_SIZE_RISK),
                          desc="Risk prediction", leave=False):
            X_batch = X[start:start + BATCH_SIZE_RISK]
            pred_ids_batch = clf.predict(X_batch)
            pred_probs_batch = clf.predict_proba(X_batch)
            all_pred_ids.extend(pred_ids_batch)
            all_pred_probs.extend(pred_probs_batch)

        pred_labels = le.inverse_transform(all_pred_ids)

        print(f"Start blast")
        fa1 = os.path.join(current_dir, "blastfile", "ARG-Risk-21.fasta")
        fa2 = os.path.join(current_dir, "blastfile", "ARG-Risk-22.fasta")
        blast21 = blast_all_at_once(arg_seqs, db_fasta=fa1,
                                    prefix="ARG-Risk-21")
        blast22 = blast_all_at_once(arg_seqs, db_fasta=fa2,
                                    prefix="ARG-Risk-22")

        print(f"Save results")

        for idx, label, prob, r21, r22 in zip(arg_indices, pred_labels, all_pred_probs, blast21, blast22):
            results[idx]["risk_label"] = label
            results[idx]["risk_probability"] = float(prob.max())
            results[idx]["blast_21_risk"] = r21
            results[idx]["blast_22_risk"] = r22

        for r in results:
            if r["Arg/non-Arg"] != "ARG":
                r["risk_label"] = ""
                r["risk_probability"] = ""
                r["blast_21_risk"] = ""
                r["blast_22_risk"] = ""

        del X
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Save results
    if not outfile.endswith(".csv"):
        outfile += ".csv"

    pd.DataFrame(results).to_csv(outfile, index=False)
    print(f"Results saved to {outfile}")

