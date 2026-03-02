import torch
from utils import process_ssaa
from new_classifier_short import RNNClassifier
import pandas as pd
from tqdm import tqdm
import os


AR_NAME_TO_LABEL = {'beta-lactam': 0, 'bacitracin': 1, 'multidrug': 2, 'MLS': 3, 'aminoglycoside': 4, 'polymyxin': 5,'fosfomycin': 6,
                    'quinolone': 7, 'chloramphenicol': 8, 'tetracycline': 9, 'glycopeptide': 10, 'peptide': 11,'sulfonamide': 12, 'trimethoprim': 13,
                    'novobiocin': 14, 'rifamycin': 15, 'nonarg':16, 'others':17}
ME_NAME_TO_LABEL = {
        'antibiotic inactivation': 0, 'antibiotic target alteration': 1, 'antibiotic efflux': 2, 'antibiotic target replacement': 3,
        'antibiotic target protection': 4, 'others': 5
    }

LABEL_TO_AR_NAME = {v: k for k, v in AR_NAME_TO_LABEL.items()}
LABEL_TO_ME_NAME = {v: k for k, v in ME_NAME_TO_LABEL.items()}


DEVICE = torch.device("cuda")
BATCH_SIZE = 64
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

current_dir = os.path.dirname(os.path.abspath(__file__))
model_class_path = os.path.join(current_dir, "classptFile", "aa_short_class.pt")
model_mech_path = os.path.join(current_dir, "mechptFile", "aa_short_mechanism.pt")
model_class.load_state_dict(torch.load(model_class_path,map_location=DEVICE))
model_mech.load_state_dict(torch.load(model_mech_path,map_location=DEVICE))
model_class.to(DEVICE)
model_mech.to(DEVICE)

def predict_with_model(model, features, lengths):
    model.eval()
    with torch.no_grad():
        logits = model(features, lengths)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
    return preds.cpu(), probs.cpu()

def ssaa_predict(input_file, outfile):
    print("Reading input file...")
    features, lengths, headers, seqs = process_ssaa(input_file)

    num_samples = features.size(0)

    all_preds_class, all_probs_class = [], []
    all_preds_mech,  all_probs_mech  = [], []

    print("Predicting ARG class...")
    for start in tqdm(
        range(0, num_samples, BATCH_SIZE),
        total=(num_samples + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Class inference"
    ):
        end = min(start + BATCH_SIZE, num_samples)

        feat_batch = features[start:end]
        len_batch  = lengths[start:end]

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
    arg_lengths  = [lengths[i] for i in arg_indices]

    print(f"Predicting mechanism for {len(arg_indices)} ARG sequences...")

    all_preds_mech, all_probs_mech = [], []

    for start in tqdm(
        range(0, len(arg_indices), BATCH_SIZE),
        total=(len(arg_indices) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Mechanism inference"
    ):
        end = min(start + BATCH_SIZE, len(arg_indices))

        feat_batch = arg_features[start:end]
        len_batch  = arg_lengths[start:end]

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
                "pred_class": class_name,
                "probability_class": prob_c[pc].item(),
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

    if not outfile.endswith(".csv"):
        outfile += ".csv"

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)

    print(f"results saved to {outfile}")






