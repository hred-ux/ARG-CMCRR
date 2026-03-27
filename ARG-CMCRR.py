import argparse
import textwrap
import torch

parser = argparse.ArgumentParser(
    prog='ARG-CMCRR',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                             ARG-CMCRR                                        ║
    ║     Deep Learning-Based Classification and Risk Assessment of                ║
    ║                  Antibiotic Resistance Genes (ARGs)                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    SUPPORTED INPUT TYPES
    ─────────────────────
      • Full-length amino acid sequences     (FASTA, .aa)
      • Full-length nucleotide sequences     (FASTA, .nt)
      • Short amino acid reads               (FASTA, 30–50 aa)
      • Short nucleotide reads               (FASTA, 90–150 nt)

    USAGE
    ─────
      Full-length sequences:
        python ARG-CMCRR.py --input input_path_data --type aa/nt --length l  --outname output_file_name  --risk True/False
      Short reads:
        python ARG-CMCRR.py --input input_path_data --type aa/nt --length s  --outname output_file_name  --risk True/False

    ARGUMENTS
    ─────────
      -i,  --input      Path to the input FASTA file.
      -t,  --type       Molecule type of the input sequences.（aa  →  amino acid sequences； nt  →  nucleotide sequences）
      -l,  --length      Length category of the input sequences.
                          l  →  full-length sequences (aa: full-length | nt: full-length)
                          s  →  short reads           (aa: 30–50 aa    | nt: 90–150 nt)
      -on, --outname    Path for the output result file.
      -r,  --risk       (Optional) Predict the ARG risk level.
                          True   →  enable risk assessment
                          False  →  skip risk assessment
                        Only supported for full-length amino acid (aa) input.
        
    """))

parser.print_help()
parser.add_argument('-i', '--input', required=False, default='./fastafile/aa_long_test.fasta', help='the test data as input')
parser.add_argument('-t', '--type', required=False, default='aa', choices=['aa', 'nt'], help='molecular type of your input file')
parser.add_argument('-l', '--length', required=False, default='l', choices=['s', 'l'],
                    help="Output format: 's' (short sequence) or 'l' (long sequence). Default: 'l'")
parser.add_argument('-on', '--outfile', required=False, default='aa_long_output', help='the name of results output')
parser.add_argument('-r', '--risk', required=False, default='True',choices=['True', 'False'], help='the risk level of ARG')

args = parser.parse_args()

if args.type == 'aa' and args.length == 's':
    import argcmcrr_ssaa as ssaa
    ssaa.ssaa_predict(args.input, args.outfile)

if args.type == 'nt' and args.length == 's':
    import argcmcrr_ssnt as ssnt

    ssnt.ssnt_predict(args.input, args.outfile)

if args.type == 'aa' and args.length == 'l' and args.risk == 'True':
    import argcmcrr_lsaa_with_risk as lsaar
    
    lsaar.lsaa_predict_with_risk(args.input, args.outfile)

if args.type == 'aa' and args.length == 'l' and args.risk == 'False':
    import argcmcrr_lsaa as lsaa
    
    lsaa.lsaa_predict(args.input, args.outfile)

if args.type == 'nt' and args.length == 'l':
    import argcmcrr_lsnt as lsnt

    lsnt.lsnt_predict(args.input, args.outfile)

