import argparse
import textwrap
import torch

parser = argparse.ArgumentParser(
    prog='ARG_CMCRR',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
    ARG_CMCRR: Classification and risk assessment of antibiotic resistance genes based on deep learning.
   --------------------------------------------------------------------------------------------------------

    The input can be long amino acid sequences(full length), long nucleotide sequences, 
    short amino acid reads (30-50aa), short nucleotide reads (90-150nt) in fasta format.
    
    USAGE:
        for full-length
            python ARG_CMCRR.py --input input_path_data --type aa/nt --length l  --outname output_file_name  --risk True/False
        for short reads
            python ARG_CMCRR.py --input input_path_data --type aa/nt --length s  --outname output_file_name  --risk True/False

    general options:
        --input/-i    the test file as input
        --type/-t     molecular type of your test data (aa for amino acid, nt for nucleotide)
        --model/-m    the model you assign to make the prediction (argmcr-l for long sequences, argmcr-s for short reads) 
        --outname/-on  the output file name
        --risk/-r      the risk level of ARG only supports the input of long amino acid sequences.
        
    """))

parser.print_help()
parser.add_argument('-i', '--input', required=False, default='/hy-tmp/github_script/fastafile/aa_long_test.fasta', help='the test data as input')
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

