"""One command reproduces the fixed quality-v2 21/13-episode experiment."""
import argparse
from pathlib import Path
from prepare_data import ROOT,locate_data,prepare
from experiments import calibrate,evaluate,convergence_check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data')
    parser.add_argument('--output',default=str(ROOT/'results'))
    args=parser.parse_args();out=Path(args.output)
    episodes,data,audit=prepare(locate_data(args.data),out)
    print(f"Data ready: {audit['selected_train']} train / {audit['selected_test']} test, 30 seconds each",flush=True)
    calibrated=calibrate(data,out)
    synthetic,real,detail,summary=evaluate(data,calibrated,out)
    convergence=convergence_check(data,calibrated,out)
    from multi_reporting import render
    render(episodes,data,audit,synthetic,real,detail,summary,convergence,out)
    from validate_results import validate
    print(f"Validation: {validate(out)['checks_count']} checks passed",flush=True)
    print(summary[summary.split=='test'].to_string(index=False),flush=True)
    print(f'Done: {out}',flush=True)


if __name__=='__main__':
    main()
