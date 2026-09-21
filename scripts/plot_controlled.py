"""Plot saved controlled experiment data. Never runs training or evaluation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diayn.controlled_plots import generate_plots, MODES, KINDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'runs/controlled_ratios_v1', help='Controlled experiment output root')
    p.add_argument('--output', type=Path, help='New plot directory; must not already exist')
    p.add_argument('--ratios', nargs='+', help='For example: 1:1 1:5 5:1')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    p.add_argument('--kinds', nargs='+', choices=KINDS, default=KINDS)
    p.add_argument('--checkpoints', nargs='+', type=int, help='Evaluation steps to include; training plots keep all logs')
    p.add_argument('--diagnostic-step', type=int, help='Confusion/outcome checkpoint; default latest selected per run')
    p.add_argument('--change-steps', nargs=2, type=int, default=(150000,200000), metavar=('START','END'))
    p.add_argument('--formats', nargs='+', choices=('png','pdf','svg'), default=('png',))
    p.add_argument('--dpi', type=int, default=180)
    args = vars(p.parse_args())
    try:
        generate_plots(**args)
    except (ValueError, FileExistsError) as exc:
        p.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
