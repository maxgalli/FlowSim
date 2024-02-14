import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("nevents", type=int, help="Number of events to generate")
    parser.add_argument("output_path", type=str, help="Output file name")
    parser.add_argument("--ttbar", action="store_true", help="Generate ttbar events")
    parser.add_argument("--qcd", action="store_true", help="Generate qcd events")
    parser.add_argument("--diboson", action="store_true", help="Generate diboson WW events")
    parser.add_argument("--zjets", action="store_true", help="Generate Z+jets events")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    return parser.parse_args()