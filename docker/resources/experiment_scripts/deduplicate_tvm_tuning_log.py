import argparse
import jsonlines

parser = argparse.ArgumentParser(
    description="Util tool to clean up tvm tuning logs (only keeping best performing schedules)."
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help="Path to the input script.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to the output script.",
)

args = parser.parse_args()

config_lines = []
with jsonlines.open(args.path) as reader:
    for obj in reader:
        config_lines.append(obj)

best_times = {}
best_configs = {}
for config in config_lines:
    config_key = config["i"][0][0] + "_" + config["i"][0][1]
    if config_key not in best_times or config["r"][0][0] < best_times[config_key]:
        best_times[config_key] = config["r"][0][0]
        best_configs[config_key] = config


with jsonlines.open(args.output, mode='w') as writer:
    for config_value in best_configs.values():
        writer.write(config_value)

