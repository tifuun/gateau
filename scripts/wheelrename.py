#!/usr/bin/env python3

PYTHON_TAGS = [
        'cp39',
        'cp310',
        'cp311',
        'cp312',
        'cp313',
        ]
PLATFORM_TAGS = ['x86_64']

from pathlib import Path
import argparse
from copy import copy

try:
    from wheel_filename import WheelFilename
except ImportError as err:
    raise Exception("Please install `wheel-filename` python package.") from err


def main():
    parser = argparse.ArgumentParser(
        prog='wheelrename',
        description='Helper script to fix wheel filename for PyPI',
        )
    parser.add_argument(
        'filename',
        type=Path,
        nargs='+',
        help="List of wheel files to rename"
        )
    parser.add_argument(
        '-n',
        '--dry-run',
        action='store_true',
        help="Only print what would be done"
        )
    args = parser.parse_args()

    for file in args.filename:
        if not file.exists():
            raise Exception(f"File {file} doesnt exist.")
        name = WheelFilename.parse(file.name)
        new_name = copy(name)
        new_name.python_tags = PYTHON_TAGS
        new_name.platform_tags = PLATFORM_TAGS
        new_name = str(new_name)

        if new_name == file.name:
            print("no rename needed.")
            continue

        new_file = file.parent / new_name

        print(f'{file} -> {new_file}')
        if not args.dry_run:
            file.rename(new_file)

if __name__ == '__main__':
    main()

