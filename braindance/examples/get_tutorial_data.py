"""Download and verify the public tutorial: python -m braindance.examples.get_tutorial_data.

Use --validate to also check Recording loading, spike counts, stimulation logs,
electrode mapping and binning against the frozen scientific reference.
"""


def main(name="closed-loop-small", version=None, cache_dir=None, offline=False, update=False,
         manifest_path=None, validate=False):
    if validate:
        from braindance.examples.validate_tutorial_data import main as validate_data

        return validate_data(name=name, version=version, cache_dir=cache_dir,
                             offline=offline, update=update, manifest_path=manifest_path)

    from braindance.tutorial import get_test_data

    folder = get_test_data(name, version=version, cache_dir=cache_dir,
                           offline=offline, update=update, manifest_path=manifest_path)
    print(f"Tutorial data: {folder}")
    return folder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="closed-loop-small")
    parser.add_argument("--version")
    parser.add_argument("--cache-dir")
    parser.add_argument("--manifest-path")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--update", action="store_true", help="Update managed tutorial files, preserving user results")
    parser.add_argument("--validate", action="store_true")
    main(**vars(parser.parse_args()))
