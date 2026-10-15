"""Rebuild release figures and tables."""

from scripts import generate_figures, generate_tables


def main() -> None:
    generate_tables.main()
    generate_figures.main()
    print("Reproduction complete: outputs/figures, outputs/tables")


if __name__ == "__main__":
    main()
