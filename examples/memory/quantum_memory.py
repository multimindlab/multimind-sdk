"""
Quantum memory status demonstration for MultiMind SDK.

QRAM, QAM, TopologicalMemory and QuantumClassicalHybridMemory are classical
*simulations* of quantum memory designs and are currently abstract scaffolds:
they do not implement the full BaseMemory contract, so constructing them
raises TypeError. This script demonstrates that honestly instead of
pretending they work. See docs/memory.md ("Scaffolds") for details.
"""

from multimind.memory import QAM, QRAM, QuantumClassicalHybridMemory, TopologicalMemory

SCAFFOLDS = [QRAM, QAM, TopologicalMemory, QuantumClassicalHybridMemory]


def main():
    print("Quantum memory classes are scaffolds; attempting to construct each one:\n")
    for cls in SCAFFOLDS:
        try:
            cls()
            print(f"{cls.__name__}: constructed (scaffold has been completed)")
        except TypeError as exc:
            print(f"{cls.__name__}: not instantiable -> {exc}")
    print("\nTo use one, subclass it and implement add_message/get_messages/clear/save/load.")


if __name__ == "__main__":
    main()
