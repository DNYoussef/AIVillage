import sys

print("--- sys.path ---")
for p in sys.path:
    print(p)
print("----------------")

try:
    import babel
    print("\n--- Babel Import ---")
    print("Successfully imported babel.")
    print(f"Location: {babel.__file__}")
    print("--------------------")
except ImportError as e:
    print(f"\nFailed to import babel: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
