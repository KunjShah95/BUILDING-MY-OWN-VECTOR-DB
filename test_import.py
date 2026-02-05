import sys
import traceback

try:
    import utils.hnsw_index as hnsw_module
    print("Module imported successfully")
    print(f"Module attributes: {dir(hnsw_module)}")
    
    from utils.hnsw_index import HNSWIndex
    print("HNSWIndex imported successfully")
except Exception as e:
    traceback.print_exc()
    print(f"\nError: {e}")
