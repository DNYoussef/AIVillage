# import pytest
# from mergekit.config import MergeKitConfig, ModelReference
# from mergekit.merger import MergeKitMerger

# def test_linear_merge():
#     config = MergeKitConfig(
#         merge_method="linear",
#         models=[ModelReference(name="model1"), ModelReference(name="model2")],
#         parameters={"weights": [0.7, 0.3]}
#     )
#     merger = MergeKitMerger(config)

#     # Mock models
#     models = [
#         {"param1": 1.0, "param2": 2.0},
#         {"param1": 2.0, "param2": 1.0}
#     ]

#     merged = merger.merge(models)

#     assert merged["param1"] == pytest.approx(1.3)
#     assert merged["param2"] == pytest.approx(1.7)

# Add more tests as needed

# Note: This test file is currently commented out as the 'mergekit' module is not part of the current RAG system implementation.
# It may be useful for future model merging functionality.
