import unittest

from rag_fusion.rrf import fuse_results_rrf


class FuseResultsRRFTests(unittest.TestCase):
    def test_documents_repeated_across_ranked_lists_are_promoted(self):
        ranked_lists = [
            ["document-a", "document-b"],
            ["document-b", "document-c"],
        ]

        fused_documents = fuse_results_rrf(ranked_lists, rrf_constant=0)

        self.assertEqual(fused_documents, ["document-b", "document-a", "document-c"])

    def test_documents_are_returned_once(self):
        ranked_lists = [["document-a", "document-a"]]

        fused_documents = fuse_results_rrf(ranked_lists, rrf_constant=0)

        self.assertEqual(fused_documents, ["document-a"])


if __name__ == "__main__":
    unittest.main()
